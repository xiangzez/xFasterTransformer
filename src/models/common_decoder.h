// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "INIReader.h"
#include "abstract_decoder.h"
#include "attention.h"
#include "debugger.h"
#include "decoder_layer.h"
#include "dist_linear.h"
#include "kvcache_manager.h"
#include "messenger.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "transpose_util.h"
#include "weight_util.h"

using namespace xft;

struct QKPO_Dummy {
    QKPO_Dummy(int dim) {}
    void forward(float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {}
};

// Template parameters:
// ATTN_CLS - class for attention impl.
// MLP_CLS - MLP implementation
// KVCacheT - data type of the cached keys/values
// ATTN_MLP_PARALLEL - true means attention and MLP are in parallel, using the same initial input
template <typename ATTN_CLS, typename MLP_CLS, typename KVCacheT = float16_t, bool ATTN_MLP_PARALLEL = false>
class CommonDecoder : public AbstractDecoder {
public:
    CommonDecoder(const std::string &modelPath, const std::string &modelType)
        : messenger(Messenger::getInstance())
#ifdef DEBUG
        , dbg("model_decoder.csv")
#endif
    {
        std::string configPath = modelPath + "/config.ini";
        INIReader reader = INIReader(configPath);
        wType = getWeightType(configPath, modelType);

        const int attHeadNum = reader.GetInteger(modelType, "head_num");
        // Use the same head number for the default multi-head attention
        const int kvHeadNum = reader.GetInteger(modelType, "kv_head_num", attHeadNum);
        const int size_per_head = reader.GetInteger(modelType, "size_per_head");
        const int imSize = reader.GetInteger(modelType, "inter_size");
        const int layers = reader.GetInteger(modelType, "num_layer");
        const int vocabSize = reader.GetInteger(modelType, "vocab_size");
        // Use 2k as default value
        const int maxPositions = reader.GetInteger(modelType, "max_pos_seq_len", 2048);
        const int hiddenSize = attHeadNum * size_per_head;
        const int embeddingSize = hiddenSize;
        const int multi_query_group_num = reader.GetInteger(modelType, "multi_query_group_num", attHeadNum);
        const float epsilon = reader.GetFloat(modelType, "layernorm_eps", 1e-6);

        std::string act = reader.Get(modelType, "activation_type");
        std::transform(act.begin(), act.end(), act.begin(), ::tolower);

        this->startId = reader.GetInteger(modelType, "start_id", 0);
        this->endId = reader.GetInteger(modelType, "end_id", startId);

        this->initSeqLen = 0;
        this->accSeqLen = 0;

        // Quantization config
        const bool quant_decoder_weights = reader.GetBoolean(modelType, "quant_decoder_weights", false);
        const int quant_wbits = reader.GetInteger(modelType, "quant_wbits", 8);
        const int quant_groupsize = reader.GetInteger(modelType, "quant_groupsize", -1);

        // Buffer related (not initialized)
        this->inputTokens = nullptr;
        this->maskSize = 0;
        this->attnMask = nullptr;
        embBuf.reset(new hpj::Matrix<float>());
        outBuf.reset(new hpj::Matrix<float>());

        // Context
        DecoderContext *ctx = getDecoderContext(layers, hiddenSize, attHeadNum, kvHeadNum, imSize, act, epsilon,
                vocabSize, embeddingSize, maxPositions);

        // Decoder
        for (int i = 0; i < layers; ++i) {
            auto pdec = new DECODER(ctx, i);
            this->setDecoderWeights(pdec, modelPath, i, quant_decoder_weights);
            this->decoders.push_back(pdec);
        }

        // Predictor
        int workers = messenger.getSize();
        int rank = messenger.getRank();
        this->predictor = new DistLinear<float16_t>(hiddenSize, vocabSize, rank, workers);
        this->setPredictorWeight(modelPath);

        // KVCache Manager
        this->kvCacheMgr.reset(new KVCacheManager<KVCacheT>(layers));
    }

    virtual ~CommonDecoder() {
        if (this->inputTokens) free(this->inputTokens);
        if (this->attnMask) free(this->attnMask);

        delete this->predictor;

        for (auto dec : this->decoders) {
            delete dec;
        }
    }

    std::tuple<float *, int, int> forward(int *ids, int64_t *dims, int step, bool logitsAll = false) {
        // Assume input has been synced with master in higher level.
        // Assume the 1st step input's shape is [userSideBS][1][seqLen].
        TimeLine t("Decoder.forward");
        TimeLine t1("Decoder.embedding");

        int userSideBS = dims[0];
        int beamSize = dims[1];
        int batchSize = (step == 0 ? userSideBS : userSideBS * beamSize); // as samples are duplicated at step 0
        int seqLen = dims[2];

        // Prepare context
        DecoderContext *ctx = this->getContext();
        ctx->resize(batchSize, seqLen, (step == 0 ? 0 : this->accSeqLen));

        if (step == 0) {
            // Enlarge buffer if needed
            prepareBuffers(ctx, userSideBS, beamSize, logitsAll);

            // Reset initial and accumulated sequence length at the first step
            this->initSeqLen = seqLen;
            this->accSeqLen = 0;
        }

        // Embedding
        this->embeddingForward(ids, this->embBuf->Data(), batchSize, seqLen);
        this->accSeqLen += seqLen;

        // Prepare attention mask
        this->prepareAttnMask(ids, step);

        // Token position ids, note: different models may have different impl.
        int *positionIds = this->getPositionIds(ids, batchSize, seqLen, step);
        t1.release();

        // Decoder: forward
        int hiddenSize = ctx->hiddenSize;
        for (int i = 0; i < this->decoders.size(); ++i) {
            int workers = this->messenger.getSize();
            KVCacheTensor<KVCacheT> &presentKey = this->kvCacheMgr->getKey(i);
            KVCacheTensor<KVCacheT> &presentValue = this->kvCacheMgr->getValue(i);

            this->decoders[i]->forwardAttention(getContext(), this->embBuf->Data(), this->outBuf->Data(), attnMask,
                    presentKey, // presentKey,
                    presentValue, // presentValue,
                    seqLen, // inputSeqLen,
                    this->accSeqLen - seqLen, // pastSeqLen
                    step == 0, // useSelfAttn,
                    true, // doLnBefore,
                    false, // returnAttn,
                    false, // returnKVs
                    false, // forPT
                    positionIds);

            // Expand the KV cache as it only has values for beam 0
            if (step == 0 && beamSize > 1) { this->kvCacheMgr->expandCache(i, userSideBS, beamSize, seqLen); }

            auto &attnOut = this->getContext()->tmpBuf;

            // Merge the result of attention
            // When attention and FFN/MLP are in parallel, do not need to reduce after attention
            if constexpr (!ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->messenger.reduceAdd(attnOut.Data(), attnOut.Data(), batchSize * seqLen * attnOut.Stride());
                }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(
                            getContext(), this->embBuf->Data(), this->outBuf->Data(), hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(
                            this->outBuf->Data(), this->embBuf->Data(), batchSize * seqLen * hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(
                            getContext(), this->embBuf->Data(), this->embBuf->Data(), hiddenSize, hiddenSize, true);
                }
            } else {
                // FFN (for multiple workers, output into outBuf and then reduce add to embBuf)
                if (this->messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(
                            getContext(), attnOut.Data(), this->outBuf->Data(), attnOut.Stride(), hiddenSize, true);
                    this->messenger.reduceAdd(
                            this->outBuf->Data(), this->embBuf->Data(), batchSize * seqLen * hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(
                            getContext(), attnOut.Data(), this->embBuf->Data(), attnOut.Stride(), hiddenSize, true);
                }
            }
        }

        // Prepare input for final Layer Norm (only care about the last row of the result)
        // Shape of embBuf: (bs, seqLen, hiddenSize)
        float *lnIn = this->embBuf->Data();
        if (seqLen > 1 && !logitsAll) { // copy is not needed when seqLen = 1 or logitsAll is true
            lnIn = this->outBuf->Data();
#pragma omp parallel for
            for (int b = 0; b < batchSize; ++b) {
                memcpy(lnIn + b * hiddenSize, this->embBuf->Data() + ((b + 1) * seqLen - 1) * hiddenSize,
                        hiddenSize * sizeof(float));
            }
        }

#ifdef DEBUG
        dbg.debugPrint("LayerNorm In:\n");
        dbg.dumpMatrix(lnIn, batchSize, hiddenSize, hiddenSize);
#endif

        // LN, as it supports inplace computing, input and output can be the same
        float *lnOut = this->embBuf->Data();
        if (!logitsAll)
            lastLayerNormForward(lnIn, lnOut, batchSize);
        else
            lastLayerNormForward(lnIn, lnOut, batchSize * seqLen);

#ifdef DEBUG
        dbg.debugPrint("LayerNorm Out:\n");
        dbg.dumpMatrix(lnOut, batchSize, hiddenSize, hiddenSize);
#endif

        // Predictor
        if (!logitsAll)
            this->predictor->forward(lnOut, this->outBuf->Data(), batchSize);
        else
            this->predictor->forward(lnOut, this->outBuf->Data(), batchSize * seqLen);

#ifdef DEBUG
        auto splitSize = this->predictor->getSplitSize();
        dbg.debugPrint("outBuf:\n");
        dbg.dumpMatrix(outBuf->Data(), batchSize, splitSize, splitSize);
#endif

        // Expand the result to make it cover multiple beams
        if (step == 0 && beamSize > 1) {
            const int splitSize = this->predictor->getSplitSize();
            for (int b = userSideBS - 1; b >= 0; --b) {
                float *src = this->outBuf->Data() + b * splitSize;
#pragma omp parallel for
                for (int idx = b * beamSize; idx < (b + 1) * beamSize; ++idx) {
                    if (idx == b) { continue; }
                    float *dst = this->outBuf->Data() + idx * splitSize;
                    memcpy(dst, src, splitSize * sizeof(float));
                }
            }
        }

        return std::tuple<float *, int, int>(
                this->outBuf->Data(), this->predictor->getSplitOffset(), this->predictor->getSplitSize());
    }

    // Reorder cached keys and values, size=batchSize*beamSize
    void reorderCache(int *idx, int size) { kvCacheMgr->reorderCache(idx, size, initSeqLen, accSeqLen); }

    // Get decoder context
    DecoderContext *getContext() { return context.get(); }

    // How many layers
    int getLayers() { return decoders.size(); }

    Messenger &getMessenger() { return messenger; }

    int getRank() { return messenger.getRank(); }

    WDataType getDataType() { return wType; }

    int getEndId() { return endId; }

    int getInitSeqLen() { return initSeqLen; }

    std::tuple<std::shared_ptr<DecoderContext>, std::shared_ptr<KVCacheManager<KVCacheT>>,
            std::shared_ptr<hpj::Matrix<float>>, std::shared_ptr<hpj::Matrix<float>>>
    getSharedResources() {
        return std::make_tuple(context, kvCacheMgr, embBuf, outBuf);
    }

    void setSharedResources(const std::tuple<std::shared_ptr<DecoderContext>, std::shared_ptr<KVCacheManager<KVCacheT>>,
            std::shared_ptr<hpj::Matrix<float>>, std::shared_ptr<hpj::Matrix<float>>> &r) {
        this->context = std::get<0>(r);
        this->kvCacheMgr = std::get<1>(r);
        this->embBuf = std::get<2>(r);
        this->outBuf = std::get<3>(r);
    }

    // When first step is skipped, call this function to make everything aligned
    void skipFirstStep(int initSeqLen) {
        // Reset initial and accumulated sequence length at the first step
        this->initSeqLen = initSeqLen;
        this->accSeqLen = initSeqLen;
    }

protected:
    using DECODER = Decoder<ATTN_CLS, MLP_CLS>;

    static bool fileExists(const std::string &filename) {
        std::ifstream file(filename);
        return file.good();
    }

    DecoderContext *getDecoderContext(int layers, const int hiddenSize, const int attHeadNum, const int kvHeadNum,
            const int imSize, const std::string &act, const float epsilon, int vocabSize, int embeddingSize,
            int maxPositions) {
        int splits = messenger.getSize();
        int splitIdx = messenger.getRank();

        if (context != nullptr) {
            if (context->hiddenSize == hiddenSize && context->attHeadNum == attHeadNum
                    && context->kvHeadNum == kvHeadNum && context->intermediateSize == imSize
                    && context->splitIdx == splitIdx) {
                return context.get();
            } else {
                printf("Different context size not unsupported!\n");
                exit(-1);
            }
        } else {
            this->context.reset(new DecoderContext(layers, hiddenSize, attHeadNum, kvHeadNum, imSize, act, epsilon,
                    vocabSize, embeddingSize, maxPositions, splitIdx, splits));
        }

        return this->context.get();
    }

    void setDecoderWeights(DECODER *pdecoder, const std::string &modelPath, int layerIdx, bool quant) {
        const int hiddenSize = getContext()->hiddenSize;
        const int imSize = getContext()->intermediateSize;
        const int kvHeadNum = getContext()->kvHeadNum;
        const int attHeadSize = getContext()->attHeadSize;
        const int mlpFactor = (getContext()->actType == DecoderContext::SWIGLU) ? 2 : 1;
        int qSize = hiddenSize;
        int kvSize = attHeadSize * kvHeadNum;
        int qkvSize = qSize + kvSize + kvSize;

#define ALLOC(size, alignment) aligned_alloc((alignment), (size))
        float *qkvWeight = nullptr;
        int8_t *qkvQWeight = nullptr;
        float *qkvScales = nullptr;
        float *qkvZeros = nullptr;
        float *qkvBias = (float *)ALLOC(qkvSize * sizeof(float), 64);

        float *attnOutWeight = nullptr;
        int8_t *attnOutQWeight = nullptr;
        float *attnOutScales = nullptr;
        float *attnOutZeros = nullptr;
        float *attnOutBias = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        float *fc1Weight = nullptr;
        int8_t *fc1QWeight = nullptr;
        float *fc1Scales = nullptr;
        float *fc1Zeros = nullptr;
        float *fc1Bias = (float *)ALLOC(imSize * sizeof(float), 64);

        float *fc2Weight = nullptr;
        int8_t *fc2QWeight = nullptr;
        float *fc2Scales = nullptr;
        float *fc2Zeros = nullptr;
        float *fc2Bias = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        float *ln1Gamma = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln1Beta = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln2Gamma = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln2Beta = (float *)ALLOC(hiddenSize * sizeof(float), 64);

        float *fc3Weight = nullptr;
        int8_t *fc3QWeight = nullptr;
        float *fc3Scales = nullptr;
        float *fc3Zeros = nullptr;

        if (quant) {
            // INT8 quant, wbits = 8, qweight dtype: int8
            qkvQWeight = (int8_t *)ALLOC(hiddenSize * qkvSize * sizeof(int8_t), 64);
            qkvZeros = (float *)ALLOC(qkvSize * sizeof(float), 64);
            qkvScales = (float *)ALLOC(qkvSize * sizeof(float), 64);

            attnOutQWeight = (int8_t *)ALLOC(hiddenSize * hiddenSize * sizeof(int8_t), 64);
            attnOutZeros = (float *)ALLOC(hiddenSize * sizeof(float), 64);
            attnOutScales = (float *)ALLOC(hiddenSize * sizeof(float), 64);

            fc1QWeight = (int8_t *)ALLOC(hiddenSize * imSize * mlpFactor * sizeof(int8_t), 64);
            fc1Zeros = (float *)ALLOC(imSize * mlpFactor * sizeof(float), 64);
            fc1Scales = (float *)ALLOC(imSize * mlpFactor * sizeof(float), 64);

            fc2QWeight = (int8_t *)ALLOC(hiddenSize * imSize * sizeof(int8_t), 64);
            fc2Zeros = (float *)ALLOC(imSize * sizeof(float), 64);
            fc2Scales = (float *)ALLOC(imSize * sizeof(float), 64);

            // printf("hiddenSize=%d, qkvSize=%d\n", hiddenSize, qkvSize);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.qweight.0.bin",
                    qkvQWeight, hiddenSize * qkvSize, WDataType::INT8);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.qzeros.0.bin",
                    qkvZeros, qkvSize, WDataType::FP32);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.scales.0.bin",
                    qkvScales, qkvSize, WDataType::FP32);

            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.qweight.0.bin",
                    attnOutQWeight, hiddenSize * hiddenSize, WDataType::INT8);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.qzeros.0.bin",
                    attnOutZeros, hiddenSize, WDataType::FP32);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.scales.0.bin",
                    attnOutScales, hiddenSize, WDataType::FP32);

            // Stardard 2 layer MLP
            if (fileExists(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.qweight.0.bin")) {
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.qweight.0.bin",
                        fc1QWeight, hiddenSize * imSize * mlpFactor, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.qzeros.0.bin",
                        fc1Zeros, imSize * mlpFactor, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.scales.0.bin",
                        fc1Scales, imSize * mlpFactor, WDataType::FP32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.qweight.0.bin",
                        fc2QWeight, hiddenSize * imSize, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.qzeros.0.bin",
                        fc2Zeros, hiddenSize, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.scales.0.bin",
                        fc2Scales, hiddenSize, WDataType::FP32);
            }
            // gate, up, down weights for Llama like model
            else {
                fc3QWeight = (int8_t *)ALLOC(hiddenSize * imSize * sizeof(int8_t), 64);
                fc3Zeros = (float *)ALLOC(hiddenSize * sizeof(float), 64);
                fc3Scales = (float *)ALLOC(hiddenSize * sizeof(float), 64);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.qweight.0.bin",
                        fc1QWeight, hiddenSize * imSize * mlpFactor, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.qzeros.0.bin",
                        fc1Zeros, imSize * mlpFactor, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.scales.0.bin",
                        fc1Scales, imSize * mlpFactor, WDataType::FP32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.qweight.0.bin", fc2QWeight,
                        hiddenSize * imSize, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.qzeros.0.bin", fc2Zeros,
                        imSize, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.scales.0.bin", fc2Scales,
                        imSize, WDataType::FP32);

                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.qweight.0.bin",
                        fc3QWeight, hiddenSize * imSize, WDataType::INT8);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.qzeros.0.bin",
                        fc3Zeros, hiddenSize, WDataType::FP32);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.scales.0.bin",
                        fc3Scales, hiddenSize, WDataType::FP32);
            }

        } else {
            qkvWeight = (float *)ALLOC(hiddenSize * qkvSize * sizeof(float), 64);
            attnOutWeight = (float *)ALLOC(hiddenSize * hiddenSize * sizeof(float), 64);
            fc1Weight = (float *)ALLOC(hiddenSize * imSize * mlpFactor * sizeof(float), 64);
            fc2Weight = (float *)ALLOC(hiddenSize * imSize * sizeof(float), 64);

            // printf("hiddenSize=%d, qkvSize=%d\n", hiddenSize, qkvSize);
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.weight.0.bin",
                    qkvWeight, hiddenSize * qkvSize, getDataType());
            loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.weight.0.bin",
                    attnOutWeight, hiddenSize * hiddenSize, getDataType());

            // Stardard 2 layer MLP
            if (fileExists(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin")) {
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor, getDataType());
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.weight.0.bin",
                        fc2Weight, hiddenSize * imSize, getDataType());
            }
            // gate, up, down weights for Llama like model
            else {
                fc3Weight = (float *)ALLOC(hiddenSize * imSize * sizeof(float), 64);
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.weight.0.bin",
                        fc1Weight, hiddenSize * imSize * mlpFactor, getDataType());
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.weight.0.bin", fc2Weight,
                        hiddenSize * imSize, getDataType());
                loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.weight.0.bin",
                        fc3Weight, hiddenSize * imSize, getDataType());
            }
        }

        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.weight.bin", ln1Gamma,
                hiddenSize, getDataType());
        loadWeight(modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.weight.bin",
                ln2Gamma, hiddenSize, getDataType());

#define READ_OPTIONAL(filename, addr, size, errmsg)                             \
    {                                                                           \
        int ret = loadWeight((filename), (addr), (size), getDataType(), false); \
        if (ret == 0) {                                                         \
            free(addr);                                                         \
            addr = nullptr;                                                     \
        } else {                                                                \
            if (ret != (size)) {                                                \
                printf("%s\n", (errmsg));                                       \
                exit(-1);                                                       \
            }                                                                   \
        }                                                                       \
    }

        // The bias is optional
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.bias.0.bin",
                qkvBias, qkvSize, "read QKV bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.bias.bin",
                attnOutBias, hiddenSize, "read attn dense bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.bias.bin", ln1Beta,
                hiddenSize, "read LN1 beta error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.bias.bin",
                ln2Beta, hiddenSize, "read LN2 beta error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.bias.0.bin",
                fc1Bias, imSize, "read FC1 bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.bias.bin", fc2Bias,
                hiddenSize, "read FC2 bias error");

#define FREE(x) \
    if ((x)) free((x))
        // Need the tranposed weights in our interface
        // ordering, trans, rows, cols, alpha, a, lda, b, ldb

        if (quant) {
            std::vector<void *> params = {qkvQWeight, qkvZeros, qkvScales, qkvBias,
                    qkvQWeight + qSize, qkvZeros + qSize, qkvScales + qSize, qkvBias + qSize,
                    qkvQWeight + qSize + kvSize, qkvZeros + qSize + kvSize, qkvScales + qSize + kvSize, qkvBias + qSize + kvSize,
                    attnOutQWeight, attnOutZeros, attnOutScales, attnOutBias,
                    ln1Gamma, ln1Beta,
                    fc1QWeight, fc1Zeros, fc1Scales, fc1Bias,
                    fc2QWeight, fc2Zeros, fc2Scales, fc2Bias,
                    ln2Gamma, ln2Beta,
                    fc3QWeight, fc3Zeros, fc3Scales};
            pdecoder->setQWeights(getContext(), params, false);
        } else {
            std::vector<float *> params = {qkvWeight, qkvBias, qkvWeight + qSize, qkvBias + qSize,
                    qkvWeight + qSize + kvSize, qkvBias + qSize + kvSize, attnOutWeight, attnOutBias, ln1Gamma, ln1Beta,
                    fc1Weight, fc1Bias, fc2Weight, fc2Bias, ln2Gamma, ln2Beta, fc3Weight};
            pdecoder->setWeights(getContext(), params, false);
        }
        FREE(qkvWeight);
        FREE(attnOutWeight);
        FREE(fc1Weight);
        FREE(fc2Weight);
        FREE(fc3Weight);
        FREE(qkvQWeight);
        FREE(attnOutQWeight);
        FREE(fc1QWeight);
        FREE(fc2QWeight);
        FREE(fc3QWeight);
        FREE(qkvZeros);
        FREE(attnOutZeros);
        FREE(fc1Zeros);
        FREE(fc2Zeros);
        FREE(fc3Zeros);
        FREE(qkvScales);
        FREE(attnOutScales);
        FREE(fc1Scales);
        FREE(fc2Scales);
        FREE(fc3Scales);
        FREE(qkvBias);
        FREE(attnOutBias);
        FREE(fc1Bias);
        FREE(fc2Bias);
        FREE(ln1Gamma);
        FREE(ln1Beta);
        FREE(ln2Gamma);
        FREE(ln2Beta);
    }

    void setPredictorWeight(const std::string &modelPath) {
        int inputSize = predictor->getInputSize();
        int outputSize = predictor->getOutputSize();

        float *weight = (float *)malloc(inputSize * outputSize * sizeof(float));
        float *bias = nullptr;

        loadWeight(modelPath + "/model.lm_head.weight.bin", weight, inputSize * outputSize, this->getDataType());

        predictor->setWeight(weight, bias);

        free(weight);
    }

    virtual void prepareBuffers(DecoderContext *ctx, int userSideBS, int beamSize, bool logitsAll = false) {
        int batchSize = ctx->batchSize;
        int hiddenSize = ctx->hiddenSize;
        int seqLen = ctx->inputSeqLen;
        int vocabSize = ctx->vocabSize;
        int maxPositions = ctx->maxPositions;
        int layers = this->decoders.size();
        int workers = this->messenger.getSize();

        // Prepare buffers (embBuf & outBuf), userSideBS * beamSize is the output rows really needed
        int logitsLen = logitsAll ? batchSize * seqLen : userSideBS * beamSize;
        int requiredRows = batchSize * seqLen;

        // The required output buffer size is bigger than the embedding size
        if (logitsLen * vocabSize > batchSize * seqLen * hiddenSize) {
            requiredRows = logitsLen * vocabSize / hiddenSize + 1;
        }
        if (requiredRows > this->embBuf->Rows()) {
            this->embBuf->Resize(requiredRows, hiddenSize);
            this->outBuf->Resize(requiredRows, hiddenSize);
        }

        // Attention mask
        int sizeRequired = batchSize * seqLen * seqLen;
        getAttnMask(sizeRequired);

        // Cached keys/values
        // The maximum sequence length is to be the same as maxPositions, at most
        // And the cache always needs to account for beam size
        int headsPerSplit = (ctx->kvHeadNum + workers - 1) / workers;
        this->kvCacheMgr->resize(maxPositions, userSideBS * beamSize, headsPerSplit, ctx->attHeadSize);
    }

    float *getAttnMask(int sizeRequired) {
        if (this->maskSize < sizeRequired) {
            if (this->attnMask) free(this->attnMask);
            this->attnMask = (float *)aligned_alloc(64, sizeRequired * sizeof(float));
            this->maskSize = sizeRequired;
        }
        return this->attnMask;
    }

    int getStartId() { return startId; }

    virtual void embeddingForward(int *ids, float *output, int batchSize, int seqLen) = 0;
    virtual void lastLayerNormForward(float *input, float *output, int rows) = 0;
    virtual void prepareAttnMask(int *ids, int step) = 0;

public:
    virtual int *getPositionIds(int *ids, int batchSize, int seqLen, int step) { return nullptr; }

protected:
    // For communication
    Messenger &messenger;

    // Execution context
    std::shared_ptr<DecoderContext> context;

    // The initial input sequence length, which is the prompt token size
    int initSeqLen;
    // Accumulated sequence length, = past_seq_len + current_seq_len
    int accSeqLen;

    // If not the master, need to receive token IDs from the master
    int *inputTokens;

    std::shared_ptr<KVCacheManager<KVCacheT>> kvCacheMgr;

    std::shared_ptr<hpj::Matrix<float>> embBuf; // used to store the embedding result
    std::shared_ptr<hpj::Matrix<float>> outBuf; // output buffer for decoder layers, same size as embBuf

protected:
    // Components most LLMs may use
    std::vector<DECODER *> decoders;
    DistLinear<float16_t> *predictor;

private:
    int maskSize; // size of allocated attnMask
    float *attnMask; // attention mask, set as private as may need to enlarge

    int startId;
    int endId;

    WDataType wType;
#ifdef DEBUG
    Debugger dbg;
#endif
};
