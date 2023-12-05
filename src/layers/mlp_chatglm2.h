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
#include <cmath>
#include "mlp_llama.h"

template <typename WeiT, typename NORM_CLS, bool INPUT_AS_RESID>
class ChatGLM2MLP : public LlamaMLP<WeiT> {
public:
    ChatGLM2MLP(DecoderContext *ctx) : LlamaMLP<WeiT>(ctx) {}

    // The inerface is for PyTorch, thus the weights are already transposed
    void setWeights(DecoderContext *ctx, std::vector<float *> &params, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int intermediateSize = ctx->intermediateSize;

        const float *gate_upW = params[0];
        const float *downW = params[2];
        const float *normW = params[4];

        REQUIRES(ctx->actType == DecoderContext::SWIGLU, "unsupported activation.");

        // Vertically split the gate weight and up weight
        hpj::Matrix<WeiT> convertedGateWeight, convertedUpWeight, convertedDownWeight;

        auto range = SplitUtil::getTaskRange(intermediateSize, ctx->numSplit, ctx->splitIdx);
        int colSplit = range.second - range.first;
        float *gateW = (float *)malloc(hiddenSize * colSplit * sizeof(float));
        float *upW = (float *)malloc(hiddenSize * colSplit * sizeof(float));
        if (trans) {
            int blockSize = colSplit * hiddenSize;
            memcpy(gateW, gate_upW + ctx->splitIdx * blockSize, blockSize * sizeof(float));
            memcpy(upW, gate_upW + intermediateSize * hiddenSize + ctx->splitIdx * blockSize,
                    blockSize * sizeof(float));
        } else {
            const float *weightPTR = gate_upW;
            for (int i = 0; i < hiddenSize; i++) {
                memcpy(gateW + i * colSplit, weightPTR + ctx->splitIdx * colSplit, colSplit * sizeof(float));
                weightPTR += intermediateSize;
                memcpy(upW + i * colSplit, weightPTR + ctx->splitIdx * colSplit, colSplit * sizeof(float));
                weightPTR += intermediateSize;
            }
        }

        this->gateWeight.set(trans, hiddenSize, colSplit, gateW);
        this->upWeight.set(trans, hiddenSize, colSplit, upW);

        free(gateW);
        free(upW);

        // Horizontally split the down weight
        this->downWeight.set(ctx, trans, intermediateSize, hiddenSize, downW, nullptr, false);

#ifdef DEBUG
        this->dbg.debugPrint("gateWeight packed weight: [%d, %d] (%d)\n", this->gateWeight.weight.Rows(),
                this->gateWeight.weight.Cols(), this->gateWeight.weight.Stride());
        this->dbg.dumpMatrix(this->gateWeight.weight);

        this->dbg.debugPrint("upWeight packed weight: [%d, %d] (%d)\n", this->upWeight.weight.Rows(),
                this->upWeight.weight.Cols(), this->upWeight.weight.Stride());
        this->dbg.dumpMatrix(this->upWeight.weight);

        this->dbg.debugPrint("downWeight packed weight: [%d, %d] (%d)\n", this->downWeight.weight.Rows(),
                this->downWeight.weight.Cols(), this->downWeight.weight.Stride());
        this->dbg.dumpMatrix(this->downWeight.weight);
#endif

        // norm.setWeight(normW, NULL, hiddenSize);
        if (normW) {
            this->normWeight.Resize(hiddenSize);
            memcpy(this->normWeight.Data(), normW, sizeof(float) * hiddenSize);
        }
    }
};
