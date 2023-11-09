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

#include <immintrin.h>
#include <omp.h>

#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <string>

#include "attention.h"
#include "debugger.h"
#include "kvcache_tensor.h"
#include "timeline.h"

template <typename ATTN_CLS, typename MLP_CLS>
class Decoder {
public:
    Decoder(DecoderContext *_ctx, int _layerIdx)
        : layerIdx(_layerIdx)
        , attn(_layerIdx, _ctx)
        , mlp(_ctx)
#ifdef DEBUG
        , dbg(Debugger::formatStr("%d_%d.csv", _layerIdx, _ctx->splitIdx))
#endif
    {
#ifdef DEBUG
        attn.setDebugger(dbg);
        mlp.setDebugger(dbg);
#endif
    }

    virtual ~Decoder() {}

    int getLayerId() { return layerIdx; }

    void setWeights(DecoderContext *ctx, std::vector<float *> &params, bool trans = true) {
        const float *queryWeight = params[0];
        const float *queryBias = params[1];
        const float *keyWeight = params[2];
        const float *keyBias = params[3];
        const float *valueWeight = params[4];
        const float *valueBias = params[5];
        const float *attnOutWeight = params[6];
        const float *attnOutBias = params[7];
        const float *gamma1 = params[8];
        const float *beta1 = params[9];

        attn.setWeights(ctx, queryWeight, queryBias, keyWeight, keyBias, valueWeight, valueBias, attnOutWeight,
                attnOutBias, gamma1, beta1, trans);

        std::vector<float *> mlpParams(params.begin() + 10, params.end());
        mlp.setWeights(ctx, mlpParams, trans);
    }

    void setQWeights(DecoderContext *ctx, std::vector<void *> &params, bool trans = true) {
        const int8_t *queryQWeight = (const int8_t *)params[0];
        const float *queryZeros = (const float *)params[1];
        const float *queryScales = (const float *)params[2];
        const float *queryBias = (const float *)params[3];

        const int8_t *keyQWeight = (const int8_t *)params[4];
        const float *keyZeros = (const float *)params[5];
        const float *keyScales = (const float *)params[6];
        const float *keyBias = (const float *)params[7];

        const int8_t *valueQWeight = (const int8_t *)params[8];
        const float *valueZeros = (const float *)params[9];
        const float *valueScales = (const float *)params[10];
        const float *valueBias = (const float *)params[11];

        const int8_t *attnOutQWeight = (const int8_t *)params[12];
        const float *attnOutZeros = (const float *)params[13];
        const float *attnOutScales = (const float *)params[14];
        const float *attnOutBias = (const float *)params[15];

        const float *gamma1 = (const float *)params[16];
        const float *beta1 = (const float *)params[17];

        attn.setQWeights(ctx, queryQWeight, queryZeros, queryScales, queryBias,
                keyQWeight, keyZeros, keyScales, keyBias,
                valueQWeight, valueZeros, valueScales, valueBias,
                attnOutQWeight, attnOutZeros, attnOutScales, attnOutBias,
                gamma1, beta1, trans);

        std::vector<void *> mlpParams(params.begin() + 18, params.end());
        mlp.setQWeights(ctx, mlpParams, trans);
    }

    template <typename KVCacheT>
    void forwardAttention(DecoderContext *ctx, float *input, float *output, const float *attnMask, KVCacheTensor<KVCacheT> &presentKey,
            KVCacheTensor<KVCacheT> &presentValue, int inputSeqLen, int pastSeqLen, bool useSelfAttn, bool doLnBefore,
            bool returnAttn, bool returnKVs, bool forPT = true, int *positionIds = nullptr) {
        TimeLine t("Decoder.forwardAttention");
        attn.forward(ctx, input, output, attnMask, presentKey, presentValue, inputSeqLen, pastSeqLen, useSelfAttn,
                doLnBefore, returnAttn, returnKVs, forPT, positionIds);
    }

    void forwardFFN(DecoderContext *ctx, float *input, float *output, int iStride, int oStride, bool doLnBefore = true) {
        TimeLine t("Decoder.forwardFFN");
        mlp.forward(ctx, input, output, iStride, oStride, doLnBefore);
    }

private:
    void copyWeights(hpj::Matrix<float> &w, int start_col, int end_col, const float *data) {
        hpj::Matrix<float> subW(w, 0, w.Rows(), start_col, end_col - start_col);
        copyWeights(subW, data);
    }

    // Copy the transposed weight into the non-transposed matrix
    void copyWeights(hpj::Matrix<float> &w, const float *data) {
        for (int j = 0; j < w.Cols(); ++j) {
            for (int i = 0; i < w.Rows(); ++i) {
                w(i, j) = *data++;
            }
        }
    }

    void copyTransposed(hpj::Matrix<float> &dst, hpj::Matrix<float> &src) {
        dst.Resize(src.Cols(), src.Rows());
        for (int i = 0; i < dst.Rows(); ++i) {
            for (int j = 0; j < dst.Cols(); ++j) {
                dst(i, j) = src(j, i);
            }
        }
    }

    // Add bias to matrix
    void biasAdd(hpj::Matrix<float> &m, hpj::Vector<float> &bias) {
        float *pbias = bias.Data();
#pragma omp parallel for
        for (int i = 0; i < m.Rows(); ++i) {
            float *p = m.Row(i);
#pragma omp simd
            for (int j = 0; j < m.Cols(); ++j) {
                p[j] += pbias[j];
            }
        }
    }

private:
    // For debug usage
    int layerIdx;

    ATTN_CLS attn;
    MLP_CLS mlp;

#ifdef DEBUG
    Debugger dbg;
#endif
};
