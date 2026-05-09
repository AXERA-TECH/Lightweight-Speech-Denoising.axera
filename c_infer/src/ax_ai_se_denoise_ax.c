/**
 * @file ax_ai_se_denoise_ax.c
 * @brief cache-only 语音增强库实现 — AX Engine 板端版本
 *
 * 支持:
 *   self_cache:   mix + feat_cache -> enh + feat_cache_out
 *   gtcrn_stream: mix + less-input no-scatter caches -> enh + updated caches
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "ax_base_type.h"
#include "tiny_se_v5_dsp.h"
#include "ax_ai_se_denoise.h"

#include "ax_sys_api.h"
#include "ax_engine_api.h"
#include "ax_engine_type.h"

#define CMM_ALIGN_SIZE              128

#define SE_MODEL_TYPE_MASK_BATCH    0
#define SE_MODEL_TYPE_GTCRN_STREAM  1
#define SE_MODEL_TYPE_SELF_CACHE    2
#define SE_MODEL_TYPE_GTCRN_SPLIT   3  /* gtcrn.axera 15-input split-cache model */

#define GTCRN_EN_CONV_CACHE_ELEMS   (1 * 16 * 16 * 33)
#define GTCRN_DE_CONV_CACHE_ELEMS   (1 * 16 * 16 * 33)
#define GTCRN_EN_TRA_CACHE_ELEMS    (1 * 3 * 1 * 16)
#define GTCRN_DE_TRA_CACHE_ELEMS    (1 * 3 * 1 * 16)
#define GTCRN_INTER_CACHE_0_ELEMS   (1 * 1 * 33 * 16)
#define GTCRN_INTER_CACHE_1_ELEMS   (1 * 1 * 33 * 16)

/* gtcrn_split sub-cache sizes (sub-regions within the packed buffers above) */
#define GTCRN_SPLIT_EN_CC0_ELEMS    (1 * 16 * 2  * 33)   /*  1056 */
#define GTCRN_SPLIT_EN_CC1_ELEMS    (1 * 16 * 4  * 33)   /*  2112 */
#define GTCRN_SPLIT_EN_CC2_ELEMS    (1 * 16 * 10 * 33)   /*  5280 */
#define GTCRN_SPLIT_DE_CC0_ELEMS    (1 * 16 * 10 * 33)   /*  5280 */
#define GTCRN_SPLIT_DE_CC1_ELEMS    (1 * 16 * 4  * 33)   /*  2112 */
#define GTCRN_SPLIT_DE_CC2_ELEMS    (1 * 16 * 2  * 33)   /*  1056 */
#define GTCRN_SPLIT_TRA_ELEMS       (1 * 1  * 16)         /*    16 */
#define GTCRN_SPLIT_INTER_ELEMS     (1 * 33 * 16)         /*   528 */

typedef struct {
    AX_ENGINE_HANDLE     handle;
    AX_ENGINE_IO_INFO_T *io_info;
    AX_ENGINE_IO_T       io;

    AX_S32 model_type;
    AX_S32 input_count;
    AX_S32 output_count;

    AX_F32 *gtcrn_en_conv_cache;
    AX_F32 *gtcrn_de_conv_cache;
    AX_F32 *gtcrn_en_tra_cache;
    AX_F32 *gtcrn_de_tra_cache;
    AX_F32 *gtcrn_inter_cache_0;
    AX_F32 *gtcrn_inter_cache_1;
    AX_F32 *gtcrn_mix_buf;

    AX_F32 *self_feat_cache;
    AX_F32 *self_mix_buf;
    AX_S32  self_context;
    AX_S32  self_step;
    AX_S32  self_f_freqs;
} AxEngine;

typedef struct {
    AX_S32 hop_len;
    AX_S32 t_model;
    AX_S32 f_freqs;
    AX_S32 f_int;
    AX_S32 context;
    AX_S32 step;
    AX_S32 model_type;

    TinySeV5DspState dsp;
    AxEngine         engine;

    AX_F32 *spec_ctx_real;
    AX_F32 *spec_ctx_imag;
    AX_F32 *spec_step_real;
    AX_F32 *spec_step_imag;

    AX_F32 *model_input_buf;
    AX_F32 *mask_full_buf;
    AX_F32 *enh_real_buf;
    AX_F32 *enh_imag_buf;

    AX_S32 frame_count;
} SeUnifiedAxState;

static AX_CHAR *cfg_trim(AX_CHAR *s) {
    while (isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    AX_CHAR *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) end--;
    *(end + 1) = '\0';
    return s;
}

AX_S32 AX_AI_SE_LoadConfig(const AX_CHAR *ini_path,
                           AX_AI_SE_Config *config,
                           AX_CHAR *model_path,
                           AX_S32 path_size) {
    if (config) {
        config->hop_len = SE_V5_HOP_LEN;
        config->t_model = SE_V5_T_MODEL;
        config->f_freqs = SE_V5_N_FREQS;
        config->f_int = SE_V5_F_INT;
        config->step = SE_V5_STEP;
        config->model_type = SE_MODEL_TYPE_MASK_BATCH;
    }
    if (model_path && path_size > 0) {
        model_path[0] = '\0';
    }

    FILE *fp = fopen(ini_path, "r");
    if (!fp) {
        fprintf(stderr, "[Config] Cannot open: %s, using defaults\n", ini_path);
        return -1;
    }

    AX_CHAR line[512];
    while (fgets(line, sizeof(line), fp)) {
        AX_CHAR *comment = strchr(line, '#');
        AX_CHAR *eq;
        AX_CHAR *key;
        AX_CHAR *val;
        if (comment) *comment = '\0';
        key = cfg_trim(line);
        if (*key == '\0' || *key == '[') continue;
        eq = strchr(key, '=');
        if (!eq) continue;
        *eq = '\0';
        val = cfg_trim(eq + 1);
        key = cfg_trim(key);

        if (config) {
            if (strcmp(key, "hop_len") == 0) config->hop_len = (AX_S32)atoi(val);
            else if (strcmp(key, "t_model") == 0) config->t_model = (AX_S32)atoi(val);
            else if (strcmp(key, "f_freqs") == 0) config->f_freqs = (AX_S32)atoi(val);
            else if (strcmp(key, "f_int") == 0) config->f_int = (AX_S32)atoi(val);
            else if (strcmp(key, "step") == 0) config->step = (AX_S32)atoi(val);
            else if (strcmp(key, "model_type") == 0) {
                if (strcmp(val, "gtcrn_stream") == 0) config->model_type = SE_MODEL_TYPE_GTCRN_STREAM;
                else if (strcmp(val, "gtcrn_split") == 0) config->model_type = SE_MODEL_TYPE_GTCRN_SPLIT;
                else if (strcmp(val, "self_cache") == 0) config->model_type = SE_MODEL_TYPE_SELF_CACHE;
                else config->model_type = SE_MODEL_TYPE_MASK_BATCH;
            }
        }

        if (model_path && path_size > 0 && strcmp(key, "model_path") == 0) {
            if (val[0] == '/') {
                snprintf(model_path, (size_t)path_size, "%s", val);
            } else {
                AX_CHAR ini_dir[512];
                AX_CHAR *last_sep;
                snprintf(ini_dir, sizeof(ini_dir), "%s", ini_path);
                last_sep = strrchr(ini_dir, '/');
                if (last_sep) {
                    *(last_sep + 1) = '\0';
                    snprintf(model_path, (size_t)path_size, "%s%s", ini_dir, val);
                } else {
                    snprintf(model_path, (size_t)path_size, "%s", val);
                }
            }
        }
    }
    fclose(fp);
    return 0;
}

static AX_VOID ax_engine_release_buffers(AxEngine *eng) {
    AX_U32 i;
    for (i = 0; i < eng->io.nInputSize; i++) {
        if (eng->io.pInputs && eng->io.pInputs[i].phyAddr) {
            AX_SYS_MemFree(eng->io.pInputs[i].phyAddr, eng->io.pInputs[i].pVirAddr);
        }
    }
    for (i = 0; i < eng->io.nOutputSize; i++) {
        if (eng->io.pOutputs && eng->io.pOutputs[i].phyAddr) {
            AX_SYS_MemFree(eng->io.pOutputs[i].phyAddr, eng->io.pOutputs[i].pVirAddr);
        }
    }
    free(eng->io.pInputs);
    free(eng->io.pOutputs);
    eng->io.pInputs = AX_NULL;
    eng->io.pOutputs = AX_NULL;
}

static AX_S32 ax_engine_flush_inputs(AxEngine *eng) {
    AX_U32 i;
    for (i = 0; i < eng->io.nInputSize; i++) {
        AX_ENGINE_IO_BUFFER_T *buf = &eng->io.pInputs[i];
        AX_S32 ret = AX_SYS_MflushCache(buf->phyAddr, buf->pVirAddr, buf->nSize);
        if (ret != 0) {
            fprintf(stderr, "[AxEngine] MflushCache input[%u] failed: 0x%x\n", i, ret);
            return -1;
        }
    }
    return 0;
}

static AX_S32 ax_engine_invalidate_outputs(AxEngine *eng) {
    AX_U32 i;
    for (i = 0; i < eng->io.nOutputSize; i++) {
        AX_ENGINE_IO_BUFFER_T *buf = &eng->io.pOutputs[i];
        AX_S32 ret = AX_SYS_MinvalidateCache(buf->phyAddr, buf->pVirAddr, buf->nSize);
        if (ret != 0) {
            fprintf(stderr, "[AxEngine] MinvalidateCache output[%u] failed: 0x%x\n", i, ret);
            return -1;
        }
    }
    return 0;
}

static AX_S32 ax_engine_init(AxEngine *eng,
                             const AX_CHAR *model_path,
                             AX_S32 t_model,
                             AX_S32 f_freqs,
                             AX_S32 step,
                             AX_S32 model_type) {
    AX_S32 ret;
    FILE *fp = AX_NULL;
    AX_VOID *model_data = AX_NULL;
    long model_size;
    AX_U32 i;

    memset(eng, 0, sizeof(*eng));

    ret = AX_SYS_Init();
    if (ret != 0) {
        fprintf(stderr, "[AxEngine] AX_SYS_Init failed: 0x%x\n", ret);
        return -1;
    }

    {
        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        ret = AX_ENGINE_Init(&npu_attr);
    }
    if (ret != 0) {
        fprintf(stderr, "[AxEngine] AX_ENGINE_Init failed: 0x%x\n", ret);
        AX_SYS_Deinit();
        return -1;
    }

    fp = fopen(model_path, "rb");
    if (!fp) {
        fprintf(stderr, "[AxEngine] Cannot open model: %s\n", model_path);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    model_data = malloc((size_t)model_size);
    if (!model_data || (long)fread(model_data, 1, (size_t)model_size, fp) != model_size) {
        fprintf(stderr, "[AxEngine] Failed to read model\n");
        fclose(fp);
        free(model_data);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }
    fclose(fp);

    ret = AX_ENGINE_CreateHandle(&eng->handle, model_data, (AX_U32)model_size);
    free(model_data);
    if (ret != 0) {
        fprintf(stderr, "[AxEngine] CreateHandle failed: 0x%x\n", ret);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }

    ret = AX_ENGINE_CreateContext(eng->handle);
    if (ret != 0) {
        fprintf(stderr, "[AxEngine] CreateContext failed: 0x%x\n", ret);
        AX_ENGINE_DestroyHandle(eng->handle);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }

    ret = AX_ENGINE_GetIOInfo(eng->handle, &eng->io_info);
    if (ret != 0) {
        fprintf(stderr, "[AxEngine] GetIOInfo failed: 0x%x\n", ret);
        AX_ENGINE_DestroyHandle(eng->handle);
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
        return -1;
    }

    eng->model_type = model_type;
    eng->input_count = (AX_S32)eng->io_info->nInputSize;
    eng->output_count = (AX_S32)eng->io_info->nOutputSize;

    memset(&eng->io, 0, sizeof(eng->io));
    eng->io.nInputSize = eng->io_info->nInputSize;
    eng->io.nOutputSize = eng->io_info->nOutputSize;
    eng->io.pInputs = (AX_ENGINE_IO_BUFFER_T *)calloc(eng->io.nInputSize, sizeof(AX_ENGINE_IO_BUFFER_T));
    eng->io.pOutputs = (AX_ENGINE_IO_BUFFER_T *)calloc(eng->io.nOutputSize, sizeof(AX_ENGINE_IO_BUFFER_T));
    if (!eng->io.pInputs || !eng->io.pOutputs) {
        fprintf(stderr, "[AxEngine] IO buffer metadata alloc failed\n");
        goto err;
    }

    for (i = 0; i < eng->io.nInputSize; i++) {
        AX_ENGINE_IO_BUFFER_T *buf = &eng->io.pInputs[i];
        buf->nSize = eng->io_info->pInputs[i].nSize;
        ret = AX_SYS_MemAlloc(&buf->phyAddr, &buf->pVirAddr, buf->nSize, CMM_ALIGN_SIZE,
                              (const AX_S8 *)eng->io_info->pInputs[i].pName);
        if (ret != 0) {
            fprintf(stderr, "[AxEngine] MemAlloc input[%u] failed: 0x%x\n", i, ret);
            goto err;
        }
        memset(buf->pVirAddr, 0, buf->nSize);
    }

    for (i = 0; i < eng->io.nOutputSize; i++) {
        AX_ENGINE_IO_BUFFER_T *buf = &eng->io.pOutputs[i];
        buf->nSize = eng->io_info->pOutputs[i].nSize;
        ret = AX_SYS_MemAlloc(&buf->phyAddr, &buf->pVirAddr, buf->nSize, CMM_ALIGN_SIZE,
                              (const AX_S8 *)eng->io_info->pOutputs[i].pName);
        if (ret != 0) {
            fprintf(stderr, "[AxEngine] MemAlloc output[%u] failed: 0x%x\n", i, ret);
            goto err;
        }
        memset(buf->pVirAddr, 0, buf->nSize);
    }

    if (model_type == SE_MODEL_TYPE_GTCRN_SPLIT) {
        size_t expected_mix_bytes = (size_t)step * (size_t)f_freqs * 2u * sizeof(AX_F32);
        if (eng->input_count != 15 || eng->output_count != 15) {
            fprintf(stderr, "[AxEngine] gtcrn_split requires 15 inputs and 15 outputs, got inputs=%d outputs=%d\n",
                    eng->input_count, eng->output_count);
            goto err;
        }
        if (eng->io.pInputs[0].nSize != expected_mix_bytes ||
            eng->io.pOutputs[0].nSize != expected_mix_bytes) {
            fprintf(stderr,
                    "[AxEngine] GTCRN split FP32 IO size mismatch: mix in=%u out=%u expected=%zu.\n",
                    eng->io.pInputs[0].nSize, eng->io.pOutputs[0].nSize, expected_mix_bytes);
            goto err;
        }
        eng->gtcrn_mix_buf = (AX_F32 *)calloc((size_t)step * (size_t)f_freqs * 2u, sizeof(AX_F32));
        eng->gtcrn_en_conv_cache = (AX_F32 *)calloc(GTCRN_EN_CONV_CACHE_ELEMS, sizeof(AX_F32));
        eng->gtcrn_de_conv_cache = (AX_F32 *)calloc(GTCRN_DE_CONV_CACHE_ELEMS, sizeof(AX_F32));
        eng->gtcrn_en_tra_cache  = (AX_F32 *)calloc(GTCRN_EN_TRA_CACHE_ELEMS, sizeof(AX_F32));
        eng->gtcrn_de_tra_cache  = (AX_F32 *)calloc(GTCRN_DE_TRA_CACHE_ELEMS, sizeof(AX_F32));
        eng->gtcrn_inter_cache_0 = (AX_F32 *)calloc(GTCRN_SPLIT_INTER_ELEMS, sizeof(AX_F32));
        eng->gtcrn_inter_cache_1 = (AX_F32 *)calloc(GTCRN_SPLIT_INTER_ELEMS, sizeof(AX_F32));
        if (!eng->gtcrn_en_conv_cache || !eng->gtcrn_de_conv_cache ||
            !eng->gtcrn_en_tra_cache  || !eng->gtcrn_de_tra_cache  ||
            !eng->gtcrn_inter_cache_0 || !eng->gtcrn_inter_cache_1 ||
            !eng->gtcrn_mix_buf) {
            fprintf(stderr, "[AxEngine] GTCRN split cache alloc failed\n");
            goto err;
        }
    } else if (model_type == SE_MODEL_TYPE_GTCRN_STREAM) {
        size_t expected_mix_bytes = (size_t)step * (size_t)f_freqs * 2u * sizeof(AX_F32);
        if (eng->input_count != 7 || eng->output_count != 7) {
            fprintf(stderr, "[AxEngine] gtcrn_stream now requires 7 inputs and 7 outputs, got inputs=%d outputs=%d\n",
                    eng->input_count, eng->output_count);
            goto err;
        }
        if (eng->io.pInputs[0].nSize != expected_mix_bytes ||
            eng->io.pOutputs[0].nSize != expected_mix_bytes) {
            fprintf(stderr,
                    "[AxEngine] GTCRN FP32 IO size mismatch: mix in=%u out=%u expected=%zu. "
                    "Rebuild axmodel with FP32 input/output or add quant/dequant in C.\n",
                    eng->io.pInputs[0].nSize, eng->io.pOutputs[0].nSize, expected_mix_bytes);
            goto err;
        }
        eng->gtcrn_mix_buf = (AX_F32 *)calloc((size_t)step * (size_t)f_freqs * 2u, sizeof(AX_F32));
        eng->gtcrn_en_conv_cache = (AX_F32 *)calloc(GTCRN_EN_CONV_CACHE_ELEMS, sizeof(AX_F32));
        eng->gtcrn_de_conv_cache = (AX_F32 *)calloc(GTCRN_DE_CONV_CACHE_ELEMS, sizeof(AX_F32));
        eng->gtcrn_en_tra_cache = (AX_F32 *)calloc(GTCRN_EN_TRA_CACHE_ELEMS, sizeof(AX_F32));
        eng->gtcrn_de_tra_cache = (AX_F32 *)calloc(GTCRN_DE_TRA_CACHE_ELEMS, sizeof(AX_F32));
        eng->gtcrn_inter_cache_0 = (AX_F32 *)calloc(GTCRN_INTER_CACHE_0_ELEMS, sizeof(AX_F32));
        eng->gtcrn_inter_cache_1 = (AX_F32 *)calloc(GTCRN_INTER_CACHE_1_ELEMS, sizeof(AX_F32));
        if (!eng->gtcrn_en_conv_cache || !eng->gtcrn_de_conv_cache ||
            !eng->gtcrn_en_tra_cache || !eng->gtcrn_de_tra_cache ||
            !eng->gtcrn_inter_cache_0 || !eng->gtcrn_inter_cache_1 ||
            !eng->gtcrn_mix_buf) {
            fprintf(stderr, "[AxEngine] GTCRN cache alloc failed\n");
            goto err;
        }
    } else if (model_type == SE_MODEL_TYPE_SELF_CACHE) {
        size_t expected_mix_bytes = (size_t)step * (size_t)f_freqs * 2u * sizeof(AX_F32);
        eng->self_context = t_model - step;
        eng->self_step = step;
        eng->self_f_freqs = f_freqs;
        if (eng->self_context < 0) {
            fprintf(stderr, "[AxEngine] invalid self_cache shape: t_model=%d step=%d\n", t_model, step);
            goto err;
        }
        {
            size_t expected_cache_bytes = (size_t)eng->self_context * (size_t)f_freqs * sizeof(AX_F32);
            if (eng->input_count != 2 || eng->output_count != 2 ||
                eng->io.pInputs[0].nSize != expected_mix_bytes ||
                eng->io.pInputs[1].nSize != expected_cache_bytes ||
                eng->io.pOutputs[0].nSize != expected_mix_bytes ||
                eng->io.pOutputs[1].nSize != expected_cache_bytes) {
                fprintf(stderr,
                        "[AxEngine] self_cache FP32 IO size mismatch: "
                        "inputs=(%u,%u) outputs=(%u,%u) expected=(%zu,%zu). "
                        "Rebuild axmodel with FP32 input/output or add quant/dequant in C.\n",
                        eng->input_count > 0 ? eng->io.pInputs[0].nSize : 0,
                        eng->input_count > 1 ? eng->io.pInputs[1].nSize : 0,
                        eng->output_count > 0 ? eng->io.pOutputs[0].nSize : 0,
                        eng->output_count > 1 ? eng->io.pOutputs[1].nSize : 0,
                        expected_mix_bytes, expected_cache_bytes);
                goto err;
            }
        }
        eng->self_feat_cache = (AX_F32 *)calloc((size_t)eng->self_context * (size_t)f_freqs, sizeof(AX_F32));
        eng->self_mix_buf = (AX_F32 *)calloc((size_t)step * (size_t)f_freqs * 2u, sizeof(AX_F32));
        if (!eng->self_feat_cache || !eng->self_mix_buf) {
            fprintf(stderr, "[AxEngine] self cache alloc failed\n");
            goto err;
        }
    }

    printf("[AxEngine] Model: %s\n", model_path);
    printf("           Type: %d, inputs=%u, outputs=%u\n",
           model_type, eng->io_info->nInputSize, eng->io_info->nOutputSize);
    for (i = 0; i < eng->io_info->nInputSize; i++) {
        printf("           Input[%u]:  %s  size=%u bytes\n",
               i, eng->io_info->pInputs[i].pName, eng->io_info->pInputs[i].nSize);
    }
    for (i = 0; i < eng->io_info->nOutputSize; i++) {
        printf("           Output[%u]: %s  size=%u bytes\n",
               i, eng->io_info->pOutputs[i].pName, eng->io_info->pOutputs[i].nSize);
    }
    return 0;

err:
    ax_engine_release_buffers(eng);
    if (eng->handle) AX_ENGINE_DestroyHandle(eng->handle);
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    return -1;
}

static AX_S32 ax_engine_run_mask_batch(AxEngine *eng,
                                       const AX_F32 *input_data,
                                       size_t input_elems,
                                       AX_F32 **output_data) {
    size_t input_bytes = input_elems * sizeof(AX_F32);
    if (eng->input_count < 1 || eng->output_count < 1) {
        fprintf(stderr, "[AxEngine] invalid IO count for mask batch\n");
        return -1;
    }
    if (input_bytes > eng->io.pInputs[0].nSize) {
        fprintf(stderr, "[AxEngine] mask batch input overflow: %zu > %u\n",
                input_bytes, eng->io.pInputs[0].nSize);
        return -1;
    }

    memcpy(eng->io.pInputs[0].pVirAddr, input_data, input_bytes);
    if (ax_engine_flush_inputs(eng) != 0) {
        return -1;
    }

    if (AX_ENGINE_RunSync(eng->handle, &eng->io) != 0) {
        fprintf(stderr, "[AxEngine] RunSync failed for mask batch\n");
        return -1;
    }
    if (ax_engine_invalidate_outputs(eng) != 0) {
        return -1;
    }

    *output_data = (AX_F32 *)eng->io.pOutputs[0].pVirAddr;
    return 0;
}

static AX_S32 ax_engine_run_gtcrn_chunk(AxEngine *eng,
                                        const AX_F32 *spec_real,
                                        const AX_F32 *spec_imag,
                                        AX_S32 step,
                                        AX_S32 f_freqs,
                                        AX_F32 *enh_real,
                                        AX_F32 *enh_imag) {
    AX_S32 k, sidx;
    if (eng->input_count != 7 || eng->output_count != 7) {
        fprintf(stderr, "[AxEngine] GTCRN requires 7 inputs and 7 outputs, got inputs=%d outputs=%d\n",
                eng->input_count, eng->output_count);
        return -1;
    }

    for (k = 0; k < f_freqs; k++) {
        for (sidx = 0; sidx < step; sidx++) {
            size_t dst = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
            size_t src = (size_t)sidx * (size_t)f_freqs + (size_t)k;
            eng->gtcrn_mix_buf[dst] = spec_real[src];
            eng->gtcrn_mix_buf[dst + 1] = spec_imag[src];
        }
    }

    memcpy(eng->io.pInputs[0].pVirAddr, eng->gtcrn_mix_buf,
           (size_t)step * (size_t)f_freqs * 2u * sizeof(AX_F32));
    memcpy(eng->io.pInputs[1].pVirAddr, eng->gtcrn_en_conv_cache, GTCRN_EN_CONV_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[2].pVirAddr, eng->gtcrn_de_conv_cache, GTCRN_DE_CONV_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[3].pVirAddr, eng->gtcrn_en_tra_cache, GTCRN_EN_TRA_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[4].pVirAddr, eng->gtcrn_de_tra_cache, GTCRN_DE_TRA_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[5].pVirAddr, eng->gtcrn_inter_cache_0, GTCRN_INTER_CACHE_0_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[6].pVirAddr, eng->gtcrn_inter_cache_1, GTCRN_INTER_CACHE_1_ELEMS * sizeof(AX_F32));
    if (ax_engine_flush_inputs(eng) != 0) {
        return -1;
    }

    if (AX_ENGINE_RunSync(eng->handle, &eng->io) != 0) {
        fprintf(stderr, "[AxEngine] RunSync failed for GTCRN\n");
        return -1;
    }
    if (ax_engine_invalidate_outputs(eng) != 0) {
        return -1;
    }

    {
        AX_F32 *enh = (AX_F32 *)eng->io.pOutputs[0].pVirAddr;
        for (k = 0; k < f_freqs; k++) {
            for (sidx = 0; sidx < step; sidx++) {
                size_t src = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
                size_t dst = (size_t)sidx * (size_t)f_freqs + (size_t)k;
                enh_real[dst] = enh[src];
                enh_imag[dst] = enh[src + 1];
            }
        }
    }

    memcpy(eng->gtcrn_en_conv_cache, eng->io.pOutputs[1].pVirAddr, GTCRN_EN_CONV_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(eng->gtcrn_de_conv_cache, eng->io.pOutputs[2].pVirAddr, GTCRN_DE_CONV_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(eng->gtcrn_en_tra_cache, eng->io.pOutputs[3].pVirAddr, GTCRN_EN_TRA_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(eng->gtcrn_de_tra_cache, eng->io.pOutputs[4].pVirAddr, GTCRN_DE_TRA_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(eng->gtcrn_inter_cache_0, eng->io.pOutputs[5].pVirAddr, GTCRN_INTER_CACHE_0_ELEMS * sizeof(AX_F32));
    memcpy(eng->gtcrn_inter_cache_1, eng->io.pOutputs[6].pVirAddr, GTCRN_INTER_CACHE_1_ELEMS * sizeof(AX_F32));
    return 0;
}

static AX_S32 ax_engine_run_gtcrn_split_chunk(AxEngine *eng,
                                               const AX_F32 *spec_real,
                                               const AX_F32 *spec_imag,
                                               AX_S32 step,
                                               AX_S32 f_freqs,
                                               AX_F32 *enh_real,
                                               AX_F32 *enh_imag) {
    AX_S32 k, sidx;
    if (eng->input_count != 15 || eng->output_count != 15) {
        fprintf(stderr, "[AxEngine] GTCRN split requires 15 inputs and 15 outputs, got %d/%d\n",
                eng->input_count, eng->output_count);
        return -1;
    }

    for (k = 0; k < f_freqs; k++) {
        for (sidx = 0; sidx < step; sidx++) {
            size_t dst = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
            size_t src = (size_t)sidx * (size_t)f_freqs + (size_t)k;
            eng->gtcrn_mix_buf[dst]     = spec_real[src];
            eng->gtcrn_mix_buf[dst + 1] = spec_imag[src];
        }
    }

    AX_F32 *en_cc0 = eng->gtcrn_en_conv_cache;
    AX_F32 *en_cc1 = eng->gtcrn_en_conv_cache + GTCRN_SPLIT_EN_CC0_ELEMS;
    AX_F32 *en_cc2 = eng->gtcrn_en_conv_cache + GTCRN_SPLIT_EN_CC0_ELEMS + GTCRN_SPLIT_EN_CC1_ELEMS;
    AX_F32 *de_cc0 = eng->gtcrn_de_conv_cache;
    AX_F32 *de_cc1 = eng->gtcrn_de_conv_cache + GTCRN_SPLIT_DE_CC0_ELEMS;
    AX_F32 *de_cc2 = eng->gtcrn_de_conv_cache + GTCRN_SPLIT_DE_CC0_ELEMS + GTCRN_SPLIT_DE_CC1_ELEMS;
    AX_F32 *en_tc0 = eng->gtcrn_en_tra_cache;
    AX_F32 *en_tc1 = eng->gtcrn_en_tra_cache + GTCRN_SPLIT_TRA_ELEMS;
    AX_F32 *en_tc2 = eng->gtcrn_en_tra_cache + 2 * GTCRN_SPLIT_TRA_ELEMS;
    AX_F32 *de_tc0 = eng->gtcrn_de_tra_cache;
    AX_F32 *de_tc1 = eng->gtcrn_de_tra_cache + GTCRN_SPLIT_TRA_ELEMS;
    AX_F32 *de_tc2 = eng->gtcrn_de_tra_cache + 2 * GTCRN_SPLIT_TRA_ELEMS;

    memcpy(eng->io.pInputs[0].pVirAddr,  eng->gtcrn_mix_buf,      (size_t)step * (size_t)f_freqs * 2u * sizeof(AX_F32));
    memcpy(eng->io.pInputs[1].pVirAddr,  en_cc0, GTCRN_SPLIT_EN_CC0_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[2].pVirAddr,  en_cc1, GTCRN_SPLIT_EN_CC1_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[3].pVirAddr,  en_cc2, GTCRN_SPLIT_EN_CC2_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[4].pVirAddr,  de_cc0, GTCRN_SPLIT_DE_CC0_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[5].pVirAddr,  de_cc1, GTCRN_SPLIT_DE_CC1_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[6].pVirAddr,  de_cc2, GTCRN_SPLIT_DE_CC2_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[7].pVirAddr,  en_tc0, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[8].pVirAddr,  en_tc1, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[9].pVirAddr,  en_tc2, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[10].pVirAddr, de_tc0, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[11].pVirAddr, de_tc1, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[12].pVirAddr, de_tc2, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[13].pVirAddr, eng->gtcrn_inter_cache_0, GTCRN_SPLIT_INTER_ELEMS * sizeof(AX_F32));
    memcpy(eng->io.pInputs[14].pVirAddr, eng->gtcrn_inter_cache_1, GTCRN_SPLIT_INTER_ELEMS * sizeof(AX_F32));
    if (ax_engine_flush_inputs(eng) != 0) return -1;

    if (AX_ENGINE_RunSync(eng->handle, &eng->io) != 0) {
        fprintf(stderr, "[AxEngine] RunSync failed for GTCRN split\n");
        return -1;
    }
    if (ax_engine_invalidate_outputs(eng) != 0) return -1;

    {
        AX_F32 *enh = (AX_F32 *)eng->io.pOutputs[0].pVirAddr;
        for (k = 0; k < f_freqs; k++) {
            for (sidx = 0; sidx < step; sidx++) {
                size_t src = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
                size_t dst = (size_t)sidx * (size_t)f_freqs + (size_t)k;
                enh_real[dst] = enh[src];
                enh_imag[dst] = enh[src + 1];
            }
        }
    }

    memcpy(en_cc0, eng->io.pOutputs[1].pVirAddr,  GTCRN_SPLIT_EN_CC0_ELEMS * sizeof(AX_F32));
    memcpy(en_cc1, eng->io.pOutputs[2].pVirAddr,  GTCRN_SPLIT_EN_CC1_ELEMS * sizeof(AX_F32));
    memcpy(en_cc2, eng->io.pOutputs[3].pVirAddr,  GTCRN_SPLIT_EN_CC2_ELEMS * sizeof(AX_F32));
    memcpy(de_cc0, eng->io.pOutputs[4].pVirAddr,  GTCRN_SPLIT_DE_CC0_ELEMS * sizeof(AX_F32));
    memcpy(de_cc1, eng->io.pOutputs[5].pVirAddr,  GTCRN_SPLIT_DE_CC1_ELEMS * sizeof(AX_F32));
    memcpy(de_cc2, eng->io.pOutputs[6].pVirAddr,  GTCRN_SPLIT_DE_CC2_ELEMS * sizeof(AX_F32));
    memcpy(en_tc0, eng->io.pOutputs[7].pVirAddr,  GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(en_tc1, eng->io.pOutputs[8].pVirAddr,  GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(en_tc2, eng->io.pOutputs[9].pVirAddr,  GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(de_tc0, eng->io.pOutputs[10].pVirAddr, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(de_tc1, eng->io.pOutputs[11].pVirAddr, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(de_tc2, eng->io.pOutputs[12].pVirAddr, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
    memcpy(eng->gtcrn_inter_cache_0, eng->io.pOutputs[13].pVirAddr, GTCRN_SPLIT_INTER_ELEMS * sizeof(AX_F32));
    memcpy(eng->gtcrn_inter_cache_1, eng->io.pOutputs[14].pVirAddr, GTCRN_SPLIT_INTER_ELEMS * sizeof(AX_F32));
    return 0;
}

static AX_S32 ax_engine_run_self_cache_chunk(AxEngine *eng,
                                             const AX_F32 *spec_real,
                                             const AX_F32 *spec_imag,
                                             AX_S32 step,
                                             AX_F32 *enh_real,
                                             AX_F32 *enh_imag) {
    AX_S32 k, sidx;
    AX_S32 f_freqs = eng->self_f_freqs;
    if (eng->input_count < 2 || eng->output_count < 2) {
        fprintf(stderr, "[AxEngine] self_cache requires 2 inputs and 2 outputs\n");
        return -1;
    }

    for (k = 0; k < f_freqs; k++) {
        for (sidx = 0; sidx < step; sidx++) {
            size_t dst = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
            size_t src = (size_t)sidx * (size_t)f_freqs + (size_t)k;
            eng->self_mix_buf[dst] = spec_real[src];
            eng->self_mix_buf[dst + 1] = spec_imag[src];
        }
    }

    memcpy(eng->io.pInputs[0].pVirAddr, eng->self_mix_buf,
           (size_t)step * (size_t)f_freqs * 2u * sizeof(AX_F32));
    memcpy(eng->io.pInputs[1].pVirAddr, eng->self_feat_cache,
           (size_t)eng->self_context * (size_t)f_freqs * sizeof(AX_F32));
    if (ax_engine_flush_inputs(eng) != 0) {
        return -1;
    }

    if (AX_ENGINE_RunSync(eng->handle, &eng->io) != 0) {
        fprintf(stderr, "[AxEngine] RunSync failed for self_cache\n");
        return -1;
    }
    if (ax_engine_invalidate_outputs(eng) != 0) {
        return -1;
    }

    {
        AX_F32 *enh = (AX_F32 *)eng->io.pOutputs[0].pVirAddr;
        for (k = 0; k < f_freqs; k++) {
            for (sidx = 0; sidx < step; sidx++) {
                size_t src = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
                size_t dst = (size_t)sidx * (size_t)f_freqs + (size_t)k;
                enh_real[dst] = enh[src];
                enh_imag[dst] = enh[src + 1];
            }
        }
    }

    memcpy(eng->self_feat_cache, eng->io.pOutputs[1].pVirAddr,
           (size_t)eng->self_context * (size_t)f_freqs * sizeof(AX_F32));
    return 0;
}

static AX_VOID ax_engine_free(AxEngine *eng) {
    free(eng->gtcrn_en_conv_cache);
    free(eng->gtcrn_de_conv_cache);
    free(eng->gtcrn_en_tra_cache);
    free(eng->gtcrn_de_tra_cache);
    free(eng->gtcrn_inter_cache_0);
    free(eng->gtcrn_inter_cache_1);
    free(eng->gtcrn_mix_buf);
    free(eng->self_feat_cache);
    free(eng->self_mix_buf);
    eng->gtcrn_en_conv_cache = AX_NULL;
    eng->gtcrn_de_conv_cache = AX_NULL;
    eng->gtcrn_en_tra_cache = AX_NULL;
    eng->gtcrn_de_tra_cache = AX_NULL;
    eng->gtcrn_inter_cache_0 = AX_NULL;
    eng->gtcrn_inter_cache_1 = AX_NULL;
    eng->gtcrn_mix_buf = AX_NULL;
    eng->self_feat_cache = AX_NULL;
    eng->self_mix_buf = AX_NULL;

    ax_engine_release_buffers(eng);
    if (eng->handle) {
        AX_ENGINE_DestroyHandle(eng->handle);
        eng->handle = AX_NULL;
    }
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
}

static AX_VOID build_model_input(SeUnifiedAxState *state) {
    AX_S32 F = state->f_freqs;
    AX_S32 Tc = state->context;
    AX_S32 S = state->step;
    AX_F32 *ch0 = state->model_input_buf;
    AX_F32 feat_tmp[SE_V5_N_FREQS];
    AX_S32 t;

    for (t = 0; t < Tc; t++) {
        tiny_se_v5_extract_feat(state->spec_ctx_real + t * F,
                                state->spec_ctx_imag + t * F, feat_tmp);
        memcpy(ch0 + t * F, feat_tmp, (size_t)F * sizeof(AX_F32));
    }
    for (t = 0; t < S; t++) {
        tiny_se_v5_extract_feat(state->spec_step_real + t * F,
                                state->spec_step_imag + t * F, feat_tmp);
        memcpy(ch0 + (Tc + t) * F, feat_tmp, (size_t)F * sizeof(AX_F32));
    }
}

static AX_VOID update_context(SeUnifiedAxState *state) {
    AX_S32 F = state->f_freqs;
    AX_S32 Tc = state->context;
    AX_S32 S = state->step;

    memmove(state->spec_ctx_real, state->spec_ctx_real + S * F,
            (size_t)(Tc - S) * (size_t)F * sizeof(AX_F32));
    memmove(state->spec_ctx_imag, state->spec_ctx_imag + S * F,
            (size_t)(Tc - S) * (size_t)F * sizeof(AX_F32));
    memcpy(state->spec_ctx_real + (Tc - S) * F, state->spec_step_real,
           (size_t)S * (size_t)F * sizeof(AX_F32));
    memcpy(state->spec_ctx_imag + (Tc - S) * F, state->spec_step_imag,
           (size_t)S * (size_t)F * sizeof(AX_F32));
}

AX_S32 AX_AI_SE_Init(AX_VOID **handle,
                     const AX_AI_SE_Config *config,
                     const AX_CHAR *model_path) {
    SeUnifiedAxState *state;
    AX_S32 dsp_ret;
    AX_S32 ctx_size;
    AX_S32 step_size;
    AX_S32 input_size;

    if (!handle || !model_path) {
        fprintf(stderr, "[AX_AI_SE_AX] Init: invalid arguments\n");
        return -1;
    }

    state = (SeUnifiedAxState *)calloc(1, sizeof(SeUnifiedAxState));
    if (!state) return -1;

    if (config) {
        state->hop_len = config->hop_len > 0 ? config->hop_len : SE_V5_HOP_LEN;
        state->t_model = config->t_model > 0 ? config->t_model : SE_V5_T_MODEL;
        state->f_freqs = config->f_freqs > 0 ? config->f_freqs : SE_V5_N_FREQS;
        state->f_int = config->f_int > 0 ? config->f_int : SE_V5_F_INT;
        state->step = config->step > 0 ? config->step : SE_V5_STEP;
        state->model_type = config->model_type;
    } else {
        state->hop_len = SE_V5_HOP_LEN;
        state->t_model = SE_V5_T_MODEL;
        state->f_freqs = SE_V5_N_FREQS;
        state->f_int = SE_V5_F_INT;
        state->step = SE_V5_STEP;
        state->model_type = SE_MODEL_TYPE_MASK_BATCH;
    }
    state->context = (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
                      state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT)
        ? 0
        : (state->t_model - state->step);

    dsp_ret = (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
               state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT)
        ? tiny_se_v5_dsp_init_sqrt_hann(&state->dsp)
        : tiny_se_v5_dsp_init(&state->dsp);
    if (dsp_ret != 0) {
        fprintf(stderr, "[AX_AI_SE_AX] DSP init failed\n");
        free(state);
        return -1;
    }

    if (ax_engine_init(&state->engine, model_path, state->t_model, state->f_freqs,
                       state->step, state->model_type) != 0) {
        tiny_se_v5_dsp_free(&state->dsp);
        free(state);
        return -1;
    }

    ctx_size = state->context * state->f_freqs;
    step_size = state->step * state->f_freqs;
    input_size = SE_V5_FEAT_CH * state->t_model * state->f_freqs;

    state->spec_ctx_real = (AX_F32 *)calloc((size_t)ctx_size, sizeof(AX_F32));
    state->spec_ctx_imag = (AX_F32 *)calloc((size_t)ctx_size, sizeof(AX_F32));
    state->spec_step_real = (AX_F32 *)calloc((size_t)step_size, sizeof(AX_F32));
    state->spec_step_imag = (AX_F32 *)calloc((size_t)step_size, sizeof(AX_F32));
    state->model_input_buf = (AX_F32 *)calloc((size_t)input_size, sizeof(AX_F32));
    state->mask_full_buf = (AX_F32 *)calloc((size_t)state->f_freqs, sizeof(AX_F32));

    if (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
        state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT ||
        state->model_type == SE_MODEL_TYPE_SELF_CACHE) {
        state->enh_real_buf = (AX_F32 *)calloc((size_t)step_size, sizeof(AX_F32));
        state->enh_imag_buf = (AX_F32 *)calloc((size_t)step_size, sizeof(AX_F32));
    }

    if (!state->spec_ctx_real || !state->spec_ctx_imag ||
        !state->spec_step_real || !state->spec_step_imag ||
        !state->model_input_buf || !state->mask_full_buf ||
        ((state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
          state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT ||
          state->model_type == SE_MODEL_TYPE_SELF_CACHE) &&
         (!state->enh_real_buf || !state->enh_imag_buf))) {
        AX_AI_SE_Free(state);
        return -1;
    }

    state->frame_count = 0;
    *handle = (AX_VOID *)state;

    printf("[AX_AI_SE_AX] hop_len=%d t_model=%d f_freqs=%d f_int=%d step=%d context=%d model_type=%d\n",
           state->hop_len, state->t_model, state->f_freqs, state->f_int,
           state->step, state->context, state->model_type);
    return 0;
}

AX_VOID AX_AI_SE_PreProcess(AX_VOID *handle, AX_AI_SE_PreParams *params) {
    SeUnifiedAxState *state;
    AX_S32 s;
    AX_S32 F;
    AX_S32 hop;

    if (!handle || !params || !params->pcm_in_batch) return;

    state = (SeUnifiedAxState *)handle;
    F = state->f_freqs;
    hop = state->hop_len;

    for (s = 0; s < state->step; s++) {
        tiny_se_v5_stft_frame(&state->dsp, params->pcm_in_batch + s * hop,
                              state->spec_step_real + s * F,
                              state->spec_step_imag + s * F);
    }

    if (state->model_type == SE_MODEL_TYPE_MASK_BATCH &&
        state->frame_count == 0 && state->context > 0) {
        AX_S32 t;
        for (t = 0; t < state->context; t++) {
            memcpy(state->spec_ctx_real + t * F, state->spec_step_real, (size_t)F * sizeof(AX_F32));
            memcpy(state->spec_ctx_imag + t * F, state->spec_step_imag, (size_t)F * sizeof(AX_F32));
        }
    }
    state->frame_count += state->step;

    if (state->model_type == SE_MODEL_TYPE_MASK_BATCH) {
        build_model_input(state);
        params->model_input = state->model_input_buf;
        params->model_input_size = SE_V5_FEAT_CH * state->t_model * state->f_freqs;
    } else {
        params->model_input = state->spec_step_real;
        params->model_input_size = state->step * state->f_freqs;
    }
}

AX_S32 AX_AI_SE_Infer(AX_VOID *handle, AX_AI_SE_InferParams *params) {
    SeUnifiedAxState *state;
    AX_F32 *output_data = AX_NULL;

    if (!handle || !params) return -1;
    state = (SeUnifiedAxState *)handle;

    if (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM) {
        if (ax_engine_run_gtcrn_chunk(&state->engine, state->spec_step_real, state->spec_step_imag,
                                      state->step, state->f_freqs,
                                      state->enh_real_buf, state->enh_imag_buf) != 0) {
            return -1;
        }
        params->model_output = state->enh_real_buf;
        params->model_output_size = state->step * state->f_freqs * 2;
        return 0;
    }

    if (state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT) {
        if (ax_engine_run_gtcrn_split_chunk(&state->engine, state->spec_step_real, state->spec_step_imag,
                                            state->step, state->f_freqs,
                                            state->enh_real_buf, state->enh_imag_buf) != 0) {
            return -1;
        }
        params->model_output = state->enh_real_buf;
        params->model_output_size = state->step * state->f_freqs * 2;
        return 0;
    }

    if (state->model_type == SE_MODEL_TYPE_SELF_CACHE) {
        if (ax_engine_run_self_cache_chunk(&state->engine, state->spec_step_real, state->spec_step_imag,
                                           state->step, state->enh_real_buf, state->enh_imag_buf) != 0) {
            return -1;
        }
        params->model_output = state->enh_real_buf;
        params->model_output_size = state->step * state->f_freqs * 2;
        return 0;
    }

    if (!params->model_input) return -1;
    if (ax_engine_run_mask_batch(&state->engine, params->model_input,
                                 (size_t)SE_V5_FEAT_CH * (size_t)state->t_model * (size_t)state->f_freqs,
                                 &output_data) != 0) {
        return -1;
    }

    params->model_output = output_data;
    params->model_output_size = SE_V5_MASK_CH * state->t_model * state->f_int;
    return 0;
}

AX_VOID AX_AI_SE_PostProcess(AX_VOID *handle, AX_AI_SE_PostParams *params) {
    SeUnifiedAxState *state;
    AX_S32 s;

    if (!handle || !params || !params->model_output || !params->pcm_out_batch) return;
    state = (SeUnifiedAxState *)handle;

    if (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
        state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT ||
        state->model_type == SE_MODEL_TYPE_SELF_CACHE) {
        for (s = 0; s < state->step; s++) {
            tiny_se_v5_istft_frame(&state->dsp,
                                   state->enh_real_buf + s * state->f_freqs,
                                   state->enh_imag_buf + s * state->f_freqs,
                                   params->pcm_out_batch + s * state->hop_len);
        }
        return;
    }

    {
        AX_S32 Tc = state->context;
        AX_S32 Fi = state->f_int;
        AX_S32 F = state->f_freqs;
        AX_S32 hop = state->hop_len;
        const AX_F32 *out = params->model_output;

        for (s = 0; s < state->step; s++) {
            AX_F32 mask_int[SE_V5_N_FREQS];
            AX_F32 enh_real[SE_V5_N_FREQS];
            AX_F32 enh_imag[SE_V5_N_FREQS];

            memcpy(mask_int, out + (Tc + s) * Fi, (size_t)Fi * sizeof(AX_F32));
            tiny_se_v5_sigmoid_inplace(mask_int, Fi);
            tiny_se_v5_interp_mask_n(mask_int, Fi, state->mask_full_buf, F);

            memcpy(enh_real, state->spec_step_real + s * F, (size_t)F * sizeof(AX_F32));
            memcpy(enh_imag, state->spec_step_imag + s * F, (size_t)F * sizeof(AX_F32));
            tiny_se_v5_apply_irm(enh_real, enh_imag, state->mask_full_buf);

            tiny_se_v5_istft_frame(&state->dsp, enh_real, enh_imag,
                                   params->pcm_out_batch + s * hop);
        }
    }

    update_context(state);
}

AX_VOID AX_AI_SE_Free(AX_VOID *handle) {
    SeUnifiedAxState *state = (SeUnifiedAxState *)handle;
    if (!state) return;

    free(state->model_input_buf);
    free(state->mask_full_buf);
    free(state->enh_real_buf);
    free(state->enh_imag_buf);
    free(state->spec_ctx_real);
    free(state->spec_ctx_imag);
    free(state->spec_step_real);
    free(state->spec_step_imag);

    ax_engine_free(&state->engine);
    tiny_se_v5_dsp_free(&state->dsp);
    free(state);
}
