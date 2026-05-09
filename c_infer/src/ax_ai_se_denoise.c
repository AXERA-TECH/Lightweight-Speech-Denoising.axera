/**
 * @file ax_ai_se_denoise.c
 * @brief cache-only 语音增强库实现 — x86 ONNX Runtime 版本
 *
 * 通过 INI 配置文件支持:
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

#include "onnxruntime_c_api.h"

#define SE_MODEL_TYPE_MASK_BATCH   0
#define SE_MODEL_TYPE_GTCRN_STREAM 1
#define SE_MODEL_TYPE_SELF_CACHE   2
#define SE_MODEL_TYPE_GTCRN_SPLIT  3  /* gtcrn.axera 15-input split-cache model */

#define GTCRN_EN_CONV_CACHE_ELEMS  (1 * 16 * 16 * 33)
#define GTCRN_DE_CONV_CACHE_ELEMS  (1 * 16 * 16 * 33)
#define GTCRN_EN_TRA_CACHE_ELEMS   (1 * 3 * 1 * 16)
#define GTCRN_DE_TRA_CACHE_ELEMS   (1 * 3 * 1 * 16)
#define GTCRN_INTER_CACHE_0_ELEMS  (1 * 1 * 33 * 16)
#define GTCRN_INTER_CACHE_1_ELEMS  (1 * 1 * 33 * 16)

/* gtcrn_split sub-cache sizes (sub-regions within the packed buffers above) */
#define GTCRN_SPLIT_EN_CC0_ELEMS   (1 * 16 * 2  * 33)   /*  1056 */
#define GTCRN_SPLIT_EN_CC1_ELEMS   (1 * 16 * 4  * 33)   /*  2112 */
#define GTCRN_SPLIT_EN_CC2_ELEMS   (1 * 16 * 10 * 33)   /*  5280 */
#define GTCRN_SPLIT_DE_CC0_ELEMS   (1 * 16 * 10 * 33)   /*  5280 */
#define GTCRN_SPLIT_DE_CC1_ELEMS   (1 * 16 * 4  * 33)   /*  2112 */
#define GTCRN_SPLIT_DE_CC2_ELEMS   (1 * 16 * 2  * 33)   /*  1056 */
#define GTCRN_SPLIT_TRA_ELEMS      (1 * 1  * 16)         /*    16 */
#define GTCRN_SPLIT_INTER_ELEMS    (1 * 33 * 16)         /*   528 */

/* ═══════════════════════════════════════════════════════════════
 *  内部类型定义
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    const OrtApi          *api;
    OrtEnv                *env;
    OrtSession            *session;
    OrtSessionOptions     *session_opts;
    OrtMemoryInfo         *memory_info;
    OrtAllocator          *allocator;
    char                 **input_names;
    char                 **output_names;
    size_t                 input_name_count;
    size_t                 output_name_count;
    int64_t  input_shape[4];
    int64_t  output_shape[4];
    size_t   input_elem_count;
    size_t   output_elem_count;
    AX_F32  *output_buf;

    AX_S32   model_type;
    AX_F32  *gtcrn_en_conv_cache;
    AX_F32  *gtcrn_de_conv_cache;
    AX_F32  *gtcrn_en_tra_cache;
    AX_F32  *gtcrn_de_tra_cache;
    AX_F32  *gtcrn_inter_cache_0;
    AX_F32  *gtcrn_inter_cache_1;
    AX_S32   gtcrn_step;
    AX_F32  *gtcrn_mix_buf;

    AX_F32  *self_feat_cache;
    AX_F32  *self_mix_buf;
    AX_S32   self_context;
    AX_S32   self_step;
    AX_S32   self_f_freqs;
} OrtEngine;

typedef struct {
    AX_S32 hop_len;
    AX_S32 t_model;
    AX_S32 f_freqs;
    AX_S32 f_int;
    AX_S32 context;
    AX_S32 step;
    AX_S32 model_type;

    TinySeV5DspState dsp;
    OrtEngine ort;

    AX_F32 *spec_ctx_real;   /* [context * f_freqs] */
    AX_F32 *spec_ctx_imag;
    AX_F32 *spec_step_real;  /* [step * f_freqs] */
    AX_F32 *spec_step_imag;

    AX_F32 *model_input_buf;
    AX_F32 *mask_full_buf;   /* [f_freqs] — 动态分配，支持任意 f_freqs */
    AX_F32 *gtcrn_enh_real;  /* [step * f_freqs] */
    AX_F32 *gtcrn_enh_imag;  /* [step * f_freqs] */

    AX_S32 frame_count;
} SeUnifiedState;

/* ═══════════════════════════════════════════════════════════════
 *  配置文件解析 (INI)
 * ═══════════════════════════════════════════════════════════════ */

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
                             AX_S32   path_size)
{
    /* 默认值 (tiny_v5) */
    if (config) {
        config->hop_len = SE_V5_HOP_LEN;
        config->t_model = SE_V5_T_MODEL;
        config->f_freqs = SE_V5_N_FREQS;
        config->f_int   = SE_V5_F_INT;
        config->step    = SE_V5_STEP;
        config->model_type = SE_MODEL_TYPE_MASK_BATCH;
    }
    if (model_path && path_size > 0)
        model_path[0] = '\0';

    FILE *fp = fopen(ini_path, "r");
    if (!fp) {
        fprintf(stderr, "[Config] Cannot open: %s, using defaults\n", ini_path);
        return -1;
    }

    AX_CHAR line[512];
    while (fgets(line, sizeof(line), fp)) {
        AX_CHAR *comment = strchr(line, '#');
        if (comment) *comment = '\0';
        AX_CHAR *p = cfg_trim(line);
        if (*p == '\0' || *p == '[') continue;
        AX_CHAR *eq = strchr(p, '=');
        if (!eq) continue;
        *eq = '\0';
        AX_CHAR *key = cfg_trim(p);
        AX_CHAR *val = cfg_trim(eq + 1);

        if (config) {
            if      (strcmp(key, "hop_len") == 0) config->hop_len = (AX_S32)atoi(val);
            else if (strcmp(key, "t_model") == 0) config->t_model = (AX_S32)atoi(val);
            else if (strcmp(key, "f_freqs") == 0) config->f_freqs = (AX_S32)atoi(val);
            else if (strcmp(key, "f_int")   == 0) config->f_int   = (AX_S32)atoi(val);
            else if (strcmp(key, "step")    == 0) config->step    = (AX_S32)atoi(val);
            else if (strcmp(key, "model_type") == 0) {
                if (strcmp(val, "gtcrn_stream") == 0) {
                    config->model_type = SE_MODEL_TYPE_GTCRN_STREAM;
                } else if (strcmp(val, "gtcrn_split") == 0) {
                    config->model_type = SE_MODEL_TYPE_GTCRN_SPLIT;
                } else if (strcmp(val, "self_cache") == 0) {
                    config->model_type = SE_MODEL_TYPE_SELF_CACHE;
                } else {
                    config->model_type = SE_MODEL_TYPE_MASK_BATCH;
                }
            }
        }
        if (model_path && path_size > 0 && strcmp(key, "model_path") == 0) {
            if (val[0] == '/') {
                snprintf(model_path, (size_t)path_size, "%s", val);
            } else {
                AX_CHAR ini_dir[512];
                snprintf(ini_dir, sizeof(ini_dir), "%s", ini_path);
                AX_CHAR *last_sep = strrchr(ini_dir, '/');
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

/* ═══════════════════════════════════════════════════════════════
 *  静态辅助函数
 * ═══════════════════════════════════════════════════════════════ */

#define ORT_CHECK_INIT(api, expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        fprintf(stderr, "[ORT Error] %s\n", \
                (api)->GetErrorMessage(_s)); \
        (api)->ReleaseStatus(_s); \
        return -1; \
    } \
} while(0)

static AX_S32 ort_cache_io_names(OrtEngine *ort) {
    const OrtApi *api = ort->api;
    OrtStatus *s = AX_NULL;
    size_t i;

    s = api->GetAllocatorWithDefaultOptions(&ort->allocator);
    if (s) {
        fprintf(stderr, "[ORT] GetAllocatorWithDefaultOptions: %s\n", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        return -1;
    }

    s = api->SessionGetInputCount(ort->session, &ort->input_name_count);
    if (s) {
        fprintf(stderr, "[ORT] SessionGetInputCount: %s\n", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        return -1;
    }
    s = api->SessionGetOutputCount(ort->session, &ort->output_name_count);
    if (s) {
        fprintf(stderr, "[ORT] SessionGetOutputCount: %s\n", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        return -1;
    }

    ort->input_names = (char **)calloc(ort->input_name_count, sizeof(char *));
    ort->output_names = (char **)calloc(ort->output_name_count, sizeof(char *));
    if (!ort->input_names || !ort->output_names) {
        fprintf(stderr, "[ORT] alloc io name arrays failed\n");
        return -1;
    }

    for (i = 0; i < ort->input_name_count; i++) {
        char *tmp = AX_NULL;
        s = api->SessionGetInputName(ort->session, i, ort->allocator, &tmp);
        if (s) {
            fprintf(stderr, "[ORT] SessionGetInputName[%zu]: %s\n", i, api->GetErrorMessage(s));
            api->ReleaseStatus(s);
            return -1;
        }
        ort->input_names[i] = strdup(tmp);
        ort->allocator->Free(ort->allocator, tmp);
        if (!ort->input_names[i]) return -1;
    }

    for (i = 0; i < ort->output_name_count; i++) {
        char *tmp = AX_NULL;
        s = api->SessionGetOutputName(ort->session, i, ort->allocator, &tmp);
        if (s) {
            fprintf(stderr, "[ORT] SessionGetOutputName[%zu]: %s\n", i, api->GetErrorMessage(s));
            api->ReleaseStatus(s);
            return -1;
        }
        ort->output_names[i] = strdup(tmp);
        ort->allocator->Free(ort->allocator, tmp);
        if (!ort->output_names[i]) return -1;
    }

    return 0;
}

static AX_S32 ort_engine_init(OrtEngine *ort, const AX_CHAR *model_path,
                                AX_S32 t_model, AX_S32 f_freqs, AX_S32 f_int,
                                AX_S32 step, AX_S32 model_type) {
    ort->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort->api) {
        fprintf(stderr, "[ORT] Failed to get ORT API\n");
        return -1;
    }
    const OrtApi *api = ort->api;

    ORT_CHECK_INIT(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                        "se_unified", &ort->env));
    ORT_CHECK_INIT(api, api->CreateSessionOptions(&ort->session_opts));
    api->SetIntraOpNumThreads(ort->session_opts, 1);
    api->SetSessionGraphOptimizationLevel(ort->session_opts, ORT_ENABLE_ALL);

    ORT_CHECK_INIT(api, api->CreateSession(ort->env, model_path,
                                            ort->session_opts, &ort->session));
    ORT_CHECK_INIT(api, api->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &ort->memory_info));
    if (ort_cache_io_names(ort) != 0) {
        return -1;
    }

    ort->model_type = model_type;

    if (model_type == SE_MODEL_TYPE_GTCRN_STREAM) {
        ort->gtcrn_step = step > 0 ? step : 1;
        if (ort->input_name_count != 7 || ort->output_name_count != 7) {
            fprintf(stderr, "[ORT] gtcrn_stream now requires 7 inputs and 7 outputs, got inputs=%zu outputs=%zu\n",
                    ort->input_name_count, ort->output_name_count);
            return -1;
        }
        ort->gtcrn_mix_buf = (AX_F32*)calloc((size_t)ort->gtcrn_step * (size_t)f_freqs * 2u, sizeof(AX_F32));
        ort->gtcrn_en_conv_cache = (AX_F32*)calloc(GTCRN_EN_CONV_CACHE_ELEMS, sizeof(AX_F32));
        ort->gtcrn_de_conv_cache = (AX_F32*)calloc(GTCRN_DE_CONV_CACHE_ELEMS, sizeof(AX_F32));
        ort->gtcrn_en_tra_cache = (AX_F32*)calloc(GTCRN_EN_TRA_CACHE_ELEMS, sizeof(AX_F32));
        ort->gtcrn_de_tra_cache = (AX_F32*)calloc(GTCRN_DE_TRA_CACHE_ELEMS, sizeof(AX_F32));
        ort->gtcrn_inter_cache_0 = (AX_F32*)calloc(GTCRN_INTER_CACHE_0_ELEMS, sizeof(AX_F32));
        ort->gtcrn_inter_cache_1 = (AX_F32*)calloc(GTCRN_INTER_CACHE_1_ELEMS, sizeof(AX_F32));
        if (!ort->gtcrn_en_conv_cache || !ort->gtcrn_de_conv_cache ||
            !ort->gtcrn_en_tra_cache || !ort->gtcrn_de_tra_cache ||
            !ort->gtcrn_inter_cache_0 || !ort->gtcrn_inter_cache_1 ||
            !ort->gtcrn_mix_buf) {
            fprintf(stderr, "[ORT] GTCRN cache malloc failed\n");
            return -1;
        }
        printf("[OrtEngine] ONNX Runtime initialized\n");
        printf("            Model:  %s\n", model_path);
        printf("            Type:   gtcrn_stream 7-input cache\n");
        printf("            Inputs: %zu  Outputs: %zu\n", ort->input_name_count, ort->output_name_count);
        return 0;
    }

    if (model_type == SE_MODEL_TYPE_GTCRN_SPLIT) {
        ort->gtcrn_step = step > 0 ? step : 1;
        if (ort->input_name_count != 15 || ort->output_name_count != 15) {
            fprintf(stderr, "[ORT] gtcrn_split requires 15 inputs and 15 outputs, got %zu/%zu\n",
                    ort->input_name_count, ort->output_name_count);
            return -1;
        }
        ort->gtcrn_mix_buf = (AX_F32*)calloc((size_t)ort->gtcrn_step * (size_t)f_freqs * 2u, sizeof(AX_F32));
        ort->gtcrn_en_conv_cache = (AX_F32*)calloc(GTCRN_EN_CONV_CACHE_ELEMS, sizeof(AX_F32));
        ort->gtcrn_de_conv_cache = (AX_F32*)calloc(GTCRN_DE_CONV_CACHE_ELEMS, sizeof(AX_F32));
        ort->gtcrn_en_tra_cache  = (AX_F32*)calloc(GTCRN_EN_TRA_CACHE_ELEMS, sizeof(AX_F32));
        ort->gtcrn_de_tra_cache  = (AX_F32*)calloc(GTCRN_DE_TRA_CACHE_ELEMS, sizeof(AX_F32));
        ort->gtcrn_inter_cache_0 = (AX_F32*)calloc(GTCRN_SPLIT_INTER_ELEMS, sizeof(AX_F32));
        ort->gtcrn_inter_cache_1 = (AX_F32*)calloc(GTCRN_SPLIT_INTER_ELEMS, sizeof(AX_F32));
        if (!ort->gtcrn_en_conv_cache || !ort->gtcrn_de_conv_cache ||
            !ort->gtcrn_en_tra_cache  || !ort->gtcrn_de_tra_cache  ||
            !ort->gtcrn_inter_cache_0 || !ort->gtcrn_inter_cache_1 ||
            !ort->gtcrn_mix_buf) {
            fprintf(stderr, "[ORT] GTCRN split cache malloc failed\n");
            return -1;
        }
        printf("[OrtEngine] ONNX Runtime initialized\n");
        printf("            Model:  %s\n", model_path);
        printf("            Type:   gtcrn_split 15-input split cache\n");
        printf("            Inputs: %zu  Outputs: %zu\n", ort->input_name_count, ort->output_name_count);
        return 0;
    }

    if (model_type == SE_MODEL_TYPE_SELF_CACHE) {
        AX_S32 context = t_model - step;
        if (context < 0) {
            fprintf(stderr, "[ORT] invalid self_cache shape: t_model=%d step=%d\n", t_model, step);
            return -1;
        }
        ort->self_context = context;
        ort->self_step = step;
        ort->self_f_freqs = f_freqs;
        ort->self_feat_cache = (AX_F32*)calloc((size_t)context * (size_t)f_freqs, sizeof(AX_F32));
        ort->self_mix_buf = (AX_F32*)calloc((size_t)step * (size_t)f_freqs * 2u, sizeof(AX_F32));
        if (!ort->self_feat_cache || !ort->self_mix_buf) {
            fprintf(stderr, "[ORT] self_cache malloc failed\n");
            return -1;
        }
        printf("[OrtEngine] ONNX Runtime initialized\n");
        printf("            Model:  %s\n", model_path);
        printf("            Type:   self_cache\n");
        printf("            Input:  mix(1,%d,%d,2) + feat_cache(1,1,%d,%d)\n",
               f_freqs, step, context, f_freqs);
        printf("            Output: enh(1,%d,%d,2) + feat_cache_out\n", f_freqs, step);
        return 0;
    }

    ort->input_shape[0]  = 1;
    ort->input_shape[1]  = SE_V5_FEAT_CH;   /* 1 */
    ort->input_shape[2]  = t_model;
    ort->input_shape[3]  = f_freqs;
    ort->output_shape[0] = 1;
    ort->output_shape[1] = SE_V5_MASK_CH;   /* 1 */
    ort->output_shape[2] = t_model;
    ort->output_shape[3] = f_int;

    ort->input_elem_count  = (size_t)(SE_V5_FEAT_CH * t_model * f_freqs);
    ort->output_elem_count = (size_t)(SE_V5_MASK_CH * t_model * f_int);

    ort->output_buf = (AX_F32*)malloc(ort->output_elem_count * sizeof(AX_F32));
    if (!ort->output_buf) {
        fprintf(stderr, "[ORT] output_buf malloc failed\n");
        return -1;
    }

    printf("[OrtEngine] ONNX Runtime initialized\n");
    printf("            Model:  %s\n", model_path);
    printf("            Input:  (1, %d, %d, %d)  %zu floats\n",
           SE_V5_FEAT_CH, t_model, f_freqs, ort->input_elem_count);
    printf("            Output: (1, %d, %d, %d)  %zu floats\n",
           SE_V5_MASK_CH, t_model, f_int, ort->output_elem_count);

    return 0;
}

static AX_S32 ort_engine_run_gtcrn_chunk(OrtEngine *ort,
                                           const AX_F32 *spec_real,
                                           const AX_F32 *spec_imag,
                                           AX_S32 step,
                                           AX_F32 *enh_real,
                                           AX_F32 *enh_imag) {
    const OrtApi *api = ort->api;
    AX_S32 f_freqs = SE_V5_N_FREQS;

    for (AX_S32 k = 0; k < f_freqs; k++) {
        for (AX_S32 sidx = 0; sidx < step; sidx++) {
            size_t dst = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
            size_t src = (size_t)sidx * (size_t)f_freqs + (size_t)k;
            ort->gtcrn_mix_buf[dst] = spec_real[src];
            ort->gtcrn_mix_buf[dst + 1] = spec_imag[src];
        }
    }

    int64_t mix_shape[4] = {1, f_freqs, step, 2};
    int64_t conv_shape[4] = {1, 16, 16, 33};
    int64_t tra_shape[4] = {1, 3, 1, 16};
    int64_t inter_shape[4] = {1, 1, 33, 16};
    OrtValue *inputs[7] = {0};
    OrtValue *outputs[7] = {0};
    OrtStatus *s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->gtcrn_mix_buf, (size_t)step * (size_t)f_freqs * 2u * sizeof(AX_F32), mix_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[0]);
    if (s) goto ort_err;
    s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->gtcrn_en_conv_cache,
        GTCRN_EN_CONV_CACHE_ELEMS * sizeof(AX_F32), conv_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[1]);
    if (s) goto ort_err;
    s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->gtcrn_de_conv_cache,
        GTCRN_DE_CONV_CACHE_ELEMS * sizeof(AX_F32), conv_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[2]);
    if (s) goto ort_err;
    s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->gtcrn_en_tra_cache,
        GTCRN_EN_TRA_CACHE_ELEMS * sizeof(AX_F32), tra_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[3]);
    if (s) goto ort_err;
    s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->gtcrn_de_tra_cache,
        GTCRN_DE_TRA_CACHE_ELEMS * sizeof(AX_F32), tra_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[4]);
    if (s) goto ort_err;
    s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->gtcrn_inter_cache_0,
        GTCRN_INTER_CACHE_0_ELEMS * sizeof(AX_F32), inter_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[5]);
    if (s) goto ort_err;
    s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->gtcrn_inter_cache_1,
        GTCRN_INTER_CACHE_1_ELEMS * sizeof(AX_F32), inter_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[6]);
    if (s) goto ort_err;

    {
        const char *input_names[7] = {
            "mix", "en_conv_cache", "de_conv_cache",
            "en_tra_cache", "de_tra_cache", "inter_cache_0", "inter_cache_1"
        };
        const char *output_names[7] = {
            "enh", "en_conv_cache_out", "de_conv_cache_out",
            "en_tra_cache_out", "de_tra_cache_out", "inter_cache_0_out", "inter_cache_1_out"
        };
        s = api->Run(ort->session, AX_NULL,
                     input_names, (const OrtValue* const*)inputs, 7,
                     output_names, 7, outputs);
    }
    if (s) goto ort_err;

    AX_F32 *enh = AX_NULL;
    AX_F32 *en_cc = AX_NULL;
    AX_F32 *de_cc = AX_NULL;
    AX_F32 *en_tc = AX_NULL;
    AX_F32 *de_tc = AX_NULL;
    AX_F32 *ic0 = AX_NULL;
    AX_F32 *ic1 = AX_NULL;
    s = api->GetTensorMutableData(outputs[0], (AX_VOID**)&enh);
    if (s) goto ort_err;
    s = api->GetTensorMutableData(outputs[1], (AX_VOID**)&en_cc);
    if (s) goto ort_err;
    s = api->GetTensorMutableData(outputs[2], (AX_VOID**)&de_cc);
    if (s) goto ort_err;
    s = api->GetTensorMutableData(outputs[3], (AX_VOID**)&en_tc);
    if (s) goto ort_err;
    s = api->GetTensorMutableData(outputs[4], (AX_VOID**)&de_tc);
    if (s) goto ort_err;
    s = api->GetTensorMutableData(outputs[5], (AX_VOID**)&ic0);
    if (s) goto ort_err;
    s = api->GetTensorMutableData(outputs[6], (AX_VOID**)&ic1);
    if (s) goto ort_err;

    for (AX_S32 k = 0; k < f_freqs; k++) {
        for (AX_S32 sidx = 0; sidx < step; sidx++) {
            size_t src = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
            size_t dst = (size_t)sidx * (size_t)f_freqs + (size_t)k;
            enh_real[dst] = enh[src];
            enh_imag[dst] = enh[src + 1];
        }
    }
    memcpy(ort->gtcrn_en_conv_cache, en_cc, GTCRN_EN_CONV_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(ort->gtcrn_de_conv_cache, de_cc, GTCRN_DE_CONV_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(ort->gtcrn_en_tra_cache, en_tc, GTCRN_EN_TRA_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(ort->gtcrn_de_tra_cache, de_tc, GTCRN_DE_TRA_CACHE_ELEMS * sizeof(AX_F32));
    memcpy(ort->gtcrn_inter_cache_0, ic0, GTCRN_INTER_CACHE_0_ELEMS * sizeof(AX_F32));
    memcpy(ort->gtcrn_inter_cache_1, ic1, GTCRN_INTER_CACHE_1_ELEMS * sizeof(AX_F32));

    for (AX_S32 i = 0; i < 7; i++) {
        if (inputs[i]) api->ReleaseValue(inputs[i]);
        if (outputs[i]) api->ReleaseValue(outputs[i]);
    }
    return 0;

ort_err:
    if (s) {
        fprintf(stderr, "[ORT] GTCRN Run: %s\n", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
    }
    for (AX_S32 i = 0; i < 7; i++) {
        if (inputs[i]) api->ReleaseValue(inputs[i]);
        if (outputs[i]) api->ReleaseValue(outputs[i]);
    }
    return -1;
}

static AX_S32 ort_engine_run_gtcrn_split_chunk(OrtEngine *ort,
                                                const AX_F32 *spec_real,
                                                const AX_F32 *spec_imag,
                                                AX_S32 step,
                                                AX_F32 *enh_real,
                                                AX_F32 *enh_imag) {
    const OrtApi *api = ort->api;
    AX_S32 f_freqs = SE_V5_N_FREQS;

    for (AX_S32 k = 0; k < f_freqs; k++) {
        for (AX_S32 sidx = 0; sidx < step; sidx++) {
            size_t dst = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
            size_t src = (size_t)sidx * (size_t)f_freqs + (size_t)k;
            ort->gtcrn_mix_buf[dst]     = spec_real[src];
            ort->gtcrn_mix_buf[dst + 1] = spec_imag[src];
        }
    }

    AX_F32 *en_cc0 = ort->gtcrn_en_conv_cache;
    AX_F32 *en_cc1 = ort->gtcrn_en_conv_cache + GTCRN_SPLIT_EN_CC0_ELEMS;
    AX_F32 *en_cc2 = ort->gtcrn_en_conv_cache + GTCRN_SPLIT_EN_CC0_ELEMS + GTCRN_SPLIT_EN_CC1_ELEMS;
    AX_F32 *de_cc0 = ort->gtcrn_de_conv_cache;
    AX_F32 *de_cc1 = ort->gtcrn_de_conv_cache + GTCRN_SPLIT_DE_CC0_ELEMS;
    AX_F32 *de_cc2 = ort->gtcrn_de_conv_cache + GTCRN_SPLIT_DE_CC0_ELEMS + GTCRN_SPLIT_DE_CC1_ELEMS;
    AX_F32 *en_tc0 = ort->gtcrn_en_tra_cache;
    AX_F32 *en_tc1 = ort->gtcrn_en_tra_cache + GTCRN_SPLIT_TRA_ELEMS;
    AX_F32 *en_tc2 = ort->gtcrn_en_tra_cache + 2 * GTCRN_SPLIT_TRA_ELEMS;
    AX_F32 *de_tc0 = ort->gtcrn_de_tra_cache;
    AX_F32 *de_tc1 = ort->gtcrn_de_tra_cache + GTCRN_SPLIT_TRA_ELEMS;
    AX_F32 *de_tc2 = ort->gtcrn_de_tra_cache + 2 * GTCRN_SPLIT_TRA_ELEMS;

    int64_t mix_shape[4]   = {1, f_freqs, step, 2};
    int64_t en_cc0_sh[4]   = {1, 16, 2,  33};
    int64_t en_cc1_sh[4]   = {1, 16, 4,  33};
    int64_t en_cc2_sh[4]   = {1, 16, 10, 33};
    int64_t de_cc0_sh[4]   = {1, 16, 10, 33};
    int64_t de_cc1_sh[4]   = {1, 16, 4,  33};
    int64_t de_cc2_sh[4]   = {1, 16, 2,  33};
    int64_t tra_sh[3]       = {1, 1, 16};
    int64_t inter_sh[3]     = {1, 33, 16};

    OrtValue *inputs[15]  = {0};
    OrtValue *outputs[15] = {0};
    OrtStatus *s = AX_NULL;

    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, ort->gtcrn_mix_buf,
        (size_t)step * (size_t)f_freqs * 2u * sizeof(AX_F32), mix_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[0]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, en_cc0,
        GTCRN_SPLIT_EN_CC0_ELEMS * sizeof(AX_F32), en_cc0_sh, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[1]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, en_cc1,
        GTCRN_SPLIT_EN_CC1_ELEMS * sizeof(AX_F32), en_cc1_sh, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[2]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, en_cc2,
        GTCRN_SPLIT_EN_CC2_ELEMS * sizeof(AX_F32), en_cc2_sh, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[3]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, de_cc0,
        GTCRN_SPLIT_DE_CC0_ELEMS * sizeof(AX_F32), de_cc0_sh, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[4]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, de_cc1,
        GTCRN_SPLIT_DE_CC1_ELEMS * sizeof(AX_F32), de_cc1_sh, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[5]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, de_cc2,
        GTCRN_SPLIT_DE_CC2_ELEMS * sizeof(AX_F32), de_cc2_sh, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[6]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, en_tc0,
        GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32), tra_sh, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[7]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, en_tc1,
        GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32), tra_sh, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[8]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, en_tc2,
        GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32), tra_sh, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[9]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, de_tc0,
        GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32), tra_sh, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[10]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, de_tc1,
        GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32), tra_sh, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[11]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, de_tc2,
        GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32), tra_sh, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[12]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, ort->gtcrn_inter_cache_0,
        GTCRN_SPLIT_INTER_ELEMS * sizeof(AX_F32), inter_sh, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[13]);
    if (s) goto split_err;
    s = api->CreateTensorWithDataAsOrtValue(ort->memory_info, ort->gtcrn_inter_cache_1,
        GTCRN_SPLIT_INTER_ELEMS * sizeof(AX_F32), inter_sh, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[14]);
    if (s) goto split_err;

    {
        const char *in_names[15] = {
            "mix",
            "en_conv_cache_0", "en_conv_cache_1", "en_conv_cache_2",
            "de_conv_cache_0", "de_conv_cache_1", "de_conv_cache_2",
            "en_tra_cache_0",  "en_tra_cache_1",  "en_tra_cache_2",
            "de_tra_cache_0",  "de_tra_cache_1",  "de_tra_cache_2",
            "inter_cache_0",   "inter_cache_1"
        };
        const char *out_names[15] = {
            "enh",
            "en_conv_cache_0_out", "en_conv_cache_1_out", "en_conv_cache_2_out",
            "de_conv_cache_0_out", "de_conv_cache_1_out", "de_conv_cache_2_out",
            "en_tra_cache_0_out",  "en_tra_cache_1_out",  "en_tra_cache_2_out",
            "de_tra_cache_0_out",  "de_tra_cache_1_out",  "de_tra_cache_2_out",
            "inter_cache_0_out",   "inter_cache_1_out"
        };
        s = api->Run(ort->session, AX_NULL,
                     in_names, (const OrtValue* const*)inputs, 15,
                     out_names, 15, outputs);
    }
    if (s) goto split_err;

    {
        AX_F32 *enh   = AX_NULL;
        AX_F32 *o_en0 = AX_NULL, *o_en1 = AX_NULL, *o_en2 = AX_NULL;
        AX_F32 *o_de0 = AX_NULL, *o_de1 = AX_NULL, *o_de2 = AX_NULL;
        AX_F32 *o_et0 = AX_NULL, *o_et1 = AX_NULL, *o_et2 = AX_NULL;
        AX_F32 *o_dt0 = AX_NULL, *o_dt1 = AX_NULL, *o_dt2 = AX_NULL;
        AX_F32 *o_ic0 = AX_NULL, *o_ic1 = AX_NULL;
        if (api->GetTensorMutableData(outputs[0], (AX_VOID**)&enh)   != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[1], (AX_VOID**)&o_en0) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[2], (AX_VOID**)&o_en1) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[3], (AX_VOID**)&o_en2) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[4], (AX_VOID**)&o_de0) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[5], (AX_VOID**)&o_de1) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[6], (AX_VOID**)&o_de2) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[7], (AX_VOID**)&o_et0) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[8], (AX_VOID**)&o_et1) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[9], (AX_VOID**)&o_et2) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[10],(AX_VOID**)&o_dt0) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[11],(AX_VOID**)&o_dt1) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[12],(AX_VOID**)&o_dt2) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[13],(AX_VOID**)&o_ic0) != AX_NULL) goto split_err;
        if (api->GetTensorMutableData(outputs[14],(AX_VOID**)&o_ic1) != AX_NULL) goto split_err;

        for (AX_S32 k = 0; k < f_freqs; k++) {
            for (AX_S32 sidx = 0; sidx < step; sidx++) {
                size_t src = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
                size_t dst = (size_t)sidx * (size_t)f_freqs + (size_t)k;
                enh_real[dst] = enh[src];
                enh_imag[dst] = enh[src + 1];
            }
        }
        memcpy(en_cc0, o_en0, GTCRN_SPLIT_EN_CC0_ELEMS * sizeof(AX_F32));
        memcpy(en_cc1, o_en1, GTCRN_SPLIT_EN_CC1_ELEMS * sizeof(AX_F32));
        memcpy(en_cc2, o_en2, GTCRN_SPLIT_EN_CC2_ELEMS * sizeof(AX_F32));
        memcpy(de_cc0, o_de0, GTCRN_SPLIT_DE_CC0_ELEMS * sizeof(AX_F32));
        memcpy(de_cc1, o_de1, GTCRN_SPLIT_DE_CC1_ELEMS * sizeof(AX_F32));
        memcpy(de_cc2, o_de2, GTCRN_SPLIT_DE_CC2_ELEMS * sizeof(AX_F32));
        memcpy(en_tc0, o_et0, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
        memcpy(en_tc1, o_et1, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
        memcpy(en_tc2, o_et2, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
        memcpy(de_tc0, o_dt0, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
        memcpy(de_tc1, o_dt1, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
        memcpy(de_tc2, o_dt2, GTCRN_SPLIT_TRA_ELEMS * sizeof(AX_F32));
        memcpy(ort->gtcrn_inter_cache_0, o_ic0, GTCRN_SPLIT_INTER_ELEMS * sizeof(AX_F32));
        memcpy(ort->gtcrn_inter_cache_1, o_ic1, GTCRN_SPLIT_INTER_ELEMS * sizeof(AX_F32));
    }

    for (AX_S32 i = 0; i < 15; i++) {
        if (inputs[i])  api->ReleaseValue(inputs[i]);
        if (outputs[i]) api->ReleaseValue(outputs[i]);
    }
    return 0;

split_err:
    if (s) {
        fprintf(stderr, "[ORT] GTCRN split Run: %s\n", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
    }
    for (AX_S32 i = 0; i < 15; i++) {
        if (inputs[i])  api->ReleaseValue(inputs[i]);
        if (outputs[i]) api->ReleaseValue(outputs[i]);
    }
    return -1;
}

static AX_S32 ort_engine_run_self_cache_chunk(OrtEngine *ort,
                                                const AX_F32 *spec_real,
                                                const AX_F32 *spec_imag,
                                                AX_S32 step,
                                                AX_F32 *enh_real,
                                                AX_F32 *enh_imag) {
    const OrtApi *api = ort->api;
    AX_S32 F = ort->self_f_freqs;
    AX_S32 context = ort->self_context;

    for (AX_S32 k = 0; k < F; k++) {
        for (AX_S32 sidx = 0; sidx < step; sidx++) {
            size_t dst = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
            size_t src = (size_t)sidx * (size_t)F + (size_t)k;
            ort->self_mix_buf[dst] = spec_real[src];
            ort->self_mix_buf[dst + 1] = spec_imag[src];
        }
    }

    int64_t mix_shape[4] = {1, F, step, 2};
    int64_t cache_shape[4] = {1, 1, context, F};
    OrtValue *inputs[2] = {0};
    OrtValue *outputs[2] = {0};
    OrtStatus *s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->self_mix_buf,
        (size_t)step * (size_t)F * 2u * sizeof(AX_F32), mix_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[0]);
    if (s) goto ort_err;
    s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info, ort->self_feat_cache,
        (size_t)context * (size_t)F * sizeof(AX_F32), cache_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[1]);
    if (s) goto ort_err;

    {
        const char *input_names[2] = {"mix", "feat_cache"};
        const char *output_names[2] = {"enh", "feat_cache_out"};
        s = api->Run(ort->session, AX_NULL,
                     input_names, (const OrtValue* const*)inputs, 2,
                     output_names, 2, outputs);
    }
    if (s) goto ort_err;

    AX_F32 *enh = AX_NULL;
    AX_F32 *cache_out = AX_NULL;
    s = api->GetTensorMutableData(outputs[0], (AX_VOID**)&enh);
    if (s) goto ort_err;
    s = api->GetTensorMutableData(outputs[1], (AX_VOID**)&cache_out);
    if (s) goto ort_err;

    for (AX_S32 k = 0; k < F; k++) {
        for (AX_S32 sidx = 0; sidx < step; sidx++) {
            size_t src = ((size_t)k * (size_t)step + (size_t)sidx) * 2u;
            size_t dst = (size_t)sidx * (size_t)F + (size_t)k;
            enh_real[dst] = enh[src];
            enh_imag[dst] = enh[src + 1];
        }
    }
    memcpy(ort->self_feat_cache, cache_out, (size_t)context * (size_t)F * sizeof(AX_F32));

    for (AX_S32 i = 0; i < 2; i++) {
        if (inputs[i]) api->ReleaseValue(inputs[i]);
        if (outputs[i]) api->ReleaseValue(outputs[i]);
    }
    return 0;

ort_err:
    if (s) {
        fprintf(stderr, "[ORT] self_cache Run: %s\n", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
    }
    for (AX_S32 i = 0; i < 2; i++) {
        if (inputs[i]) api->ReleaseValue(inputs[i]);
        if (outputs[i]) api->ReleaseValue(outputs[i]);
    }
    return -1;
}

static AX_S32 ort_engine_run(OrtEngine *ort,
                               const AX_F32 *input_data,
                               AX_F32 **output_data) {
    const OrtApi *api = ort->api;

    OrtValue *input_tensor  = AX_NULL;
    OrtValue *output_tensor = AX_NULL;

    OrtStatus *s = api->CreateTensorWithDataAsOrtValue(
        ort->memory_info,
        (AX_VOID*)input_data,
        ort->input_elem_count * sizeof(AX_F32),
        ort->input_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor);
    if (s) {
        fprintf(stderr, "[ORT] CreateTensor: %s\n", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        return -1;
    }

    {
        const char *input_names[]  = {"input"};
        const char *output_names[] = {"output"};
        s = api->Run(ort->session, AX_NULL,
                     input_names,  (const OrtValue* const*)&input_tensor,  1,
                     output_names, 1,
                     &output_tensor);
    }
    if (s) {
        fprintf(stderr, "[ORT] Run: %s\n", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        api->ReleaseValue(input_tensor);
        return -1;
    }

    AX_F32 *out_ptr = AX_NULL;
    s = api->GetTensorMutableData(output_tensor, (AX_VOID**)&out_ptr);
    if (s) {
        fprintf(stderr, "[ORT] GetTensorMutableData: %s\n",
                api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        api->ReleaseValue(input_tensor);
        api->ReleaseValue(output_tensor);
        return -1;
    }

    memcpy(ort->output_buf, out_ptr, ort->output_elem_count * sizeof(AX_F32));

    api->ReleaseValue(input_tensor);
    api->ReleaseValue(output_tensor);

    *output_data = ort->output_buf;
    return 0;
}

static AX_VOID ort_engine_free(OrtEngine *ort) {
    const OrtApi *api = ort->api;
    size_t i;
    if (!api) return;
    if (ort->input_names) {
        for (i = 0; i < ort->input_name_count; i++) free(ort->input_names[i]);
        free(ort->input_names);
        ort->input_names = AX_NULL;
    }
    if (ort->output_names) {
        for (i = 0; i < ort->output_name_count; i++) free(ort->output_names[i]);
        free(ort->output_names);
        ort->output_names = AX_NULL;
    }
    if (ort->output_buf)    { free(ort->output_buf); ort->output_buf = AX_NULL; }
    if (ort->gtcrn_en_conv_cache) { free(ort->gtcrn_en_conv_cache); ort->gtcrn_en_conv_cache = AX_NULL; }
    if (ort->gtcrn_de_conv_cache) { free(ort->gtcrn_de_conv_cache); ort->gtcrn_de_conv_cache = AX_NULL; }
    if (ort->gtcrn_en_tra_cache)  { free(ort->gtcrn_en_tra_cache); ort->gtcrn_en_tra_cache = AX_NULL; }
    if (ort->gtcrn_de_tra_cache)  { free(ort->gtcrn_de_tra_cache); ort->gtcrn_de_tra_cache = AX_NULL; }
    if (ort->gtcrn_inter_cache_0) { free(ort->gtcrn_inter_cache_0); ort->gtcrn_inter_cache_0 = AX_NULL; }
    if (ort->gtcrn_inter_cache_1) { free(ort->gtcrn_inter_cache_1); ort->gtcrn_inter_cache_1 = AX_NULL; }
    if (ort->gtcrn_mix_buf)       { free(ort->gtcrn_mix_buf); ort->gtcrn_mix_buf = AX_NULL; }
    if (ort->self_feat_cache)   { free(ort->self_feat_cache); ort->self_feat_cache = AX_NULL; }
    if (ort->self_mix_buf)      { free(ort->self_mix_buf); ort->self_mix_buf = AX_NULL; }
    if (ort->memory_info)   { api->ReleaseMemoryInfo(ort->memory_info); }
    if (ort->session)       { api->ReleaseSession(ort->session); }
    if (ort->session_opts)  { api->ReleaseSessionOptions(ort->session_opts); }
    if (ort->env)           { api->ReleaseEnv(ort->env); }
}

static AX_VOID build_model_input(SeUnifiedState *state) {
    AX_S32 F  = state->f_freqs;
    AX_S32 Tc = state->context;
    AX_S32 S  = state->step;

    AX_F32 *ch0 = state->model_input_buf;
    AX_F32 feat_tmp[SE_V5_N_FREQS];

    for (AX_S32 t = 0; t < Tc; t++) {
        const AX_F32 *sr = state->spec_ctx_real + t * F;
        const AX_F32 *si = state->spec_ctx_imag + t * F;
        tiny_se_v5_extract_feat(sr, si, feat_tmp);
        memcpy(ch0 + t * F, feat_tmp, (size_t)F * sizeof(AX_F32));
    }

    for (AX_S32 s = 0; s < S; s++) {
        const AX_F32 *sr = state->spec_step_real + s * F;
        const AX_F32 *si = state->spec_step_imag + s * F;
        tiny_se_v5_extract_feat(sr, si, feat_tmp);
        memcpy(ch0 + (Tc + s) * F, feat_tmp, (size_t)F * sizeof(AX_F32));
    }
}

static AX_VOID update_context(SeUnifiedState *state) {
    AX_S32 F  = state->f_freqs;
    AX_S32 Tc = state->context;
    AX_S32 S  = state->step;

    memmove(state->spec_ctx_real,
            state->spec_ctx_real + S * F,
            (size_t)(Tc - S) * (size_t)F * sizeof(AX_F32));
    memmove(state->spec_ctx_imag,
            state->spec_ctx_imag + S * F,
            (size_t)(Tc - S) * (size_t)F * sizeof(AX_F32));

    memcpy(state->spec_ctx_real + (Tc - S) * F,
           state->spec_step_real, (size_t)S * (size_t)F * sizeof(AX_F32));
    memcpy(state->spec_ctx_imag + (Tc - S) * F,
           state->spec_step_imag, (size_t)S * (size_t)F * sizeof(AX_F32));
}

/* ═══════════════════════════════════════════════════════════════
 *  公共 API 实现
 * ═══════════════════════════════════════════════════════════════ */

AX_S32 AX_AI_SE_Init(AX_VOID **handle,
                      const AX_AI_SE_Config *config,
                      const AX_CHAR *model_path) {
    if (!handle || !model_path) {
        fprintf(stderr, "[AX_AI_SE] Init: invalid arguments\n");
        return -1;
    }

    SeUnifiedState *state = (SeUnifiedState*)calloc(1, sizeof(SeUnifiedState));
    if (!state) {
        fprintf(stderr, "[AX_AI_SE] Init: calloc failed\n");
        return -1;
    }

    if (config) {
        state->hop_len = config->hop_len > 0 ? config->hop_len : SE_V5_HOP_LEN;
        state->t_model = config->t_model > 0 ? config->t_model : SE_V5_T_MODEL;
        state->f_freqs = config->f_freqs > 0 ? config->f_freqs : SE_V5_N_FREQS;
        state->f_int   = config->f_int   > 0 ? config->f_int   : SE_V5_F_INT;
        state->step    = config->step    > 0 ? config->step    : SE_V5_STEP;
        state->model_type = config->model_type;
    } else {
        state->hop_len = SE_V5_HOP_LEN;
        state->t_model = SE_V5_T_MODEL;
        state->f_freqs = SE_V5_N_FREQS;
        state->f_int   = SE_V5_F_INT;
        state->step    = SE_V5_STEP;
        state->model_type = SE_MODEL_TYPE_MASK_BATCH;
    }
    state->context = (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
                      state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT)
        ? 0
        : (state->t_model - state->step);

    printf("[AX_AI_SE] hop_len=%d  t_model=%d  f_freqs=%d  f_int=%d  context=%d  step=%d  model_type=%d\n",
           state->hop_len, state->t_model, state->f_freqs, state->f_int,
           state->context, state->step, state->model_type);

    AX_S32 dsp_ret = (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
                      state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT)
        ? tiny_se_v5_dsp_init_sqrt_hann(&state->dsp)
        : tiny_se_v5_dsp_init(&state->dsp);
    if (dsp_ret != 0) {
        fprintf(stderr, "[AX_AI_SE] DSP init failed\n");
        free(state);
        return -1;
    }
    printf("[AX_AI_SE] DSP initialized\n");

    if (ort_engine_init(&state->ort, model_path,
                         state->t_model, state->f_freqs, state->f_int, state->step,
                         state->model_type) != 0) {
        fprintf(stderr, "[AX_AI_SE] ORT engine init failed\n");
        tiny_se_v5_dsp_free(&state->dsp);
        free(state);
        return -1;
    }

    AX_S32 ctx_size = state->context * state->f_freqs;
    state->spec_ctx_real = (AX_F32*)calloc((size_t)ctx_size, sizeof(AX_F32));
    state->spec_ctx_imag = (AX_F32*)calloc((size_t)ctx_size, sizeof(AX_F32));
    if (!state->spec_ctx_real || !state->spec_ctx_imag) goto err;

    AX_S32 step_size = state->step * state->f_freqs;
    state->spec_step_real = (AX_F32*)calloc((size_t)step_size, sizeof(AX_F32));
    state->spec_step_imag = (AX_F32*)calloc((size_t)step_size, sizeof(AX_F32));
    if (!state->spec_step_real || !state->spec_step_imag) goto err;

    AX_S32 input_size = SE_V5_FEAT_CH * state->t_model * state->f_freqs;
    state->model_input_buf = (AX_F32*)calloc((size_t)input_size, sizeof(AX_F32));
    if (!state->model_input_buf) goto err;

    /* mask_full_buf 大小为 f_freqs，支持任意模型 */
    state->mask_full_buf = (AX_F32*)calloc((size_t)state->f_freqs, sizeof(AX_F32));
    if (!state->mask_full_buf) goto err;

    if (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
        state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT ||
        state->model_type == SE_MODEL_TYPE_SELF_CACHE) {
        AX_S32 enh_size = state->step * state->f_freqs;
        state->gtcrn_enh_real = (AX_F32*)calloc((size_t)enh_size, sizeof(AX_F32));
        state->gtcrn_enh_imag = (AX_F32*)calloc((size_t)enh_size, sizeof(AX_F32));
        if (!state->gtcrn_enh_real || !state->gtcrn_enh_imag) goto err;
    }

    state->frame_count = 0;
    *handle = (AX_VOID*)state;

    printf("[AX_AI_SE] Initialization complete\n\n");
    return 0;

err:
    if (state->spec_ctx_real)   free(state->spec_ctx_real);
    if (state->spec_ctx_imag)   free(state->spec_ctx_imag);
    if (state->spec_step_real)  free(state->spec_step_real);
    if (state->spec_step_imag)  free(state->spec_step_imag);
    if (state->model_input_buf) free(state->model_input_buf);
    if (state->mask_full_buf)   free(state->mask_full_buf);
    if (state->gtcrn_enh_real)  free(state->gtcrn_enh_real);
    if (state->gtcrn_enh_imag)  free(state->gtcrn_enh_imag);
    ort_engine_free(&state->ort);
    tiny_se_v5_dsp_free(&state->dsp);
    free(state);
    return -1;
}

AX_VOID AX_AI_SE_PreProcess(AX_VOID *handle, AX_AI_SE_PreParams *params) {
    if (!handle || !params || !params->pcm_in_batch) return;
    SeUnifiedState *state = (SeUnifiedState*)handle;

    AX_S32 F   = state->f_freqs;
    AX_S32 S   = state->step;
    AX_S32 hop = state->hop_len;

    /* 对 step 帧依次做 STFT */
    for (AX_S32 s = 0; s < S; s++) {
        const AX_F32 *pcm = params->pcm_in_batch + s * hop;
        tiny_se_v5_stft_frame(&state->dsp, pcm,
                               state->spec_step_real + s * F,
                               state->spec_step_imag + s * F);
    }
    if (state->model_type == SE_MODEL_TYPE_MASK_BATCH &&
        state->frame_count == 0 && state->context > 0) {
        for (AX_S32 t = 0; t < state->context; t++) {
            memcpy(state->spec_ctx_real + t * F, state->spec_step_real, (size_t)F * sizeof(AX_F32));
            memcpy(state->spec_ctx_imag + t * F, state->spec_step_imag, (size_t)F * sizeof(AX_F32));
        }
    }
    state->frame_count += S;

    if (state->model_type == SE_MODEL_TYPE_MASK_BATCH) {
        /* 构建模型输入: context 帧 + step 帧 */
        build_model_input(state);
        params->model_input      = state->model_input_buf;
        params->model_input_size = SE_V5_FEAT_CH * state->t_model * state->f_freqs;
    } else {
        params->model_input = state->spec_step_real;
        params->model_input_size = state->step * state->f_freqs;
    }
}

AX_S32 AX_AI_SE_Infer(AX_VOID *handle, AX_AI_SE_InferParams *params) {
    if (!handle || !params) return -1;
    SeUnifiedState *state = (SeUnifiedState*)handle;

    if (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM) {
        if (ort_engine_run_gtcrn_chunk(
                &state->ort,
                state->spec_step_real,
                state->spec_step_imag,
                state->step,
                state->gtcrn_enh_real,
                state->gtcrn_enh_imag) != 0) {
            return -1;
        }
        params->model_output = state->gtcrn_enh_real;
        params->model_output_size = state->step * state->f_freqs * 2;
        return 0;
    }

    if (state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT) {
        if (ort_engine_run_gtcrn_split_chunk(
                &state->ort,
                state->spec_step_real,
                state->spec_step_imag,
                state->step,
                state->gtcrn_enh_real,
                state->gtcrn_enh_imag) != 0) {
            return -1;
        }
        params->model_output = state->gtcrn_enh_real;
        params->model_output_size = state->step * state->f_freqs * 2;
        return 0;
    }

    if (state->model_type == SE_MODEL_TYPE_SELF_CACHE) {
        if (ort_engine_run_self_cache_chunk(
                &state->ort,
                state->spec_step_real,
                state->spec_step_imag,
                state->step,
                state->gtcrn_enh_real,
                state->gtcrn_enh_imag) != 0) {
            return -1;
        }
        params->model_output = state->gtcrn_enh_real;
        params->model_output_size = state->step * state->f_freqs * 2;
        return 0;
    }

    if (!params->model_input) return -1;

    AX_F32 *output_data = AX_NULL;
    if (ort_engine_run(&state->ort, params->model_input, &output_data) != 0)
        return -1;

    params->model_output      = output_data;
    params->model_output_size = SE_V5_MASK_CH * state->t_model * state->f_int;
    return 0;
}

AX_VOID AX_AI_SE_PostProcess(AX_VOID *handle, AX_AI_SE_PostParams *params) {
    if (!handle || !params || !params->model_output || !params->pcm_out_batch) return;
    SeUnifiedState *state = (SeUnifiedState*)handle;

    if (state->model_type == SE_MODEL_TYPE_GTCRN_STREAM ||
        state->model_type == SE_MODEL_TYPE_GTCRN_SPLIT ||
        state->model_type == SE_MODEL_TYPE_SELF_CACHE) {
        for (AX_S32 s = 0; s < state->step; s++) {
            tiny_se_v5_istft_frame(&state->dsp,
                                   state->gtcrn_enh_real + s * state->f_freqs,
                                   state->gtcrn_enh_imag + s * state->f_freqs,
                                   params->pcm_out_batch + s * state->hop_len);
        }
        return;
    }

    AX_S32 Tc  = state->context;
    AX_S32 Fi  = state->f_int;
    AX_S32 F   = state->f_freqs;
    AX_S32 S   = state->step;
    AX_S32 hop = state->hop_len;

    const AX_F32 *out = params->model_output;

    for (AX_S32 s = 0; s < S; s++) {
        AX_F32 mask_int[SE_V5_N_FREQS];   /* 足够大，支持 f_int 最大到 257 */
        memcpy(mask_int, out + (Tc + s) * Fi, (size_t)Fi * sizeof(AX_F32));

        tiny_se_v5_sigmoid_inplace(mask_int, Fi);

        /* 使用运行时 f_int 参数的插值函数，支持任意模型 */
        tiny_se_v5_interp_mask_n(mask_int, Fi, state->mask_full_buf, F);

        AX_F32 enh_real[SE_V5_N_FREQS];
        AX_F32 enh_imag[SE_V5_N_FREQS];
        memcpy(enh_real, state->spec_step_real + s * F, (size_t)F * sizeof(AX_F32));
        memcpy(enh_imag, state->spec_step_imag + s * F, (size_t)F * sizeof(AX_F32));
        tiny_se_v5_apply_irm(enh_real, enh_imag, state->mask_full_buf);

        tiny_se_v5_istft_frame(&state->dsp, enh_real, enh_imag,
                                params->pcm_out_batch + s * hop);
    }

    update_context(state);
}

AX_VOID AX_AI_SE_Free(AX_VOID *handle) {
    if (!handle) return;
    SeUnifiedState *state = (SeUnifiedState*)handle;

    if (state->model_input_buf) free(state->model_input_buf);
    if (state->mask_full_buf)   free(state->mask_full_buf);
    if (state->gtcrn_enh_real)  free(state->gtcrn_enh_real);
    if (state->gtcrn_enh_imag)  free(state->gtcrn_enh_imag);
    if (state->spec_ctx_real)   free(state->spec_ctx_real);
    if (state->spec_ctx_imag)   free(state->spec_ctx_imag);
    if (state->spec_step_real)  free(state->spec_step_real);
    if (state->spec_step_imag)  free(state->spec_step_imag);

    ort_engine_free(&state->ort);
    tiny_se_v5_dsp_free(&state->dsp);

    free(state);
    printf("[AX_AI_SE] Resources freed\n");
}
