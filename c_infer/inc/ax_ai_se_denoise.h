/**
 * @file ax_ai_se_denoise.h
 * @brief 统一语音增强批量公共 API
 *
 * 支持多种模型，仅通过 INI 配置文件区分，无需重新编译:
 *
 *   模型              t_model  f_freqs  f_int  step  context
 *   ─────────────────────────────────────────────────────────
 *   tiny_conv_v5        34      257      17     6      28
 *   conv_gtcrn_small    64      257     129     6      58
 *
 * 处理流程 (批量版，每次处理 step 帧):
 *   1. AX_AI_SE_Init          → 初始化 DSP 状态 + 推理引擎
 *   2. AX_AI_SE_PreProcess    → 批量前处理 step 帧 (一次调用)
 *   3. AX_AI_SE_Infer         → 模型推理
 *   4. AX_AI_SE_PostProcess   → 批量后处理 step 帧 (一次调用)
 *   5. AX_AI_SE_Free          → 释放所有资源
 *
 * 通用参数 (所有模型相同):
 *   n_fft=512, hop_len=256, win_len=512
 *   输入: (1, 1, t_model, 257)  — [log1p_mag]
 *   输出: (1, 1, t_model, f_int) — IRM mask (压缩频率，无 sigmoid)
 *   采样率: 16000 Hz
 */

#ifndef AX_AI_SE_DENOISE_H
#define AX_AI_SE_DENOISE_H

#include "ax_base_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 *  配置结构体
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    AX_S32 hop_len;    /**< 帧移 (默认 256) */
    AX_S32 t_model;    /**< 模型时间维度 (tiny_v5=34, conv_gtcrn=64) */
    AX_S32 f_freqs;    /**< 模型输入频率维度 (默认 257) */
    AX_S32 f_int;      /**< 模型输出压缩频率维度 (tiny_v5=17, conv_gtcrn=129) */
    AX_S32 step;       /**< 每次推理处理的新帧数 (默认 6) */
    AX_S32 model_type; /**< 0=mask batch, 1=GTCRN stream cache, 2=self-developed cache */
} AX_AI_SE_Config;

/* ═══════════════════════════════════════════════════════════════
 *  参数结构体
 * ═══════════════════════════════════════════════════════════════ */

/**
 * @brief 批量前处理参数
 *
 * pcm_in_batch 指向连续存放的 step 帧 PCM:
 *   [0 .. hop_len-1]                     → 第 0 帧
 *   [hop_len .. 2*hop_len-1]             → 第 1 帧
 *   ...
 *   [(step-1)*hop_len .. step*hop_len-1] → 第 step-1 帧
 */
typedef struct {
    const AX_F32 *pcm_in_batch;      /**< [in]  step 帧连续 PCM */
    AX_F32       *model_input;       /**< [out] 模型输入 buffer */
    AX_S32        model_input_size;  /**< [out] float 个数 */
} AX_AI_SE_PreParams;

typedef struct {
    const AX_F32 *model_input;        /**< [in]  模型输入 */
    AX_F32       *model_output;       /**< [out] 模型输出 buffer */
    AX_S32        model_output_size;  /**< [out] float 个数 */
} AX_AI_SE_InferParams;

/**
 * @brief 批量后处理参数
 *
 * pcm_out_batch 指向连续存放的 step 帧 PCM 输出缓冲区:
 *   [0 .. hop_len-1]         → 第 0 帧输出
 *   [hop_len .. 2*hop_len-1] → 第 1 帧输出
 *   ...
 */
typedef struct {
    const AX_F32 *model_output;   /**< [in]  模型输出 */
    AX_F32       *pcm_out_batch;  /**< [out] step 帧连续 PCM 输出 */
} AX_AI_SE_PostParams;

/* ═══════════════════════════════════════════════════════════════
 *  公共 API
 * ═══════════════════════════════════════════════════════════════ */

/**
 * @brief 从 INI 文件加载模型配置
 *
 * 切换模型只需修改 INI 文件中的 t_model / f_int / model_path，
 * 无需重新编译。
 *
 * @param ini_path   INI 文件路径
 * @param config     [out] 填充的配置结构体
 * @param model_path [out] 模型路径缓冲区 (可为 NULL)
 * @param path_size  model_path 缓冲区大小
 * @return 0 成功, -1 文件打开失败 (仍使用默认值)
 */
AX_S32 AX_AI_SE_LoadConfig(const AX_CHAR *ini_path,
                             AX_AI_SE_Config *config,
                             AX_CHAR *model_path,
                             AX_S32   path_size);

AX_S32 AX_AI_SE_Init(AX_VOID **handle,
                      const AX_AI_SE_Config *config,
                      const AX_CHAR *model_path);

/**
 * @brief 批量前处理: 一次处理 step 帧 PCM → 构建模型输入
 * @param handle  句柄
 * @param params  pcm_in_batch 指向 step*hop_len 个连续 float
 */
AX_VOID AX_AI_SE_PreProcess(AX_VOID *handle, AX_AI_SE_PreParams *params);

AX_S32  AX_AI_SE_Infer(AX_VOID *handle, AX_AI_SE_InferParams *params);

/**
 * @brief 批量后处理: 一次处理 step 帧 mask → iSTFT → step 帧 PCM 输出
 * @param handle  句柄
 * @param params  pcm_out_batch 指向 step*hop_len 个连续 float 输出缓冲区
 */
AX_VOID AX_AI_SE_PostProcess(AX_VOID *handle, AX_AI_SE_PostParams *params);

AX_VOID AX_AI_SE_Free(AX_VOID *handle);

#ifdef __cplusplus
}
#endif

#endif /* AX_AI_SE_DENOISE_H */
