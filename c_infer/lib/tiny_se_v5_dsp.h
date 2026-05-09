/**
 * @file tiny_se_v5_dsp.h
 * @brief TinyConvSE_v4 前后处理 DSP 库
 *
 * 使用 kiss_fft 实现 STFT / iSTFT，与 rnnoise 共用同一套 FFT 库。
 *
 * v4 与 v3 的关键区别:
 *   v3: 3ch 输入 [log1p_mag, cos_phase, sin_phase] → 2ch CRM mask → tanh → 复数乘法
 *   v4: 1ch 输入 [log1p_mag] → 1ch IRM mask → sigmoid → 幅度掩蔽 + 原相位
 *
 * 模型参数 (TinyConvSE_v4):
 *   n_fft   = 512   (FFT 点数)
 *   hop_len = 256   (帧移)
 *   win_len = 512   (窗长)
 *   F       = 257   (n_fft/2 + 1)
 *   F_int   = 17    (模型内部压缩频率)
 *
 *   T_model=34, context=28, step=6 (与 v3 相同)
 *
 * 处理流程:
 *   PCM → float → 加窗 STFT → 特征提取 [log1p_mag] (1ch)
 *   → 模型推理 → IRM mask (1ch) → sigmoid → 频率插值
 *   → 幅度掩蔽 (mag * IRM) + 原相位 → 增强频谱 → iSTFT → PCM
 */

#ifndef TINY_SE_V5_DSP_H
#define TINY_SE_V5_DSP_H

#include <stdint.h>
#include "rnnoise_src/kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 *  模型参数常量
 * ═══════════════════════════════════════════════════════════════ */

#define SE_V5_N_FFT        512
#define SE_V5_HOP_LEN      256
#define SE_V5_WIN_LEN      512
#define SE_V5_N_FREQS      (SE_V5_N_FFT / 2 + 1)   /* 257 */
#define SE_V5_F_INT        17
#define SE_V5_SAMPLE_RATE  16000

#define SE_V5_T_MODEL      34
#define SE_V5_RECEPTIVE_FIELD  29
#define SE_V5_CONTEXT      (SE_V5_RECEPTIVE_FIELD - 1)   /* 28 */
#define SE_V5_STEP          6

/* v4: 1ch 输入 (log1p_mag), 1ch IRM 输出 */
#define SE_V5_FEAT_CH      1
#define SE_V5_MASK_CH      1

/* ═══════════════════════════════════════════════════════════════
 *  DSP 状态 (跨帧连续)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    float analysis_buf[SE_V5_WIN_LEN - SE_V5_HOP_LEN];
    float synthesis_buf[SE_V5_WIN_LEN];
    float hann_window[SE_V5_WIN_LEN];

    kiss_fft_state *fft_cfg;
    kiss_fft_state *ifft_cfg;

    int first_frame;
    int istft_first_frame;
    int initialized;
} TinySeV5DspState;

/* ═══════════════════════════════════════════════════════════════
 *  API
 * ═══════════════════════════════════════════════════════════════ */

int  tiny_se_v5_dsp_init(TinySeV5DspState *st);
int  tiny_se_v5_dsp_init_sqrt_hann(TinySeV5DspState *st);
void tiny_se_v5_dsp_free(TinySeV5DspState *st);

void tiny_se_v5_stft_frame(TinySeV5DspState *st,
                            const float *pcm_in,
                            float *spec_real,
                            float *spec_imag);

void tiny_se_v5_istft_frame(TinySeV5DspState *st,
                             const float *spec_real,
                             const float *spec_imag,
                             float *pcm_out);

/** 特征提取: 仅 log1p(mag), 1 通道 */
void tiny_se_v5_extract_feat(const float *spec_real,
                              const float *spec_imag,
                              float *feat_out);

/** 频率插值: F_int → F_full, 1 通道 (f_int 硬编码为 SE_V5_F_INT=17) */
void tiny_se_v5_interp_mask(const float *mask_in, float *mask_out);

/** 频率插值: F_int → F_full, 1 通道 (f_int 运行时指定，支持多模型) */
void tiny_se_v5_interp_mask_n(const float *mask_in, int f_int,
                               float *mask_out, int f_out);

/** IRM 幅度掩蔽: enh = mag * irm, 保持原相位 */
void tiny_se_v5_apply_irm(float *spec_real,
                            float *spec_imag,
                            const float *mask_full);

/** sigmoid 激活 */
void tiny_se_v5_sigmoid_inplace(float *data, int n);

#ifdef __cplusplus
}
#endif

#endif /* TINY_SE_V5_DSP_H */
