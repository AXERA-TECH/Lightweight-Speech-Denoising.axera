/**
 * @file tiny_se_v5_dsp.c
 * @brief TinyConvSE_v4 DSP 实现
 *
 * STFT/iSTFT 与 v3 完全相同 (相同的 FFT 参数)。
 * 区别在于:
 *   - extract_feat: 仅输出 log1p(mag) (1通道, v3=3通道)
 *   - interp_mask: 1通道 IRM (v3=2通道 CRM)
 *   - apply_irm: 幅度掩蔽+保相位 (v3=CRM复数乘法)
 *   - sigmoid_inplace: 替代 v3 的 tanh_inplace
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "tiny_se_v5_dsp.h"

/* ═══════════════════════════════════════════════════════════════
 *  静态辅助函数
 * ═══════════════════════════════════════════════════════════════ */

static void make_hann_window(float *win, int N) {
    for (int n = 0; n < N; n++) {
        win[n] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * n / N));
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  公共 API 实现
 * ═══════════════════════════════════════════════════════════════ */

int tiny_se_v5_dsp_init(TinySeV5DspState *st) {
    if (!st) return -1;

    memset(st, 0, sizeof(TinySeV5DspState));

    make_hann_window(st->hann_window, SE_V5_WIN_LEN);

    st->fft_cfg = rnn_fft_alloc(SE_V5_N_FFT, NULL, NULL, 0);
    if (!st->fft_cfg) {
        fprintf(stderr, "[tiny_se_v5_dsp] rnn_fft_alloc (forward) failed\n");
        return -1;
    }

    st->ifft_cfg = st->fft_cfg;

    st->first_frame = 1;
    st->istft_first_frame = 1;
    st->initialized = 1;
    return 0;
}

int tiny_se_v5_dsp_init_sqrt_hann(TinySeV5DspState *st) {
    int ret = tiny_se_v5_dsp_init(st);
    if (ret != 0) return ret;
    for (int i = 0; i < SE_V5_WIN_LEN; i++)
        st->hann_window[i] = sqrtf(st->hann_window[i]);
    return 0;
}

void tiny_se_v5_dsp_free(TinySeV5DspState *st) {
    if (!st) return;
    if (st->fft_cfg) {
        rnn_fft_free(st->fft_cfg, 0);
        st->fft_cfg  = NULL;
        st->ifft_cfg = NULL;
    }
    st->initialized = 0;
}

void tiny_se_v5_stft_frame(TinySeV5DspState *st,
                            const float *pcm_in,
                            float *spec_real,
                            float *spec_imag) {
    float frame[SE_V5_WIN_LEN];
    int overlap = SE_V5_WIN_LEN - SE_V5_HOP_LEN;

    if (st->first_frame) {
        for (int i = 0; i < overlap; i++) {
            int idx = overlap - i;
            if (idx >= overlap) idx = overlap - 1;
            frame[i] = pcm_in[idx];
        }
        st->first_frame = 0;
    } else {
        memcpy(frame, st->analysis_buf, overlap * sizeof(float));
    }
    memcpy(frame + overlap, pcm_in, SE_V5_HOP_LEN * sizeof(float));

    memcpy(st->analysis_buf, pcm_in, overlap * sizeof(float));

    for (int i = 0; i < SE_V5_WIN_LEN; i++)
        frame[i] *= st->hann_window[i];

    kiss_fft_cpx cx_in[SE_V5_N_FFT];
    kiss_fft_cpx cx_out[SE_V5_N_FFT];
    for (int i = 0; i < SE_V5_N_FFT; i++) {
        cx_in[i].r = frame[i];
        cx_in[i].i = 0.0f;
    }
    rnn_fft(st->fft_cfg, cx_in, cx_out, 0);

    for (int k = 0; k < SE_V5_N_FREQS; k++) {
        spec_real[k] = cx_out[k].r * SE_V5_N_FFT;
        spec_imag[k] = cx_out[k].i * SE_V5_N_FFT;
    }
}

void tiny_se_v5_istft_frame(TinySeV5DspState *st,
                             const float *spec_real,
                             const float *spec_imag,
                             float *pcm_out) {
    int overlap = SE_V5_WIN_LEN - SE_V5_HOP_LEN;

    kiss_fft_cpx cx_in[SE_V5_N_FFT];
    kiss_fft_cpx cx_out[SE_V5_N_FFT];

    cx_in[0].r = spec_real[0]; cx_in[0].i = spec_imag[0];
    for (int k = 1; k < SE_V5_N_FREQS - 1; k++) {
        cx_in[k].r = spec_real[k];
        cx_in[k].i = spec_imag[k];
        cx_in[SE_V5_N_FFT - k].r =  spec_real[k];
        cx_in[SE_V5_N_FFT - k].i = -spec_imag[k];
    }
    cx_in[SE_V5_N_FREQS - 1].r = spec_real[SE_V5_N_FREQS - 1];
    cx_in[SE_V5_N_FREQS - 1].i = spec_imag[SE_V5_N_FREQS - 1];

    rnn_ifft(st->ifft_cfg, cx_in, cx_out, 0);

    float frame[SE_V5_WIN_LEN];
    for (int i = 0; i < SE_V5_WIN_LEN; i++)
        frame[i] = (cx_out[i].r / SE_V5_N_FFT) * st->hann_window[i];

    for (int i = 0; i < SE_V5_WIN_LEN; i++)
        st->synthesis_buf[i] += frame[i];

    if (st->istft_first_frame) {
        memset(pcm_out, 0, SE_V5_HOP_LEN * sizeof(float));
        st->istft_first_frame = 0;
    } else {
        for (int i = 0; i < overlap; i++) {
            float w0 = st->hann_window[i];
            float w1 = st->hann_window[i + SE_V5_HOP_LEN];
            float env = w0 * w0 + w1 * w1;
            if (env < 1e-8f) env = 1e-8f;
            pcm_out[i] = st->synthesis_buf[i] / env;
        }
    }

    memmove(st->synthesis_buf,
            st->synthesis_buf + SE_V5_HOP_LEN,
            (SE_V5_WIN_LEN - SE_V5_HOP_LEN) * sizeof(float));
    memset(st->synthesis_buf + (SE_V5_WIN_LEN - SE_V5_HOP_LEN), 0,
           SE_V5_HOP_LEN * sizeof(float));
}

void tiny_se_v5_extract_feat(const float *spec_real,
                              const float *spec_imag,
                              float *feat_out) {
    /* v4: 仅输出 log1p(mag), 1 通道 */
    for (int k = 0; k < SE_V5_N_FREQS; k++) {
        float r = spec_real[k];
        float i = spec_imag[k];
        float mag = sqrtf(r * r + i * i + 1e-12f);
        feat_out[k] = log1pf(mag);
    }
}

void tiny_se_v5_interp_mask(const float *mask_in, float *mask_out) {
    tiny_se_v5_interp_mask_n(mask_in, SE_V5_F_INT, mask_out, SE_V5_N_FREQS);
}

void tiny_se_v5_interp_mask_n(const float *mask_in, int f_int,
                               float *mask_out, int f_out) {
    for (int k = 0; k < f_out; k++) {
        float x_out = (k + 0.5f) / (float)f_out;
        float pos   = x_out * (float)f_int - 0.5f;

        if (pos <= 0.0f) {
            mask_out[k] = mask_in[0];
        } else if (pos >= (float)(f_int - 1)) {
            mask_out[k] = mask_in[f_int - 1];
        } else {
            int   j    = (int)pos;
            float frac = pos - (float)j;
            mask_out[k] = mask_in[j] * (1.0f - frac) + mask_in[j + 1] * frac;
        }
    }
}

void tiny_se_v5_apply_irm(float *spec_real,
                            float *spec_imag,
                            const float *mask_full) {
    /*
     * IRM 幅度掩蔽:
     *   enh_mag = sqrt(r^2 + i^2) * mask
     *   保持原相位: enh_real = enh_mag * cos(phase), enh_imag = enh_mag * sin(phase)
     *   等价于: enh_real = r * mask, enh_imag = i * mask
     *   (因为 cos(phase) = r/mag, sin(phase) = i/mag, enh_mag * r/mag = r * mask)
     */
    for (int k = 0; k < SE_V5_N_FREQS; k++) {
        spec_real[k] *= mask_full[k];
        spec_imag[k] *= mask_full[k];
    }
}

void tiny_se_v5_sigmoid_inplace(float *data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}
