/**
 * @file test_se_denoise_ax.c
 * @brief 统一语音增强测试程序 — AX650 NPU 板端版本
 *
 * 通过 INI 配置文件切换模型，无需重新编译:
 *   tiny_conv_v5:     models/tiny_v5_config.ini
 *   conv_gtcrn_small: models/conv_gtcrn_small_config.ini
 *
 * Usage:
 *   ./test_se_denoise_ax <input.wav> <output.wav> <config.ini> [model.axmodel]
 *
 * 示例:
 *   ./test_se_denoise_ax data/sample_1_noisy.wav output/out_tiny.wav \
 *       models/tiny_v5_config.ini
 *
 *   ./test_se_denoise_ax data/sample_1_noisy.wav output/out_gtcrn.wav \
 *       models/conv_gtcrn_small_config.ini
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ax_base_type.h"
#include "ax_ai_se_denoise.h"
#include "tiny_se_v5_dsp.h"

/* ═══════════════════════════════════════════════════════════════
 *  WAV 读写
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    AX_CHAR  chunk_id[4];
    AX_S32   chunk_size;
    AX_CHAR  format[4];
    AX_CHAR  subchunk1_id[4];
    AX_S32   subchunk1_size;
    AX_S16   audio_format;
    AX_S16   num_channels;
    AX_S32   sample_rate;
    AX_S32   byte_rate;
    AX_S16   block_align;
    AX_S16   bits_per_sample;
    AX_CHAR  subchunk2_id[4];
    AX_S32   subchunk2_size;
} WavHeader;

static AX_S32 read_wav_file(const AX_CHAR *filename, AX_F32 **data,
                              AX_S32 *num_samples, AX_S32 *sample_rate_out) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "[Error] Cannot open: %s\n", filename); return -1; }

    AX_CHAR riff[4], wave[4];
    AX_S32  file_size;
    if (fread(riff, 1, 4, fp) != 4 || fread(&file_size, 4, 1, fp) != 1 ||
        fread(wave, 1, 4, fp) != 4) { fclose(fp); return -1; }
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "[Error] Not a valid WAV file\n"); fclose(fp); return -1;
    }

    AX_S32  sample_rate = 0;
    AX_S16  audio_format = 0, num_channels = 0, bits_per_sample = 0;
    AX_S16 *pcm_i16 = NULL;
    AX_S32  n_samples = 0;

    while (1) {
        AX_CHAR chunk_id[4]; AX_S32 chunk_size;
        if (fread(chunk_id, 1, 4, fp) != 4) break;
        if (fread(&chunk_size, 4, 1, fp) != 1) break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            AX_S32 byte_rate; AX_S16 block_align;
            if (fread(&audio_format,    2, 1, fp) != 1) break;
            if (fread(&num_channels,    2, 1, fp) != 1) break;
            if (fread(&sample_rate,     4, 1, fp) != 1) break;
            if (fread(&byte_rate,       4, 1, fp) != 1) break;
            if (fread(&block_align,     2, 1, fp) != 1) break;
            if (fread(&bits_per_sample, 2, 1, fp) != 1) break;
            if (chunk_size > 16) fseek(fp, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            n_samples = chunk_size / (AX_S32)sizeof(AX_S16);
            pcm_i16 = (AX_S16*)malloc((size_t)n_samples * sizeof(AX_S16));
            if (!pcm_i16) { fclose(fp); return -1; }
            if ((AX_S32)fread(pcm_i16, sizeof(AX_S16), (size_t)n_samples, fp) != n_samples) {
                free(pcm_i16); fclose(fp); return -1;
            }
            break;
        } else { fseek(fp, chunk_size, SEEK_CUR); }
    }
    fclose(fp);

    if (audio_format != 1 || num_channels != 1 || bits_per_sample != 16) {
        fprintf(stderr, "[Error] Only 16-bit mono PCM WAV supported\n");
        free(pcm_i16); return -1;
    }

    *data = (AX_F32*)malloc((size_t)n_samples * sizeof(AX_F32));
    if (!*data) { free(pcm_i16); return -1; }
    for (AX_S32 i = 0; i < n_samples; i++)
        (*data)[i] = (AX_F32)pcm_i16[i] / 32768.0f;
    free(pcm_i16);

    *num_samples     = n_samples;
    *sample_rate_out = sample_rate;
    printf("[Info] Read: %s\n", filename);
    printf("       SampleRate=%d Hz  Samples=%d  Duration=%.2f s\n",
           sample_rate, n_samples, (AX_F32)n_samples / (AX_F32)sample_rate);
    return 0;
}

static AX_S32 write_wav_file(const AX_CHAR *filename, const AX_F32 *data,
                               AX_S32 num_samples, AX_S32 sample_rate) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "[Error] Cannot create: %s\n", filename); return -1; }

    AX_S16 *pcm_i16 = (AX_S16*)malloc((size_t)num_samples * sizeof(AX_S16));
    if (!pcm_i16) { fclose(fp); return -1; }
    for (AX_S32 i = 0; i < num_samples; i++) {
        AX_F32 v = data[i] * 32768.0f;
        if (v >  32767.0f) v =  32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm_i16[i] = (AX_S16)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    }

    WavHeader hdr = {
        .chunk_id       = {'R','I','F','F'},
        .chunk_size     = 36 + num_samples * (AX_S32)sizeof(AX_S16),
        .format         = {'W','A','V','E'},
        .subchunk1_id   = {'f','m','t',' '},
        .subchunk1_size = 16,
        .audio_format   = 1,
        .num_channels   = 1,
        .sample_rate    = sample_rate,
        .byte_rate      = sample_rate * (AX_S32)sizeof(AX_S16),
        .block_align    = sizeof(AX_S16),
        .bits_per_sample= 16,
        .subchunk2_id   = {'d','a','t','a'},
        .subchunk2_size = num_samples * (AX_S32)sizeof(AX_S16)
    };
    fwrite(&hdr,    sizeof(WavHeader), 1, fp);
    fwrite(pcm_i16, sizeof(AX_S16), (size_t)num_samples, fp);
    fclose(fp);
    free(pcm_i16);
    printf("[Info] Written: %s\n", filename);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  main
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr,
            "Usage: %s <input.wav> <output.wav> <config.ini> [model.axmodel]\n\n"
            "  切换模型只需修改 config.ini，无需重新编译:\n"
            "    tiny_conv_v5:      models/tiny_v5_config.ini\n"
            "    conv_gtcrn_small:  models/conv_gtcrn_small_config.ini\n",
            argv[0]);
        return 1;
    }

    const AX_CHAR *input_wav  = argv[1];
    const AX_CHAR *output_wav = argv[2];
    const AX_CHAR *ini_path   = argv[3];
    const AX_CHAR *model_arg  = (argc >= 5) ? argv[4] : AX_NULL;

    printf("========================================\n");
    printf(" Unified Speech Enhancement (AX NPU)\n");
    printf("========================================\n\n");

    AX_AI_SE_Config cfg;
    AX_CHAR model_from_ini[256] = {0};
    AX_AI_SE_LoadConfig(ini_path, &cfg, model_from_ini, (AX_S32)sizeof(model_from_ini));

    const AX_CHAR *model_path = (model_arg && model_arg[0]) ? model_arg : model_from_ini;
    if (!model_path || model_path[0] == '\0') {
        fprintf(stderr, "[Error] No model path specified\n"); return 1;
    }

    printf("[Config] ini=%s\n", ini_path);
    printf("[Config] model=%s\n", model_path);
    printf("[Config] hop_len=%d  t_model=%d  f_freqs=%d  f_int=%d  step=%d\n\n",
           cfg.hop_len, cfg.t_model, cfg.f_freqs, cfg.f_int, cfg.step);

    AX_VOID *handle = AX_NULL;
    if (AX_AI_SE_Init(&handle, &cfg, model_path) != 0) {
        fprintf(stderr, "[Error] AX_AI_SE_Init failed\n"); return -1;
    }

    AX_F32 *input_signal = AX_NULL;
    AX_S32  num_samples  = 0;
    AX_S32  sample_rate  = 0;
    if (read_wav_file(input_wav, &input_signal, &num_samples, &sample_rate) != 0) {
        AX_AI_SE_Free(handle); return -1;
    }

    const AX_S32 hop_len = cfg.hop_len;
    const AX_S32 step    = cfg.step;

    AX_S32 n_frames         = num_samples / hop_len;
    AX_S32 n_frames_aligned = (n_frames / step) * step;
    AX_S32 n_chunks         = n_frames_aligned / step;
    double audio_duration   = (double)num_samples / (double)sample_rate;

    printf("[Step 3] Starting inference (step=%d, chunks=%d)...\n", step, n_chunks);

    AX_F32 *output_signal = (AX_F32*)calloc((size_t)(n_frames_aligned * hop_len), sizeof(AX_F32));
    if (!output_signal) { free(input_signal); AX_AI_SE_Free(handle); return -1; }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    double total_infer_ms = 0.0;
    AX_S32 infer_count    = 0;

    for (AX_S32 chunk = 0; chunk < n_chunks; chunk++) {
        AX_S32 base = chunk * step * hop_len;

        AX_AI_SE_PreParams pre = {0};
        pre.pcm_in_batch = input_signal + base;
        AX_AI_SE_PreProcess(handle, &pre);

        struct timespec inf_s, inf_e;
        clock_gettime(CLOCK_MONOTONIC, &inf_s);

        AX_AI_SE_InferParams infer = {0};
        infer.model_input = pre.model_input;
        if (AX_AI_SE_Infer(handle, &infer) != 0) {
            fprintf(stderr, "[Error] Infer failed at chunk %d\n", chunk);
            free(output_signal); free(input_signal); AX_AI_SE_Free(handle); return -1;
        }

        clock_gettime(CLOCK_MONOTONIC, &inf_e);
        total_infer_ms += (inf_e.tv_sec - inf_s.tv_sec) * 1000.0 +
                           (inf_e.tv_nsec - inf_s.tv_nsec) / 1e6;
        infer_count++;

        AX_AI_SE_PostParams post = {0};
        post.model_output  = infer.model_output;
        post.pcm_out_batch = output_signal + base;
        AX_AI_SE_PostProcess(handle, &post);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                       (t_end.tv_nsec - t_start.tv_nsec) / 1e6;

    printf("         Inference done (%d calls)\n\n", infer_count);

    AX_S32 out_offset  = hop_len;
    AX_S32 out_samples = n_frames_aligned * hop_len - out_offset;
    if (out_samples > num_samples) out_samples = num_samples;

    if (write_wav_file(output_wav, output_signal + out_offset, out_samples, sample_rate) != 0) {
        free(output_signal); free(input_signal); AX_AI_SE_Free(handle); return -1;
    }

    double avg_infer_ms = infer_count > 0 ? total_infer_ms / infer_count : 0.0;
    double rtf          = total_ms / (audio_duration * 1000.0);

    printf("========================================\n");
    printf(" Performance Stats (AX NPU)\n");
    printf("========================================\n");
    printf("  Audio duration:   %.2f s\n",   audio_duration);
    printf("  Total frames:     %d\n",        n_frames);
    printf("  Aligned frames:   %d\n",        n_frames_aligned);
    printf("  Step:             %d\n",        step);
    printf("  Infer calls:      %d  (%.1f calls/s)\n",
           infer_count, (double)infer_count / audio_duration);
    printf("  Avg infer time:   %.3f ms\n",   avg_infer_ms);
    printf("  Total wall time:  %.1f ms\n",   total_ms);
    printf("  Realtime speedup: %.1f x\n",    audio_duration / (total_ms / 1000.0));
    printf("  RTF:              %.4f\n",       rtf);
    printf("========================================\n\n");

    free(output_signal);
    free(input_signal);
    AX_AI_SE_Free(handle);
    printf("[Info] Done.\n");
    return 0;
}
