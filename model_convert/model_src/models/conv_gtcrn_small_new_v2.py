import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_gtcrn_small_new import (
    QuantCausalDSBlock,
    QuantConvDualPath,
    QuantEncoder,
    QuantSkipFuse,
)


class QuantDeconvUpBlock(nn.Module):
    """
    频率维上采样块.
    用 ConvTranspose2d 替代 Resize，避免导出图中的 Resize 算子。
    kernel=(1,3), stride=(1,2), padding=(0,1) 时:
      F_out = 2 * F_in - 1
    可精确匹配 33→65、65→129。
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
            output_padding=(0, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class QuantDeconvDecoder(nn.Module):
    """
    去 Resize 的量化友好 decoder.
    结构:
      DS blocks -> Deconv upsample -> Concat skip fuse -> refine block
    """
    def __init__(self, enc_channels, dilations):
        super().__init__()
        self.ds_blocks = nn.ModuleList([
            QuantCausalDSBlock(enc_channels[-1], kernel_size=(3, 3), dilation=(d, 1))
            for d in reversed(dilations)
        ])

        self.up_blocks = nn.ModuleList()
        self.skip_fuses = nn.ModuleList()
        self.refine_blocks = nn.ModuleList()

        ch_seq = list(reversed(enc_channels))
        for i in range(len(ch_seq) - 1):
            in_ch = ch_seq[i]
            out_ch = ch_seq[i + 1]
            skip_ch = ch_seq[i + 1]
            self.up_blocks.append(QuantDeconvUpBlock(in_ch, out_ch))
            self.skip_fuses.append(QuantSkipFuse(out_ch, skip_ch=skip_ch))
            self.refine_blocks.append(
                QuantCausalDSBlock(out_ch, kernel_size=(3, 3), dilation=(1, 1))
            )

        self.out_head = nn.Sequential(
            QuantCausalDSBlock(ch_seq[-1], kernel_size=(3, 3), dilation=(1, 1)),
            nn.Conv2d(ch_seq[-1], 1, 1, bias=False),
        )

    def forward(self, x, skip_feats, freq_sizes):
        for block in self.ds_blocks:
            x = block(x)

        n_up = len(self.up_blocks)
        n_skip = len(skip_feats)
        for i in range(n_up):
            target_idx = n_skip - 2 - i
            target_f = freq_sizes[target_idx]
            skip_feat = skip_feats[target_idx]

            x = self.up_blocks[i](x)
            if x.shape[-1] != target_f:
                raise RuntimeError(
                    f'Deconv output freq mismatch: got {x.shape[-1]}, expected {target_f}'
                )
            x = self.skip_fuses[i](x, skip_feat)
            x = self.refine_blocks[i](x)

        return self.out_head(x)


class ConvGTCRN_Small_New_V2(nn.Module):
    """
    conv_gtcrn_small_new 的增强版.

    目标:
      1. 保留量化友好的基础算子集合
      2. 去掉 decoder 中的两个 Resize
      3. 用可学习的反卷积上采样提升恢复能力
    """
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        enc_channels=(16, 32, 64),
        enc_dilations=(1, 2, 4, 1, 2, 4),
        num_dual_path=2,
        dp_n_layers=3,
        dp_kernel_t=5,
        dp_kernel_f=5,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_freqs = n_fft // 2 + 1

        freq_sizes = []
        freq = self.n_freqs
        for _ in enc_channels:
            freq = (freq + 1) // 2
            freq_sizes.append(freq)
        self.n_freqs_internal = freq_sizes[0]
        self._enc_freq_sizes = freq_sizes
        self.receptive_field = 1 + sum(2 * d for d in enc_dilations) * 2

        self.encoder = QuantEncoder(enc_channels, enc_dilations)
        self.dual_paths = nn.ModuleList([
            QuantConvDualPath(
                enc_channels[-1],
                kernel_t=dp_kernel_t,
                kernel_f=dp_kernel_f,
                n_layers=dp_n_layers,
            )
            for _ in range(num_dual_path)
        ])
        self.decoder = QuantDeconvDecoder(enc_channels, enc_dilations)
        self.mask_act = nn.Sigmoid()
        self._export_mode = False

    def set_export_mode(self, mode=True):
        self._export_mode = mode

    def prepare_for_npu_export(self):
        for block in self.encoder.ds_blocks:
            block.dw = block.dw.to_npu_friendly()
        for dp in self.dual_paths:
            for layer in dp.inter_layers:
                layer.prepare_for_npu_export()
        for block in self.decoder.ds_blocks:
            block.dw = block.dw.to_npu_friendly()
        for block in self.decoder.refine_blocks:
            block.dw = block.dw.to_npu_friendly()
        self.decoder.out_head[0].dw = self.decoder.out_head[0].dw.to_npu_friendly()

    def _net_forward(self, feat):
        x, skip_feats, freq_sizes = self.encoder(feat)
        for dp in self.dual_paths:
            x = dp(x)
        x = self.decoder(x, skip_feats, freq_sizes)
        if not self._export_mode:
            x = self.mask_act(x)
        return x

    def forward(self, x):
        if self._export_mode:
            return self._net_forward(x)

        device = x.device
        n_samples = x.shape[1]
        stft_kwargs = {
            'n_fft': self.n_fft,
            'hop_length': self.hop_len,
            'win_length': self.win_len,
            'window': torch.hann_window(self.win_len, device=device),
            'onesided': True,
        }

        spec = torch.stft(x, **stft_kwargs, return_complex=True)
        spec_real = spec.real.permute(0, 2, 1)
        spec_imag = spec.imag.permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-12)
        spec_phase = torch.atan2(spec_imag, spec_real + 1e-12)

        feat = torch.log1p(spec_mag).unsqueeze(1)
        mask_int = self._net_forward(feat)

        mask = F.interpolate(
            mask_int,
            size=(mask_int.shape[2], self.n_freqs),
            mode='nearest',
        )

        irm = mask[:, 0]
        enh_mag = spec_mag * irm
        enh_real = enh_mag * torch.cos(spec_phase)
        enh_imag = enh_mag * torch.sin(spec_phase)

        spec_enh = torch.complex(enh_real, enh_imag).permute(0, 2, 1)
        output = torch.istft(spec_enh, **stft_kwargs)
        pad_len = n_samples - output.shape[1]
        if pad_len > 0:
            output = F.pad(output, (0, pad_len))
        return output[:, :n_samples]
