import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    """因果卷积: 时间维度仅依赖当前与历史帧."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        time_pad = (kernel_size[0] - 1) * dilation[0]
        freq_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        self.pad = nn.ConstantPad2d((freq_pad, freq_pad, time_pad, 0), 0.0)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))

    def to_npu_friendly(self):
        if self.dilation == (1, 1):
            return self

        eff_kernel = (
            (self.kernel_size[0] - 1) * self.dilation[0] + 1,
            (self.kernel_size[1] - 1) * self.dilation[1] + 1,
        )
        rewritten = CausalConv2d(
            self.in_ch,
            self.out_ch,
            kernel_size=eff_kernel,
            stride=self.stride,
            dilation=1,
            groups=self.groups,
            bias=self.conv.bias is not None,
        )
        rewritten = rewritten.to(device=self.conv.weight.device, dtype=self.conv.weight.dtype)
        with torch.no_grad():
            rewritten.conv.weight.zero_()
            rewritten.conv.weight[:, :, ::self.dilation[0], ::self.dilation[1]].copy_(self.conv.weight)
            if self.conv.bias is not None:
                rewritten.conv.bias.copy_(self.conv.bias)
        return rewritten


class QuantCausalDSBlock(nn.Module):
    """
    量化友好的因果深度可分离残差块.
    仅使用 Conv/BN/ReLU/Add, 去掉 SE、Mul、PReLU 和多分支结构。
    """
    def __init__(self, channels, kernel_size=(3, 3), dilation=(1, 1)):
        super().__init__()
        self.dw = CausalConv2d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(channels)
        self.dw_act = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_bn(self.pw(x))
        x = x + residual
        return self.out_act(x)


class QuantCausalInter1d(nn.Module):
    """时间方向因果 1D 深度可分离卷积."""
    def __init__(self, channels, kernel_t, dilation):
        super().__init__()
        self.channels = channels
        self.kernel_t = kernel_t
        self.dilation = dilation
        pad = (kernel_t - 1) * dilation
        self.pad = nn.ConstantPad1d((pad, 0), 0.0)
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_t,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm1d(channels)
        self.dw_act = nn.ReLU(inplace=True)
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm1d(channels)
        self.pw_act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.dw_act(self.dw_bn(self.dw(self.pad(x))))
        x = self.pw_bn(self.pw(x))
        x = x + residual
        return self.pw_act(x)

    def prepare_for_npu_export(self):
        if self.dilation == 1:
            return

        eff_kernel = (self.kernel_t - 1) * self.dilation + 1
        rewritten_pad = nn.ConstantPad1d((eff_kernel - 1, 0), 0.0)
        rewritten_dw = nn.Conv1d(
            self.channels,
            self.channels,
            eff_kernel,
            dilation=1,
            groups=self.channels,
            bias=False,
        ).to(device=self.dw.weight.device, dtype=self.dw.weight.dtype)
        with torch.no_grad():
            rewritten_dw.weight.zero_()
            rewritten_dw.weight[:, :, ::self.dilation].copy_(self.dw.weight)
        self.pad = rewritten_pad
        self.dw = rewritten_dw
        self.dilation = 1


class QuantIntraFreq1d(nn.Module):
    """频率方向非因果 1D 深度可分离卷积."""
    def __init__(self, channels, kernel_f):
        super().__init__()
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_f,
            padding=kernel_f // 2,
            groups=channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm1d(channels)
        self.dw_act = nn.ReLU(inplace=True)
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm1d(channels)
        self.pw_act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_bn(self.pw(x))
        x = x + residual
        return self.pw_act(x)


class QuantConvDualPath(nn.Module):
    """
    保留双路径思想，但只使用 Conv/BN/ReLU/Add。
    去掉注意力和更复杂的激活形式，尽量减小量化敏感性。
    """
    def __init__(self, channels, kernel_t=5, kernel_f=5, n_layers=3):
        super().__init__()
        self.intra_layers = nn.ModuleList([
            QuantIntraFreq1d(channels, kernel_f) for _ in range(n_layers)
        ])
        self.intra_norm = nn.BatchNorm2d(channels)
        self.intra_act = nn.ReLU(inplace=True)

        self.inter_layers = nn.ModuleList([
            QuantCausalInter1d(channels, kernel_t, dilation=2 ** i)
            for i in range(n_layers)
        ])
        self.inter_norm = nn.BatchNorm2d(channels)
        self.inter_act = nn.ReLU(inplace=True)

    def forward(self, x):
        bsz, ch, n_frames, n_freqs = x.shape

        xi = x.permute(0, 2, 1, 3).reshape(bsz * n_frames, ch, n_freqs)
        for layer in self.intra_layers:
            xi = layer(xi)
        xi = xi.reshape(bsz, n_frames, ch, n_freqs).permute(0, 2, 1, 3)
        x = self.intra_act(x + self.intra_norm(xi))

        xt = x.permute(0, 3, 1, 2).reshape(bsz * n_freqs, ch, n_frames)
        for layer in self.inter_layers:
            xt = layer(xt)
        xt = xt.reshape(bsz, n_freqs, ch, n_frames).permute(0, 2, 3, 1)
        x = self.inter_act(x + self.inter_norm(xt))

        return x


class QuantEncoder(nn.Module):
    def __init__(self, enc_channels, dilations):
        super().__init__()
        self.down_layers = nn.ModuleList()
        in_ch = 1
        for out_ch in enc_channels:
            self.down_layers.append(nn.Sequential(
                CausalConv2d(in_ch, out_ch, kernel_size=(1, 5), stride=(1, 2), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        self.ds_blocks = nn.ModuleList([
            QuantCausalDSBlock(enc_channels[-1], kernel_size=(3, 3), dilation=(d, 1))
            for d in dilations
        ])

    def forward(self, x):
        skip_feats = []
        freq_sizes = []
        for layer in self.down_layers:
            x = layer(x)
            skip_feats.append(x)
            freq_sizes.append(x.shape[-1])
        for block in self.ds_blocks:
            x = block(x)
        return x, skip_feats, freq_sizes


class QuantSkipFuse(nn.Module):
    """
    更量化友好的 skip 融合.
    使用 Concat + 1x1 Conv 替代 SE-gate + Add，避免 ReduceMean/Mul/Sigmoid。
    """
    def __init__(self, dec_ch, skip_ch=None):
        super().__init__()
        if skip_ch is None:
            skip_ch = dec_ch
        if skip_ch != dec_ch:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(skip_ch, dec_ch, 1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.skip_proj = nn.Identity()

        self.fuse = nn.Sequential(
            nn.Conv2d(dec_ch * 2, dec_ch, 1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, dec_feat, skip_feat):
        skip = self.skip_proj(skip_feat)
        return self.fuse(torch.cat([dec_feat, skip], dim=1))


class QuantDecoder(nn.Module):
    def __init__(self, enc_channels, dilations):
        super().__init__()
        self.ds_blocks = nn.ModuleList([
            QuantCausalDSBlock(enc_channels[-1], kernel_size=(3, 3), dilation=(d, 1))
            for d in reversed(dilations)
        ])

        self.up_convs = nn.ModuleList()
        self.skip_fuses = nn.ModuleList()

        ch_seq = list(reversed(enc_channels))
        for i in range(len(ch_seq) - 1):
            in_ch = ch_seq[i]
            out_ch = ch_seq[i + 1]
            skip_ch = ch_seq[i + 1]
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            self.skip_fuses.append(QuantSkipFuse(out_ch, skip_ch=skip_ch))

        self.out_conv = nn.Conv2d(ch_seq[-1], 1, 1, bias=False)

    def forward(self, x, skip_feats, freq_sizes):
        for block in self.ds_blocks:
            x = block(x)

        n_up = len(self.up_convs)
        n_skip = len(skip_feats)
        for i in range(n_up):
            target_idx = n_skip - 2 - i
            target_f = freq_sizes[target_idx]
            skip_feat = skip_feats[target_idx]

            # 用 nearest 替代 bilinear，减小上采样量化误差。
            x = F.interpolate(x, size=(x.shape[2], target_f), mode='nearest')
            x = self.up_convs[i](x)
            x = self.skip_fuses[i](x, skip_feat)

        return self.out_conv(x)


class ConvGTCRN_Small_New(nn.Module):
    """
    量化友好的 ConvGTCRN Small 变体.

    设计目标:
      1. 不包含 ReduceMean
      2. 避免 PReLU、SE、Mul、双线性上采样
      3. 尽量使用 Conv/BN/ReLU/Add/Concat/NearestResize 这类更稳的算子
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
        self.decoder = QuantDecoder(enc_channels, enc_dilations)
        self.mask_act = nn.Sigmoid()
        self._export_mode = False

    def set_export_mode(self, mode=True):
        self._export_mode = mode

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
