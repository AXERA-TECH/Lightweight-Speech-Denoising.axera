"""
TinyConvSE_v5 — 渐进升通道频率压缩版

v4 → v5 核心改动 (降低大频率阶段的内存峰值):

  v4 内存瓶颈分析 (T=34):
    freq_compress 各阶段输出特征:
      Stage 0: (24, 34, 129) → 24×34×129×4B = 400 KB  ← 峰值最大
      Stage 1: (24, 34,  65) → 24×34× 65×4B = 201 KB
      Stage 2: (24, 34,  33) → 24×34× 33×4B = 102 KB
      Stage 3: (24, 34,  17) → 24×34× 17×4B =  53 KB
    残差块主路径: 4×24×34×17×4B = 217 KB (4个激活同时存在)
    → 实际峰值由 Stage 0 + Stage 1 同时存在决定，远超 200 KB

  v5 解决方案 — 渐进升通道 (compress_ch → hidden_channels):
    前 (compress_stages-1) 个压缩阶段使用小通道数 compress_ch，
    最后一个压缩阶段升到 hidden_channels，之后残差块全程使用 hidden_channels。

    Stage 0: (compress_ch, 34, 129)  小通道，大频率
    Stage 1: (compress_ch, 34,  65)  小通道，中频率
    Stage 2: (compress_ch, 34,  33)  小通道，小频率
    Stage 3: (hidden_ch,   34,  17)  大通道，最小频率 ← 最后一级升通道

  默认参数:
    compress_ch:     12  (前3级压缩通道)
    hidden_channels: 28  (最后一级 + 残差块, v4=24)
    skip_channels:   14  (v4=12)

  内存估算 (T=34):
    Stage 0: 12×34×129×4B = 200 KB  (v4=400 KB, 降低 50%)
    Stage 1: 12×34× 65×4B = 101 KB  (v4=201 KB, 降低 50%)
    Stage 2: 12×34× 33×4B =  51 KB
    Stage 3: 28×34× 17×4B =  62 KB
    残差块主路径: 4×28×34×17×4B = 253 KB
    skip 累加器: 14×34×17×4B = 32 KB
    → 峰值: max(Stage0, 残差块) ≈ 200 KB vs 253 KB
    → 总 RAM (中间特征): ~285 KB (v4 ~244 KB, 但 Stage0 峰值从 400→200 KB)

  参数量估算:
    compress: 1→12 + 12→12 + 12→12 + 12→28 ≈ 4.5K
    blocks: 6× LightCausalBlockWithSkip(28, 14) ≈ 10K
    output head: ≈ 0.4K
    总计: ~15K (v4=12.9K, 略增)

架构:
  freq_compress:
    Stage 0~2: Conv2d(compress_ch→compress_ch, stride=1×2) × 3
    Stage 3:   Conv2d(compress_ch→hidden_ch,   stride=1×2) × 1
    F: 257→129→65→33→17
  6× LightCausalBlockWithSkip (C=hidden_ch, C_skip=skip_ch)
  PReLU(skip_ch) → Conv2d(skip_ch→1) → Sigmoid

ONNX 算子: Conv, PRelu, Add (全部量化兼容)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    """因果卷积: 时间维度只看过去帧, 频率维度对称 padding"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        time_pad = (kernel_size[0] - 1) * dilation[0]
        freq_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        self.pad = nn.ConstantPad2d((freq_pad, freq_pad, time_pad, 0), 0.0)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))

    def to_npu_friendly(self):
        """Expand dilated causal conv into an equivalent non-dilated larger kernel conv."""
        if self.dilation == (1, 1):
            return self

        eff_kernel = (
            (self.kernel_size[0] - 1) * self.dilation[0] + 1,
            (self.kernel_size[1] - 1) * self.dilation[1] + 1,
        )
        rewritten = CausalConv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=eff_kernel,
            stride=self.stride,
            dilation=1,
            groups=self.groups,
            bias=self.conv.bias is not None,
        )
        rewritten = rewritten.to(device=self.conv.weight.device, dtype=self.conv.weight.dtype)

        with torch.no_grad():
            rewritten.conv.weight.zero_()
            rewritten.conv.weight[
                :,
                :,
                ::self.dilation[0],
                ::self.dilation[1],
            ].copy_(self.conv.weight)
            if self.conv.bias is not None:
                rewritten.conv.bias.copy_(self.conv.bias)
        return rewritten


class LightCausalBlockWithSkip(nn.Module):
    """
    轻量因果残差块 + 跳跃连接 (TCN / ConvTasNet 风格)
    与 v3/v4 完全相同的结构。
    """
    def __init__(
        self,
        channels,
        skip_channels,
        kernel_size=(3, 3),
        dilation=(1, 1),
        learnable_out_act=True,
    ):
        super().__init__()
        self.depthwise = CausalConv2d(
            channels, channels, kernel_size,
            dilation=dilation, groups=channels, bias=False,
        )
        self.dw_bn  = nn.BatchNorm2d(channels)
        self.dw_act = nn.PReLU(channels)

        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.pw_bn  = nn.BatchNorm2d(channels)

        self.skip_conv = nn.Conv2d(channels, skip_channels, 1, bias=False)

        self.out_act = nn.PReLU(channels) if learnable_out_act else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dw_act(self.dw_bn(self.depthwise(x)))
        x = self.pw_bn(self.pointwise(x))
        skip = self.skip_conv(x)
        x = self.out_act(x + residual)
        return x, skip


class TinyConvSE_v5(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        compress_channels=12,   # 前几级压缩阶段使用的小通道数
        hidden_channels=28,     # 最后一级压缩 + 残差块通道数
        skip_channels=14,       # TCN skip 通道数
        num_blocks=6,
        kernel_size=(3, 3),
        dilations=(1, 2, 4, 1, 2, 4),
        compress_stages=4,
    ):
        super().__init__()
        self.n_fft    = n_fft
        self.hop_len  = hop_len
        self.win_len  = win_len
        self.n_freqs  = n_fft // 2 + 1
        self.compress_stages = compress_stages

        if len(dilations) < num_blocks:
            raise ValueError('len(dilations) must be >= num_blocks')
        if max(dilations[:num_blocks]) > 4:
            raise ValueError('Max dilation must be <= 4 for quantization compatibility')

        f = self.n_freqs
        for _ in range(compress_stages):
            f = (f + 1) // 2
        self.n_freqs_internal = f   # 17 (n_fft=512, stages=4)

        self.receptive_field = 1 + sum(2 * d for d in dilations[:num_blocks])

        # ── 渐进升通道频率压缩 ──
        # 前 (compress_stages-1) 级: 小通道 compress_channels
        # 最后 1 级: 升到 hidden_channels
        compress_layers = []
        in_ch = 1
        for s in range(compress_stages):
            # 最后一级升到 hidden_channels，其余用 compress_channels
            out_ch = hidden_channels if s == compress_stages - 1 else compress_channels
            compress_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3),
                          stride=(1, 2), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(out_ch),
            ]
            in_ch = out_ch
        self.freq_compress = nn.Sequential(*compress_layers)

        # ── 因果残差块 + 跳跃连接 ──
        self.blocks = nn.ModuleList([
            LightCausalBlockWithSkip(
                hidden_channels,
                skip_channels,
                kernel_size=kernel_size,
                dilation=(dilations[i], 1),
                learnable_out_act=(i != num_blocks - 1),
            )
            for i in range(num_blocks)
        ])

        # ── 输出头: skip 累加 → PReLU → IRM (1通道, sigmoid) ──
        self.output_act  = nn.PReLU(skip_channels)
        self.mask_conv   = nn.Conv2d(skip_channels, 1, kernel_size=1)
        self.mask_act    = nn.Sigmoid()

        self._export_mode = False

    def set_export_mode(self, mode=True):
        """
        ONNX 导出模式:
          input  : (B, 1, T, n_freqs)
          output : (B, 1, T, n_freqs_internal)   — 无 Sigmoid

        推理时外部处理:
          mask_full = linear_interp(sigmoid(mask_int), size=n_freqs)
          enh_mag = spec_mag * mask_full → 保持原相位 → iSTFT
        """
        self._export_mode = mode

    def prepare_for_npu_export(self):
        """
        Rewrite dilated depthwise causal convs into equivalent non-dilated kernels.
        This avoids AX620L backend failures on depthwise dilated conv tiling.
        """
        for block in self.blocks:
            block.depthwise = block.depthwise.to_npu_friendly()

    def _net_forward(self, feat):
        """纯网络前向: feat(B,1,T,F) → mask(B,1,T,F_int)"""
        x = self.freq_compress(feat)

        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        x = self.output_act(skip_sum)
        x = self.mask_conv(x)
        if not self._export_mode:
            x = self.mask_act(x)
        return x   # (B, 1, T, F_int)

    def forward(self, x):
        """
        训练模式: x (B, L) → output (B, L)
        导出模式: x (B, 1, T, F) → mask (B, 1, T, F_int)
        """
        if self._export_mode:
            return self._net_forward(x)

        device = x.device
        n_samples = x.shape[1]
        stft_kwargs = {
            'n_fft':       self.n_fft,
            'hop_length':  self.hop_len,
            'win_length':  self.win_len,
            'window':      torch.hann_window(self.win_len, device=device),
            'onesided':    True,
        }

        spec = torch.stft(x, **stft_kwargs, return_complex=True)
        spec_real = spec.real.permute(0, 2, 1)   # (B, T, F)
        spec_imag = spec.imag.permute(0, 2, 1)
        spec_mag  = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-12)
        spec_phase = torch.atan2(spec_imag, spec_real + 1e-12)

        feat = torch.log1p(spec_mag).unsqueeze(1)  # (B, 1, T, F)

        mask_compressed = self._net_forward(feat)  # (B, 1, T, F_int)

        mask = F.interpolate(
            mask_compressed,
            size=(mask_compressed.shape[2], self.n_freqs),
            mode='bilinear',
            align_corners=False,
        )  # (B, 1, T, F)

        irm = mask[:, 0]  # (B, T, F)
        enh_mag = spec_mag * irm
        enh_real = enh_mag * torch.cos(spec_phase)
        enh_imag = enh_mag * torch.sin(spec_phase)

        spec_enh = torch.complex(enh_real, enh_imag).permute(0, 2, 1)
        output = torch.istft(spec_enh, **stft_kwargs)
        pad_len = n_samples - output.shape[1]
        if pad_len > 0:
            output = F.pad(output, (0, pad_len))
        return output[:, :n_samples]


if __name__ == '__main__':
    import io

    model = TinyConvSE_v5().eval()

    params = sum(p.numel() for p in model.parameters())
    C_compress = model.freq_compress[0].weight.shape[0]   # compress_channels
    C_hidden   = model.blocks[0].pointwise.weight.shape[0]     # hidden_channels
    C_skip     = model.blocks[0].skip_conv.weight.shape[0]     # skip_channels
    T = 34
    F_int = model.n_freqs_internal
    step = T - (model.receptive_field - 1)

    print("=" * 65)
    print("TinyConvSE_v5 规格 (渐进升通道)")
    print("=" * 65)
    print(f"  参数量:         {params:,} = {params*4/1024:.1f} KB")
    print(f"  感受野:         {model.receptive_field} 帧 = "
          f"{model.receptive_field * model.hop_len / 16000 * 1000:.0f} ms")
    print(f"  T_input:        {T} 帧 (RF={model.receptive_field} + step={step} - 1)")
    print(f"  F_internal:     {F_int}  (compress_stages={model.compress_stages})")
    print(f"  compress_ch={C_compress}, hidden_ch={C_hidden}, skip_ch={C_skip}")
    print(f"  1s 调用次数:    {16000 / (step * model.hop_len):.1f} 次")
    print()

    # 各压缩阶段特征尺寸
    print("  freq_compress 各阶段特征 (T=34):")
    f_sizes = [model.n_freqs]
    for _ in range(model.compress_stages):
        f_sizes.append((f_sizes[-1] + 1) // 2)

    for s in range(model.compress_stages):
        f_in  = f_sizes[s]
        f_out = f_sizes[s + 1]
        out_ch = C_hidden if s == model.compress_stages - 1 else C_compress
        mem = out_ch * T * f_out * 4 / 1024
        label = "(升通道)" if s == model.compress_stages - 1 else ""
        print(f"    Stage {s}: ({out_ch:2d}, {T}, {f_out:3d})  {mem:6.1f} KB  {label}")
    print()

    # 内存估算
    feat_main = 4 * C_hidden * T * F_int * 4 / 1024
    feat_skip = C_skip * T * F_int * 4 / 1024
    feat_s0   = C_compress * T * f_sizes[1] * 4 / 1024  # 最大的压缩阶段
    feat_input = 1 * T * model.n_freqs * 4 / 1024
    feat_total = feat_main + feat_skip

    print(f"  中间特征峰值 (主路径):  4×{C_hidden}×{T}×{F_int}×4B = {feat_main:.1f} KB")
    print(f"  skip 累加器:            {C_skip}×{T}×{F_int}×4B = {feat_skip:.1f} KB")
    print(f"  压缩Stage0峰值:         {C_compress}×{T}×{f_sizes[1]}×4B = {feat_s0:.1f} KB  (v4={C_hidden}×{T}×{f_sizes[1]}×4B={C_hidden*T*f_sizes[1]*4/1024:.0f} KB)")
    print(f"  模型输入:               1×{T}×{model.n_freqs}×4B = {feat_input:.1f} KB")
    print(f"  总 RAM (中间特征):      {feat_total:.1f} KB")
    print(f"  参数 (ROM):             {params*4/1024:.1f} KB")
    total = feat_total + feat_input + params * 4 / 1024
    print(f"  ROM + RAM:              {total:.1f} KB  {'✓ < 500KB' if total < 500 else '✗ 超限'}")
    print()

    # 前向测试
    x = torch.randn(1, 16000)
    with torch.no_grad():
        y = model(x)
    print(f"  前向: {tuple(x.shape)} → {tuple(y.shape)}")

    # 因果性验证
    a = torch.randn(1, 16000)
    x1 = torch.cat([a, torch.randn(1, 16000)], dim=1)
    x2 = torch.cat([a, torch.randn(1, 16000)], dim=1)
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    hop = model.hop_len
    diff = (y1[:, :16000 - hop * 2] - y2[:, :16000 - hop * 2]).abs().max()
    print(f"  因果性检验 (应≈0): {diff.item():.2e}")

    # ONNX 算子验证
    try:
        import onnx, onnxsim
        from collections import Counter
        model.set_export_mode(True)
        dummy = torch.randn(1, 1, T, model.n_freqs)
        buf = io.BytesIO()
        torch.onnx.export(model, dummy, buf, opset_version=18,
                          input_names=['input'], output_names=['output'])
        buf.seek(0)
        m = onnx.load(buf)
        m_sim, ok = onnxsim.simplify(m)
        ops = Counter(n.op_type for n in m_sim.graph.node)
        print(f"  ONNX 算子: {dict(ops)}")
        print(f"  onnxsim: {'OK' if ok else 'FAIL'}")
    except ImportError:
        print("  (onnx/onnxsim 未安装，跳过 ONNX 验证)")

    # 与 v4 对比
    print()
    print("  与 v4 对比:")
    print(f"    compress Stage0 峰值: {feat_s0:.0f} KB  vs  v4 {C_hidden*T*f_sizes[1]*4/1024:.0f} KB  (降低 {(1-feat_s0/(C_hidden*T*f_sizes[1]*4/1024))*100:.0f}%)")
    print(f"    残差块主路径:         {feat_main:.0f} KB  vs  v4 {4*24*T*F_int*4/1024:.0f} KB")
    print(f"    参数量:               {params} vs v4 12889")
