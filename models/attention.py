from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):  # for 64 channel
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, do_activation=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.do_activation = do_activation

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, q, k, v):

        """
        q shape: 1 x N'x C
        k shape: 1 x N'x C
        v shape: 1 x N'x C
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # activation here
        if self.do_activation:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        # activation here
        if self.do_activation:
            output = self.activation(output)

        output = self.layer_norm(output + residual)

        return output

    def query_self_cross(self, q, k, v, mask):
        """
                q shape: 1 x N'x C
                k shape: 1 x N'x C
                v shape: 1 x N'x C
                """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # activation here
        if self.do_activation:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn = self.attention.su_mask(q, k, v, mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        # activation here
        if self.do_activation:
            output = self.activation(output)

        output = self.layer_norm(output + residual)

        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron module
    """

    def __init__(self, dim, mlp_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = self.mlp(x) + x
        x = self.norm(x)
        return x


class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, spatial_features, frequency_features):
        spatial_att = self.spatial_attention(spatial_features)  # torch.Size([2, 1, 64, 64])
        frequency_att = self.channel_attention(frequency_features)  # torch.Size([2, 512, 1, 1])

        #
        fused_features = spatial_att * spatial_features + frequency_att * frequency_features
        return fused_features


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7.'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # torch.Size([2, 512, 64, 64])
        avg_out = torch.mean(x, dim=1, keepdim=True)  # torch.Size([2, 1, 64, 64])
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # torch.Size([2, 1, 64, 64])
        x = torch.cat([avg_out, max_out], dim=1)  # torch.Size([2, 2, 64, 64])
        x = self.conv1(x)  # torch.Size([2, 1, 64, 64])
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # torch.Size([2, 512, 64, 64])
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # torch.Size([2, 512, 1, 1])
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # torch.Size([2, 512, 1, 1])
        out = self.sigmoid(avg_out + max_out)  # torch.Size([2, 512, 1, 1])
        return out



class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Conv: DWConv (k×k, groups=C) + PWConv (1×1)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=padding, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return x


def _build_radial_masks(h: int, w: int, alpha: float, device: torch.device):
    """Build low/high-pass binary masks on the 2D frequency grid after fftshift."""
    # Frequency coordinates centered at (H/2, W/2)
    u = torch.arange(h, device=device).unsqueeze(1).expand(h, w)
    v = torch.arange(w, device=device).unsqueeze(0).expand(h, w)
    ru = (u - h / 2.0)
    rv = (v - w / 2.0)
    r = torch.sqrt(ru * ru + rv * rv)  # [H, W]

    r_max = math.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2)
    low_mask = (r <= alpha * r_max).to(torch.float32)          # pass low radius
    high_mask = (r >= (1.0 - alpha) * r_max).to(torch.float32) # pass high radius
    return low_mask, high_mask  # [H, W]


# ---------- Decomposition ----------

class SpatialFreqDecomposer(nn.Module):
    """

    """
    def __init__(self, pool_kernel: int = 4, pool_stride: int = 4, fft_alpha: float = 0.15):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride, ceil_mode=False)
        self.fft_alpha = fft_alpha

    def split_spatial(self, Fs: torch.Tensor):
        """Return Fs_low, Fs_high with same size as Fs."""
        B, C, H, W = Fs.shape
        Fs_low = self.pool(Fs)
        Fs_low_up = F.interpolate(Fs_low, size=(H, W), mode="bilinear", align_corners=False)
        Fs_high = Fs - Fs_low_up
        return Fs_low_up, Fs_high

    def split_frequency(self, Ff: torch.Tensor):
        """
        """
        B, C, H, W = Ff.shape
        device = Ff.device

        # 2D FFT per-channel
        # Move to complex after FFT; we center the spectrum using fftshift
        F = torch.fft.fft2(Ff, dim=(-2, -1))
        F_shift = torch.fft.fftshift(F, dim=(-2, -1))  # center zero-frequency

        # Build masks once per spatial size, then broadcast to (B,C,H,W)
        M_low, M_high = _build_radial_masks(H, W, self.fft_alpha, device)
        M_low = M_low.view(1, 1, H, W).expand(B, C, H, W)
        M_high = M_high.view(1, 1, H, W).expand(B, C, H, W)

        # Apply masks
        F_low_shift = F_shift * M_low
        F_high_shift = F_shift * M_high

        # Inverse shift and iFFT back to spatial domain
        F_low = torch.fft.ifftshift(F_low_shift, dim=(-2, -1))
        F_high = torch.fft.ifftshift(F_high_shift, dim=(-2, -1))

        f_low = torch.fft.ifft2(F_low, dim=(-2, -1)).real
        f_high = torch.fft.ifft2(F_high, dim=(-2, -1)).real
        return f_low, f_high


# ---------- High-frequency multi-scale enhancement ----------

class HighFreqMultiScaleEnhance(nn.Module):
    """
    Multi-kernel DWConv on high-frequency sum, then 1×1 fuse + GELU + residual.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = DepthwiseSeparableConv(channels, channels, kernel_size=1, padding=0)
        self.dw3 = DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1)
        self.dw5 = DepthwiseSeparableConv(channels, channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, Fs_high: torch.Tensor, Ff_high: torch.Tensor):
        # Aggregate high-frequency parts
        Fh = Fs_high + Ff_high

        # Depthwise separable convolutions at multiple scales
        f1 = self.dw1(Fh)
        f3 = self.dw3(Fh)
        f5 = self.dw5(Fh)

        # Concatenate and project
        cat = torch.cat([f1, f3, f5], dim=1)
        fused = self.fuse(cat)
        fused = self.bn(fused)
        out = self.act(fused) + Fh  # residual connection
        return out  # F_hf


# ---------- Low-frequency differential attention ----------

class LowFreqDifferentialAttention(nn.Module):
    """
    Differential self-attention on low-frequency sum:
    """
    def __init__(self, channels: int, attn_dim: int = 64, ffn_expansion: int = 4):
        super().__init__()
        self.q1 = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.k1 = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.q2 = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.k2 = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)

        self.lambda_param = nn.Parameter(torch.tensor(0.5))  # learnable λ
        self.proj_out = nn.Conv2d(attn_dim, channels, kernel_size=1, bias=False)

        hidden = ffn_expansion * channels
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, Fs_low: torch.Tensor, Ff_low: torch.Tensor):
        # Aggregate low-frequency parts
        Fl = Fs_low + Ff_low  # (B, C, H, W)
        B, C, H, W = Fl.shape
        N = H * W

        # Linear projections
        q1 = self.q1(Fl).flatten(2).transpose(1, 2)  # (B, N, d)
        k1 = self.k1(Fl).flatten(2)                  # (B, d, N)
        q2 = self.q2(Fl).flatten(2).transpose(1, 2)  # (B, N, d)
        k2 = self.k2(Fl).flatten(2)                  # (B, d, N)
        v  = self.v(Fl).flatten(2).transpose(1, 2)   # (B, N, d)

        d = q1.shape[-1]
        attn_main = torch.bmm(q1, k1) / math.sqrt(d)        # (B, N, N)
        attn_supp = torch.bmm(q2, k2) / math.sqrt(d)        # (B, N, N)
        A = F.softmax(attn_main - self.lambda_param * attn_supp, dim=-1)  # (B, N, N)

        # Apply attention to V and project back
        out = torch.bmm(A, v)                                # (B, N, d)
        out = out.transpose(1, 2).view(B, -1, H, W)          # (B, d, H, W)
        out = self.proj_out(out)                             # (B, C, H, W)
        out = self.ffn(out) + Fl                             # residual over aggregated low-freq
        return out  # F_lf


# ---------- Gated dual-frequency aggregation ----------

class GatedDualFrequencyAggregation(nn.Module):
    """
    Gate-controlled fusion of F_hf and F_lf:
        g = sigmoid(Conv1x1([F_hf, F_lf]))
        F_s^f = Conv3x3(g ⊙ F_hf + (1-g) ⊙ F_lf)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, F_hf: torch.Tensor, F_lf: torch.Tensor):
        cat = torch.cat([F_hf, F_lf], dim=1)
        g = self.gate(cat)                     # (B, C, H, W)
        fused = g * F_hf + (1.0 - g) * F_lf    # weighted fusion
        out = self.fuse(fused)
        out = self.bn(out)
        return out  # F_s^f


# ---------- DSFR (Spatial-frequency feature fusion) ----------

class DSFR(nn.Module):
    """
    Full DSFR module
    """
    def __init__(
        self,
        channels: int,
        pool_kernel: int = 4,
        pool_stride: int = 4,
        fft_alpha: float = 0.15,
        attn_dim: int = 64,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.decomposer = SpatialFreqDecomposer(pool_kernel=pool_kernel,
                                                pool_stride=pool_stride,
                                                fft_alpha=fft_alpha)
        self.hf_enhance = HighFreqMultiScaleEnhance(channels=channels)
        self.lf_attention = LowFreqDifferentialAttention(channels=channels,
                                                         attn_dim=attn_dim,
                                                         ffn_expansion=ffn_expansion)
        self.aggregator = GatedDualFrequencyAggregation(channels=channels)

    def forward(self, Fs: torch.Tensor, Ff: torch.Tensor):
        """
        Args:
            Fs: spatial features  (B, C, H, W)
            Ff: frequency features (B, C, H, W)
        Returns:
            F_sf: fused spatial representation F_s^f (B, C, H, W)
        """
        # 1) Frequency decomposition for Fs and Ff
        Fs_low, Fs_high = self.decomposer.split_spatial(Fs)   # (B, C, H, W) each
        Ff_low, Ff_high = self.decomposer.split_frequency(Ff) # (B, C, H, W) each

        # 2) High-frequency enhancement
        F_hf = self.hf_enhance(Fs_high, Ff_high)               # (B, C, H, W)

        # 3) Low-frequency differential attention
        F_lf = self.lf_attention(Fs_low, Ff_low)               # (B, C, H, W)

        # 4) Gated aggregation
        F_sf = self.aggregator(F_hf, F_lf)                     # (B, C, H, W)
        return F_sf



class AttentionFusion(nn.Module):

    def __init__(self, in_channels: int,
                 pool_kernel: int = 4,
                 pool_stride: int = 4,
                 fft_alpha: float = 0.15,
                 attn_dim: int = 64,
                 ffn_expansion: int = 4):
        super().__init__()
        self.dsfr = DSFR(channels=in_channels,
                         pool_kernel=pool_kernel,
                         pool_stride=pool_stride,
                         fft_alpha=fft_alpha,
                         attn_dim=attn_dim,
                         ffn_expansion=ffn_expansion)

    def forward(self, spatial_features: torch.Tensor, frequency_features: torch.Tensor):
        """
        Args:
            spatial_features: Fs (B, C, H, W)
            frequency_features: Ff (B, C, H, W)
        Returns:
            fused_features: F_s^f (B, C, H, W)
        """
        fused_features = self.dsfr(spatial_features, frequency_features)
        return fused_features
