"""
MORSE: Morphological Observation with Reweighted Spectral Encoding
Memory-efficient implementation with configurable architecture depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


# ---------------------------
# 1. Morphological primitives
# ---------------------------

def make_disk_kernel(radius: int, device=None, dtype=torch.float32):
    """Binary disk structuring element (fixed, non-learnable)."""
    size = 2 * radius + 1
    y, x = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing='ij',
    )
    cy = cx = radius
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
    kernel = mask.to(dtype=dtype)
    return kernel


class MorphOp2d(nn.Module):
    """
    Differentiable *hard* morphology using max/min pooling with fixed SE.
    """
    def __init__(self, radius: int, op: str = "dilation"):
        super().__init__()
        assert op in ["dilation", "erosion"]
        self.radius = radius
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        r = self.radius
        pad = (r, r, r, r)
        x_p = F.pad(x, pad, mode="replicate")

        B, C, H, W = x.shape
        k = 2 * r + 1
        # Unfold to patches: (B, C * k * k, L) where L = H*W
        patches = F.unfold(x_p, kernel_size=k, stride=1)  # (B, C*k*k, H*W)
        patches = patches.view(B, C, k * k, H * W)        # (B, C, K, L)

        if self.op == "dilation":
            y, _ = patches.max(dim=2)                    # (B, C, L)
        else:
            y, _ = patches.min(dim=2)

        y = y.view(B, C, H, W)
        return y


class MorphProfileBlock(nn.Module):
    """
    Multi-scale morphological profile:
    For each radius r:
      - opening: erosion(r) then dilation(r)
      - closing: dilation(r) then erosion(r)
      - gradient: closing - opening
    Concatenate all scales + original.
    """
    def __init__(self, radii: List[int]):
        super().__init__()
        self.radii = radii
        self.erosions = nn.ModuleList([MorphOp2d(r, "erosion") for r in radii])
        self.dilations = nn.ModuleList([MorphOp2d(r, "dilation") for r in radii])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feats = [x]  # original intensity
        for erode, dilate in zip(self.erosions, self.dilations):
            er = erode(x)
            op = dilate(er)          # opening
            di = dilate(x)
            cl = erode(di)           # closing
            grad = cl - op           # morphological gradient

            feats.extend([op, cl, grad])

        return torch.cat(feats, dim=1)  # (B, C*(1+3*len(radii)), H, W)


# ---------------------------
# 2. Spectral / frequency branch
# ---------------------------

class HaarWaveletDecompose(nn.Module):
    """
    Non-trainable 2D Haar wavelet decomposition into LL, LH, HL, HH.
    """
    def __init__(self):
        super().__init__()
        # 1D Haar filters
        lp = torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2)])
        hp = torch.tensor([-1 / math.sqrt(2), 1 / math.sqrt(2)])

        # Build 2D kernels (LL, LH, HL, HH)
        ll = torch.outer(lp, lp)
        lh = torch.outer(lp, hp)
        hl = torch.outer(hp, lp)
        hh = torch.outer(hp, hp)

        k = torch.stack([ll, lh, hl, hh], dim=0)  # (4, 2, 2)
        k = k.unsqueeze(1)                        # (4, 1, 2, 2)

        self.register_buffer("kernel", k)         # fixed, no grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        k = self.kernel.to(x.device)             # (4, 1, 2, 2)
        k = k.repeat(C, 1, 1, 1)                 # (C*4, 1, 2, 2)

        # Apply depthwise convolution
        y = F.conv2d(x, k, stride=2, padding=0, groups=C)  # (B, C*4, H/2, W/2)

        return y


class SpectralFusionBlock(nn.Module):
    """
    Lightweight fusion of LL vs high-frequency subbands with gating.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels % 4 == 0
        base_c = in_channels // 4

        self.low_proj = nn.Conv2d(base_c, base_c, 1)
        self.high_proj = nn.Conv2d(3 * base_c, base_c, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * base_c, base_c, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, 2, 1),  # logits for [low, high]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,4C,H,W) -> split into LL, (LH,HL,HH)
        B, C, H, W = x.shape
        base_c = C // 4
        ll = x[:, :base_c]
        hf = x[:, base_c:]

        ll_p = self.low_proj(ll)
        hf_p = self.high_proj(hf)

        mix = torch.cat([ll_p, hf_p], dim=1)
        logits = self.gate(mix)             # (B,2,1,1)
        weights = F.softmax(logits, dim=1)  # (B,2,1,1)

        fused = weights[:, 0:1] * ll_p + weights[:, 1:2] * hf_p
        return fused  # (B, base_c, H, W)


class SpectralBranch(nn.Module):
    """
    Full spectral module with Haar decomposition and fusion.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.wavelet = HaarWaveletDecompose()
        self.fusion = SpectralFusionBlock(in_channels * 4)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        w = self.wavelet(x)          # (B,4C,H/2,W/2)
        fused = self.fusion(w)       # (B,C,H/2,W/2)
        out = self.conv(fused)       # (B,out_channels,H/2,W/2)
        return out


# ---------------------------
# 3. Memory-Efficient Cross-branch interaction
# ---------------------------

class EfficientCrossMorphSpectralAttention(nn.Module):
    """
    MEMORY-EFFICIENT cross-attention using channel-wise attention instead of spatial.
    Avoids O(N^2) memory by using O(C^2) where C << N.
    """
    def __init__(self, c_m: int, c_s: int, c_mid: int):
        super().__init__()
        # Channel attention instead of spatial attention
        self.q_m = nn.Conv2d(c_m, c_mid, 1)
        self.k_s = nn.Conv2d(c_s, c_mid, 1)
        self.v_s = nn.Conv2d(c_s, c_mid, 1)
        self.out_m = nn.Conv2d(c_mid, c_m, 1)

        # Spatial pooling to reduce resolution for attention
        self.pool = nn.AdaptiveAvgPool2d(7)  # Pool to 7x7 grid

    def forward(self, fm: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        # fm: (B,Cm,H,W), fs: (B,Cs,Hs,Ws)
        if fs.shape[-2:] != fm.shape[-2:]:
            fs = F.interpolate(fs, size=fm.shape[-2:], mode="bilinear", align_corners=False)

        B, Cm, H, W = fm.shape

        # Option 1: Channel-wise attention (most memory efficient)
        # Global average pooling for each channel
        q = self.q_m(fm)  # (B, Cmid, H, W)
        k = self.k_s(fs)  # (B, Cmid, H, W)
        v = self.v_s(fs)  # (B, Cmid, H, W)

        # Pool spatial dimensions
        q_pool = self.pool(q)  # (B, Cmid, 7, 7)
        k_pool = self.pool(k)
        v_pool = self.pool(v)

        # Flatten spatial dimensions
        q_flat = q_pool.view(B, q_pool.shape[1], -1).transpose(1, 2)  # (B, 49, Cmid)
        k_flat = k_pool.view(B, k_pool.shape[1], -1)                  # (B, Cmid, 49)
        v_flat = v_pool.view(B, v_pool.shape[1], -1).transpose(1, 2)  # (B, 49, Cmid)

        # Compute attention on reduced spatial resolution
        attn = torch.bmm(q_flat, k_flat) / math.sqrt(q_pool.shape[1])  # (B, 49, 49)
        attn = F.softmax(attn, dim=-1)
        out_pool = torch.bmm(attn, v_flat)  # (B, 49, Cmid)
        out_pool = out_pool.transpose(1, 2).view(B, -1, 7, 7)

        # Upsample back to original resolution
        out = F.interpolate(out_pool, size=(H, W), mode="bilinear", align_corners=False)
        out = self.out_m(out)

        return fm + out  # residual refinement


# ---------------------------
# 4. Backbone & classifier
# ---------------------------

class MorphSpectralBlock(nn.Module):
    """
    A block that fuses morphological and spectral features.
    """
    def __init__(self, in_morph_c: int, in_spec_c: int, out_c: int):
        super().__init__()
        self.cross_attn = EfficientCrossMorphSpectralAttention(
            c_m=in_morph_c, c_s=in_spec_c, c_mid=out_c // 2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_morph_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.proj = nn.Conv2d(in_morph_c, out_c, 1)

    def forward(self, fm: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        fm_ref = self.cross_attn(fm, fs)
        out = self.conv(fm_ref) + self.proj(fm_ref)
        return F.relu(out, inplace=True)


class MorphSpectralClassifier(nn.Module):
    """
    MORSE: Morphological Observation with Reweighted Spectral Encoding

    Memory-efficient architecture with configurable depth.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        morph_radii: Tuple[int, ...] = (1, 2, 4),
        base_width: int = 32,
        num_blocks: int = 2,  # Number of fusion blocks
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.base_width = base_width

        # Morphological profile
        self.morph_prof = MorphProfileBlock(list(morph_radii))
        self.morph_mult = (1 + 3 * len(morph_radii))
        morph_in = in_channels * self.morph_mult

        # First conv on morph stack
        self.morph_stem = nn.Sequential(
            nn.Conv2d(morph_in, base_width, 3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        # Spectral branch
        self.spec_branch = SpectralBranch(
            in_channels=in_channels,
            out_channels=base_width,
        )

        # Dynamic number of fusion blocks
        self.blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        current_channels = base_width
        for i in range(num_blocks):
            out_channels = current_channels * 2
            self.blocks.append(
                MorphSpectralBlock(
                    in_morph_c=current_channels,
                    in_spec_c=base_width,
                    out_c=out_channels,
                )
            )
            self.downsample.append(nn.MaxPool2d(2))
            current_channels = out_channels

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, current_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(current_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        # 1) Morphological profile
        m = self.morph_prof(x)   # (B, C*(1+3R), H, W)
        m = self.morph_stem(m)   # (B, base, H, W)

        # 2) Spectral branch
        s = self.spec_branch(x)  # (B, base, H/2, W/2)

        # 3) Hierarchical fusion through dynamic blocks
        h = m
        for i, (block, down) in enumerate(zip(self.blocks, self.downsample)):
            h = block(h, s)  # Fuse with spectral features
            if i < len(self.blocks) - 1:  # Don't downsample after last block
                h = down(h)

        # 4) Global classification head
        logits = self.head(h)
        return logits


# ---------------------------
# Memory profiling utilities
# ---------------------------

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def profile_model(model, input_tensor, device='cuda'):
    """Profile model memory and compute time"""
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Warm up
    with torch.no_grad():
        _ = model(input_tensor)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    mem_before = get_gpu_memory()

    # Forward pass
    import time
    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    torch.cuda.synchronize()
    forward_time = time.time() - start

    mem_after = get_gpu_memory()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        'params': count_parameters(model),
        'mem_allocated_mb': mem_after - mem_before,
        'peak_mem_mb': peak_mem,
        'forward_time_ms': forward_time * 1000,
        'output_shape': output.shape
    }


# ---------------------------
# Main testing
# ---------------------------
if __name__ == "__main__":
    print("="*80)
    print("MORSE Model - Memory Profiling")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cpu':
        print("WARNING: Running on CPU. GPU profiling not available.")

    # Test configurations
    configs = [
        {'base_width': 32, 'num_blocks': 2, 'batch_size': 32, 'desc': 'Default (small)'},
        {'base_width': 32, 'num_blocks': 3, 'batch_size': 32, 'desc': 'Medium depth'},
        {'base_width': 64, 'num_blocks': 2, 'batch_size': 16, 'desc': 'Wider channels'},
    ]

    for config in configs:
        print(f"\n{'-'*80}")
        print(f"Testing: {config['desc']}")
        print(f"Config: base_width={config['base_width']}, "
              f"num_blocks={config['num_blocks']}, "
              f"batch_size={config['batch_size']}")
        print(f"{'-'*80}")

        try:
            model = MorphSpectralClassifier(
                in_channels=3,
                num_classes=9,
                morph_radii=(1, 2, 4),
                base_width=config['base_width'],
                num_blocks=config['num_blocks'],
                dropout=0.3
            )

            x = torch.randn(config['batch_size'], 3, 224, 224)

            if device == 'cuda':
                stats = profile_model(model, x, device)

                print(f"\nParameters: {stats['params']:,}")
                print(f"Memory allocated: {stats['mem_allocated_mb']:.2f} MB")
                print(f"Peak memory: {stats['peak_mem_mb']:.2f} MB")
                print(f"Forward time: {stats['forward_time_ms']:.2f} ms")
                print(f"Output shape: {stats['output_shape']}")
                print(f"\n✓ Configuration successful!")
            else:
                # CPU testing
                params = count_parameters(model)
                output = model(x)
                print(f"\nParameters: {params:,}")
                print(f"Output shape: {output.shape}")
                print(f"\n✓ Configuration successful!")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue

    print(f"\n{'='*80}")
    print("Profiling complete!")
    print("="*80)