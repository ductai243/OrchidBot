import torch
import torch.nn as nn
import torch.nn.functional as F

# === Depthwise Separable Conv Block ===
class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# === Spatial Reduction Attention ===
class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.pool = nn.AvgPool2d(kernel_size=reduction, stride=reduction) if reduction > 1 else nn.Identity()
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, -1).permute(0, 2, 1)
        x_pool = self.pool(x)
        kv = self.kv(x_pool)
        k, v = torch.chunk(kv, 2, dim=1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)
        attn = torch.softmax(q @ k / (C ** 0.5), dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(out)

# === DualPathFusionFFN ===
class DualPathFusionFFN(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU()
        )
        self.axial_h = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.axial_v = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        x_local = self.local(x)
        x_axial = self.axial_h(x) + self.axial_v(x)
        x = x_local + x_axial
        x = self.fusion(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj(x)
        return identity + x

# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn1 = SpatialReductionAttention(channels)
        self.ffn1 = DualPathFusionFFN(channels)
        self.attn2 = SpatialReductionAttention(channels)
        self.ffn2 = DualPathFusionFFN(channels)

    def forward(self, x):
        x = self.attn1(x) + x
        x = self.ffn1(x)
        x = self.attn2(x) + x
        x = self.ffn2(x)
        return x

# === Patch Embedding and Merging ===
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.proj(x))

class OverlapPatchMerge(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.proj(x))

# === Context Path ===
class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1_embed = OverlapPatchEmbed(3, 32)
        self.stage2_merge = OverlapPatchMerge(32, 64)
        self.stage3_merge = OverlapPatchMerge(64, 128)
        self.stage4_merge = OverlapPatchMerge(128, 256)
        self.trans1 = TransformerBlock(32)
        self.trans2 = TransformerBlock(64)
        self.trans3 = TransformerBlock(128)
        self.trans4 = TransformerBlock(256)

    def forward(self, x):
        c1 = self.trans1(self.stage1_embed(x))
        c2 = self.trans2(self.stage2_merge(c1))
        c3 = self.trans3(self.stage3_merge(c2))
        c4 = self.trans4(self.stage4_merge(c3))
        return c1, c2, c3, c4

# === BLFF ===
class BLFF(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(in_ch1 + in_ch2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        x_merge = self.merge(x)
        return self.global_att(x_merge) * x_merge + self.local_att(x_merge) * x_merge

# === SE Block ===
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

# === DPSegNet ===
class SNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.context_path = ContextPath()

        self.hcb1 = DepthwiseSeparableConvBlock(3, 32)
        self.hcb2 = DepthwiseSeparableConvBlock(32, 64)
        self.hcb3 = DepthwiseSeparableConvBlock(128, 128)  # fix in_ch
        self.hcb4 = DepthwiseSeparableConvBlock(256, 256)  # fix in_ch

        self.up_c3 = nn.Conv2d(128, 128, 1)
        self.up_c4 = nn.Conv2d(256, 128, 1)

        self.align_c3 = nn.Conv2d(64, 128, 1)
        self.align_x3 = nn.Conv2d(128, 256, 1)  # align x3_down lên 256
        self.align_c4 = nn.Conv2d(128, 256, 1)

        self.reduce_c1 = nn.Conv2d(32, 256, 1)
        self.reduce_c2 = nn.Conv2d(64, 256, 1)
        self.reduce_c3 = nn.Conv2d(128, 256, 1)
        self.reduce_c4 = nn.Conv2d(256, 256, 1)

        self.se = SEBlock(256 * 4)
        self.reduce_cp = nn.Conv2d(256 * 4, 512, 1)
        self.up_sp = nn.Conv2d(256, 512, 1)

        self.fuse3 = BLFF(128, 128, 128)
        self.fuse4 = BLFF(256, 256, 256)

        self.final_blff = BLFF(512, 512, 256)
        self.final_out = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x1 = self.hcb1(x)
        x1_down = F.avg_pool2d(x1, 2)
        x2 = self.hcb2(x1_down)
        x2_down = F.avg_pool2d(x2, 2)

        c1, c2, c3, c4 = self.context_path(x)

        c3_up = F.interpolate(self.up_c3(c3), size=x2_down.shape[2:], mode='bilinear', align_corners=False)
        x2_down_aligned = self.align_c3(x2_down)
        f3 = self.fuse3(x2_down_aligned, c3_up)
        x3 = self.hcb3(f3)
        x3_down = F.avg_pool2d(x3, 2)
        x3_down = self.align_x3(x3_down)  # align lên 256 trước fuse4

        c4_up = F.interpolate(self.up_c4(c4), size=x3_down.shape[2:], mode='bilinear', align_corners=False)
        c4_up = self.align_c4(c4_up)
        f4 = self.fuse4(x3_down, c4_up)
        x4 = self.hcb4(f4)

        fusion_size = (x.shape[2] // 4, x.shape[3] // 4)

        c1_up = F.interpolate(self.reduce_c1(c1), size=fusion_size, mode='bilinear', align_corners=False)
        c2_up = F.interpolate(self.reduce_c2(c2), size=fusion_size, mode='bilinear', align_corners=False)
        c3_up = F.interpolate(self.reduce_c3(c3), size=fusion_size, mode='bilinear', align_corners=False)
        c4_up = F.interpolate(self.reduce_c4(c4), size=fusion_size, mode='bilinear', align_corners=False)

        y_sp = F.interpolate(x4, size=fusion_size, mode='bilinear', align_corners=False)
        y_sp = self.up_sp(y_sp)

        cp_cat = torch.cat([c1_up, c2_up, c3_up, c4_up], dim=1)
        cp_cat = self.se(cp_cat)
        cp_out = self.reduce_cp(cp_cat)

        fused = self.final_blff(y_sp, cp_out)
        out = self.final_out(fused)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
