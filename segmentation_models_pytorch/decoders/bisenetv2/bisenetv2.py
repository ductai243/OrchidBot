import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseConvBN(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, padding=1):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class StemBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(out_ch, out_ch // 2, kernel_size=1),
            ConvBNReLU(out_ch // 2, out_ch, kernel_size=3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        return self.fuse(torch.cat([left, right], dim=1))


class GatherExpansion(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.stride = stride

        if stride == 1:
            self.branch = nn.Sequential(
                ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
                DepthwiseConvBN(in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
                ConvBN(mid_ch, out_ch, kernel_size=1),
            )
            self.shortcut = nn.Identity()
        elif stride == 2:
            self.branch = nn.Sequential(
                ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
                DepthwiseConvBN(in_ch, mid_ch, kernel_size=3, stride=2, padding=1),
                DepthwiseConvBN(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1),
                ConvBN(mid_ch, out_ch, kernel_size=1),
            )
            self.shortcut = nn.Sequential(
                DepthwiseConvBN(in_ch, in_ch, kernel_size=3, stride=2, padding=1),
                ConvBN(in_ch, out_ch, kernel_size=1),
            )
        else:
            raise ValueError("stride must be 1 or 2")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch(x) + self.shortcut(x))


class ContextEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv_gap = ConvBNReLU(in_ch, out_ch, kernel_size=1)
        self.conv_out = ConvBNReLU(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        context = self.gap(x)
        context = self.bn(context)
        context = self.conv_gap(context)
        return self.conv_out(x + context)


class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return self.stage3(x)


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1_2 = StemBlock(3, 16)
        self.stage3 = nn.Sequential(
            GatherExpansion(16, 32, stride=2),
            GatherExpansion(32, 32, stride=1),
        )
        self.stage4 = nn.Sequential(
            GatherExpansion(32, 64, stride=2),
            GatherExpansion(64, 64, stride=1),
        )
        self.stage5 = nn.Sequential(
            GatherExpansion(64, 128, stride=2),
            GatherExpansion(128, 128, stride=1),
            GatherExpansion(128, 128, stride=1),
            GatherExpansion(128, 128, stride=1),
            ContextEmbedding(128, 128),
        )

    def forward(self, x):
        x = self.stage1_2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.stage5(x)


class BilateralGuidedAggregation(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.detail_left = nn.Sequential(
            DepthwiseConvBN(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.detail_right = nn.Sequential(
            ConvBN(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.semantic_left = nn.Sequential(
            DepthwiseConvBN(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.semantic_right = ConvBN(channels, channels, kernel_size=3, stride=1, padding=1)
        self.fuse = ConvBNReLU(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, detail, semantic):
        detail_size = detail.shape[2:]

        detail_left = self.detail_left(detail)
        detail_right = self.detail_right(detail)

        semantic_left = torch.sigmoid(self.semantic_left(semantic))
        semantic_right = self.semantic_right(semantic)
        semantic_right = F.interpolate(
            semantic_right,
            size=detail_size,
            mode="bilinear",
            align_corners=False,
        )
        semantic_right = torch.sigmoid(semantic_right)

        detail_out = detail_left * semantic_right
        semantic_out = detail_right * semantic_left
        semantic_out = F.interpolate(
            semantic_out,
            size=detail_size,
            mode="bilinear",
            align_corners=False,
        )

        return self.fuse(detail_out + semantic_out)


class SegmentationHead(nn.Module):
    def __init__(self, in_ch, mid_ch, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_ch, num_classes, kernel_size=1),
        )

    def forward(self, x, output_size):
        x = self.block(x)
        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class BiSeNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.aggregation = BilateralGuidedAggregation(128)
        self.seg_head = SegmentationHead(128, 1024, num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        detail = self.detail_branch(x)
        semantic = self.semantic_branch(x)
        fused = self.aggregation(detail, semantic)
        return self.seg_head(fused, input_size)


# === Initialize ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiSeNetV2(num_classes=3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
