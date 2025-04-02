import torch
import torch.nn as nn
import torch.nn.functional as F

class LightChannelAttention3d(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels//ratio, 1),
            nn.ReLU(),
            nn.Conv3d(channels//ratio, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(self.pool(x))

class LightSpatialAttention3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))

class EfficientFusion(nn.Module):
    def __init__(self, num_modalities=4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, num_modalities, 1, 1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.shared_conv = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        weights = self.softmax(self.weights.repeat(B,1,1,1,1))
        fused = (x * weights).sum(dim=1, keepdim=True)
        return self.shared_conv(fused)

class LightDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
        self.att = nn.Sequential(
            LightChannelAttention3d(out_ch),
            LightSpatialAttention3d()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.att(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            LightDoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = LightDoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)

class LDCM(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        features = [24, 48, 96, 192]  # 减少下采样次数
        
        # 初始融合和卷积
        self.inc = EfficientFusion(in_channels)
        self.conv1 = LightDoubleConv(8, features[0])
        
        # 下采样 (只做3次下采样)
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # 上采样 (匹配下采样次数)
        self.up1 = Up(features[3], features[2], features[1])
        self.up2 = Up(features[1], features[1], features[0])
        
        # 最终输出卷积 (确保输出尺寸正确)
        self.out_conv = nn.Conv3d(features[0], num_classes, 1)
        
        # 添加最后的插值层确保尺寸匹配
        self.final_upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        original_size = x.shape[2:]  # 保存原始空间尺寸
        
        x = self.inc(x)
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        
        # 最终输出
        x = self.out_conv(x)
        
        # 如果尺寸不匹配，进行插值
        if x.shape[2:] != original_size:
            x = self.final_upsample(x)
        
        return x

if __name__ == '__main__':
    x = torch.randn(1, 4, 160, 160, 128)
    net = LDCM(in_channels=4, num_classes=4)
    y = net(x)
    print("Params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)