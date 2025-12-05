import torch
import torch.nn as nn
import torch.nn.functional as F

# Changes from U-Net:
# 1. Convolutions use same-padding, allowing concatenation of skip connections
# 2. Batch normalization is used to improve convergence speed
# 3. "up-convolution" is replaced with transposed convolution

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=17):
        super().__init__()

        self.enc1 = UNet._make_layer(in_channels, 64)
        self.enc2 = UNet._make_layer(64, 128)
        self.enc3 = UNet._make_layer(128, 256)
        self.enc4 = UNet._make_layer(256, 512)

        self.bottleneck = UNet._make_layer(512, 1024)

        self.up_conv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNet._make_layer(1024, 512)
        self.up_conv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNet._make_layer(512, 256)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNet._make_layer(256, 128)
        self.up_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNet._make_layer(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2))

        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2))

        d4 = self.dec4(torch.cat([self.up_conv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up_conv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up_conv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up_conv1(d2), e1], dim=1))

        # Output
        out = self.out_conv(d1)
        out = F.interpolate(out, size=(48, 64), mode="bilinear", align_corners=False)
        return out
