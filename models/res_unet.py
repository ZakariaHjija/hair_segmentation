import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# Blocs de base
# ----------------------------------------

def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel//2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

def separable_conv_bn_relu(in_ch, out_ch, kernel=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=kernel//2, groups=in_ch), # depthwise
        nn.BatchNorm2d(in_ch),
        nn.ReLU(),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1), # pointwise
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel=3, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=kernel//2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=kernel//2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.add(out, x)  # Changed from out += x to use torch.add
        return out

# ----------------------------------------
# PrismaNet
# ----------------------------------------
class ResUnet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super().__init__()
        # Encoder
        self.conv0 = conv_bn_relu(in_channels, 16)
        self.res0 = ResidualBlock(16)

        self.conv1 = conv_bn_relu(16, 32, stride=2)
        self.res1 = ResidualBlock(32)

        self.conv2 = conv_bn_relu(32, 64, stride=2)
        self.res2 = ResidualBlock(64)

        self.conv3 = conv_bn_relu(64, 128, stride=2)
        self.res3 = ResidualBlock(128)

        self.conv4 = conv_bn_relu(128, 128, stride=2)
        self.res4 = ResidualBlock(128)

        # Decoder
        self.up1_conv = conv_bn_relu(128, 64)
        self.up1_res = ResidualBlock(64)

        self.up2_conv = conv_bn_relu(64, 32)
        self.up2_res = ResidualBlock(32)

        self.up3_conv = conv_bn_relu(32, 16)
        self.up3_res = ResidualBlock(16)

        self.up4_conv = conv_bn_relu(16, 8)
        self.up4_res = ResidualBlock(8)

        self.final_conv = nn.Conv2d(8, n_classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x0 = self.conv0(x)
        x0 = self.res0(x0)

        b1 = self.conv1(x0)
        b1 = self.res1(b1)

        b2 = self.conv2(b1)
        b2 = self.res2(b2)

        b3 = self.conv3(b2)
        b3 = self.res3(b3)

        b4 = self.conv4(b3)
        b4 = self.res4(b4)

        # Decoder
        up1 = F.interpolate(b4, scale_factor=2, mode='bilinear', align_corners=True)
        up1 = up1 + b3
        up1 = self.up1_conv(up1)
        up1 = self.up1_res(up1)

        up2 = F.interpolate(up1, scale_factor=2, mode='bilinear', align_corners=True)
        up2 = up2 + b2
        up2 = self.up2_conv(up2)
        up2 = self.up2_res(up2)

        up3 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True)
        up3 = up3 + b1
        up3 = self.up3_conv(up3)
        up3 = self.up3_res(up3)

        up4 = F.interpolate(up3, scale_factor=2, mode='bilinear', align_corners=True)
        up4 = up4 + x0
        up4 = self.up4_conv(up4)
        up4 = self.up4_res(up4)


        out = self.final_conv(up4)
        out = self.activation(out)
        return out

# ----------------------------------------
# Test
# ----------------------------------------
if __name__ == "__main__":
    model = ResUnet(in_channels=3, n_classes=1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)