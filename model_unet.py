import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, out_channels):
        super(UNetPlusPlus, self).__init__()

        self.encoder = nn.Sequential(
            DoubleConv(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = DoubleConv(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2)

        x3 = self.middle(x2)

        x4 = self.up1(x3)
        x5 = torch.cat([x1, x4], dim=1)
        x6 = self.decoder1(x5)

        x7 = self.up2(x6)
        x8 = self.decoder2(x7)

        return x8

