import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, out_channels):
        super(DenseNet, self).__init__()

        # Load pretrained DenseNet model
        self.encoder = models.densenet121(weights=True)

        # Remove the classification head
        self.encoder = nn.Sequential(*list(self.encoder.features.children()))

        # Custom decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
