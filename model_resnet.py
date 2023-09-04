import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, out_channels):
        super(ResNet, self).__init__()

        # Load pretrained ResNet model
        self.encoder = models.resnet50(weights=True)

        # Remove the classification head
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # Custom decoder
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(2048, 1024, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #     nn.Conv2d(1024, 512, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #     nn.Conv2d(512, 256, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #     nn.Conv2d(256, 64, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, out_channels, 1)
        # )

        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

