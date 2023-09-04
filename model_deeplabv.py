import torch
import torch.nn as nn
from torchsummary import summary

class DeepLav(nn.Module):
    def __init__(self, out_channels):
        super(DeepLav, self).__init__()

        # Load pretrained segmentation model(Vision Transformer & ResNet-50)
        pretrained_model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet50', pretrained=True)
        # print(pretrained_model)
        # Use only the backbone
        self.encoder = pretrained_model.backbone

        # self.layer_init = nn.Squential(
        #     pretrained_model.backbone.conv1,
        #     pretrained_model.backbone.bn1,
        #     pretrained_model.backbone.relu,
        #     pretrained_model.backbone.maxpool
        # )
        # self.layer1 = pretrained_model.backbone.layer1
        # self.layer2 = pretrained_model.backbone.layer2
        # self.layer3 = pretrained_model.backbone.layer3
        # self.layer4 = pretrained_model.backbone.layer4


        # Custom decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)['out']  # 'out' is the output tensor of the backbone
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    model = DeepLav(1)
    summary(model, input_size=(3, 256, 256), device='cpu')