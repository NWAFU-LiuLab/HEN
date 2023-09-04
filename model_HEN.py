import timm
import torch.nn as nn
import torch
import torch.nn.functional as F

class HEN(nn.Module):
    def __init__(self, out_channels):
        super(HEN, self).__init__()

        # Load pretrained segmentation model
        pretrained_model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet50', pretrained=True)

        # Use only the backbone
        self.resnet = pretrained_model.backbone

        # Initialize a pretrained ViT model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, img_size = 256)
        self.vit.head = nn.Identity()  # Remove the classification head

        # Custom decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048 + 768, 512, 3, padding=1),  # Change the input channel size to match the output of the merged encoder
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        # Get ResNet features
        x_resnet = self.resnet(x)['out']  # 'out' is the output tensor of the backbone
        # Get ViT features
        B, C, H, W = x_resnet.shape
        x_vit = self.vit(x)  # The output size of ViT is (B, 768)
        x_vit = F.interpolate(x_vit.unsqueeze(-1).unsqueeze(-1), size=(H, W), mode='bilinear', align_corners=True)
        x_vit = x_vit.view(B, 768, H, W)  # Reshape to fit the input size of decoder (B, 768, H, W)
        # Merge the output of ResNet50 and ViT
        x = torch.cat([x_resnet, x_vit], dim=1)

        x = self.decoder(x)
        return x
