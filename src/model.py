import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class UNetConfig(PretrainedConfig):
    model_type = "unet_face_keypoints"

    def __init__(
        self,
        in_ch=3,
        out_ch=14,
        ch_mul=32,
        **kwargs,
    ):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        super().__init__(**kwargs)

class UnetPreTrainedModel(PreTrainedModel):
    config_class = UNetConfig
    base_model_prefix = "unet"

    def _init_weights(self, module):
        pass

class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.1):
        super().__init__()
        self.se = SEBlock(out_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.dropout = nn.Dropout2d(p=dropout_prob)
        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.se(x)
        return x + res

class UNet(UnetPreTrainedModel):
    def __init__(self, config: UNetConfig):
        super().__init__(config)
        self.config = config
        ch_mul = config.ch_mul

        self.enc1 = UNetBlock(config.in_ch, ch_mul)
        self.enc2 = UNetBlock(ch_mul, ch_mul * 2)
        self.enc3 = UNetBlock(ch_mul * 2, ch_mul * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(ch_mul * 4, ch_mul * 8)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_mul * 8, ch_mul * 4, kernel_size=1)
        )
        self.dec3 = UNetBlock(ch_mul * 8, ch_mul * 4)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_mul * 4, ch_mul * 2, kernel_size=1)
        )
        self.dec2 = UNetBlock(ch_mul * 4, ch_mul * 2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_mul * 2, ch_mul, kernel_size=1)
        )
        self.dec1 = UNetBlock(ch_mul * 2, ch_mul)

        self.final = nn.Conv2d(ch_mul, config.out_ch, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.post_init()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return self.sigmoid(out)
