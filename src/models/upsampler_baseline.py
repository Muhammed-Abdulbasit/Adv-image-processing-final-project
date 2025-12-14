import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineUpsampler(nn.Module):
    def __init__(self, in_ch=6, out_ch=3, base_ch=32):
        super().__init__()
        # in_ch = mix_hr(3) + upsampled clean_lr_pred(3)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 1),
        )

    def forward(self, clean_lr_pred, mix_hr):
        # clean_lr_pred: Bx3x256x256
        # mix_hr: Bx3xH xW (e.g. 512x512)

        clean_up = F.interpolate(
            clean_lr_pred, size=mix_hr.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([mix_hr, clean_up], dim=1)
        out = self.net(x)
        out = torch.sigmoid(out)
        return out
