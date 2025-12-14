import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallRefineBlock(nn.Module):
    def __init__(self, in_ch, out_ch=3, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class VDesirrUpsampler(nn.Module):
    """
    Simple 2-stage multi-scale upsampler inspired by V-DESIRR:
    - Stage 1 (256x256): refine base prediction using low-res mixture.
    - Stage 2 (512x512): upsample refined prediction and refine again using full-res mixture.
    """
    def __init__(self, base_ch=64):
        super().__init__()
        # Stage 1: input = mix_lr (3) + clean_lr_pred (3)
        self.stage1 = SmallRefineBlock(in_ch=6, out_ch=3, base_ch=base_ch)

        # Stage 2: input = mix_hr (3) + upsampled refined (3)
        self.stage2 = SmallRefineBlock(in_ch=6, out_ch=3, base_ch=base_ch)

    def forward(self, clean_lr_pred, mix_hr):
        # Stage 1: operate at 256x256
        B, C, H_hr, W_hr = mix_hr.shape
        mix_lr = F.interpolate(
            mix_hr, size=clean_lr_pred.shape[-2:], mode="bilinear", align_corners=False
        )
        x1 = torch.cat([mix_lr, clean_lr_pred], dim=1)
        refined_lr = self.stage1(x1)  # Bx3x256x256

        # Stage 2: upsample refined_lr to HR and refine with mix_hr
        refined_up = F.interpolate(
            refined_lr, size=(H_hr, W_hr), mode="bilinear", align_corners=False
        )
        x2 = torch.cat([mix_hr, refined_up], dim=1)
        refined_hr = self.stage2(x2)  # Bx3xH_hrxW_hr

        return refined_hr
