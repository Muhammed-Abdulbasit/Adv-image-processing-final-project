import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import SyntheticReflectionDataset
from src.models.unet_base import UNetBase
from src.models.upsampler_baseline import BaselineUpsampler
from src.models.upsampler_vdesirr import VDesirrUpsampler

# Try to import LPIPS, but don't crash if something goes wrong
try:
    import lpips  # for perceptual similarity (LPIPS)
    HAS_LPIPS = True
except ImportError:
    lpips = None
    HAS_LPIPS = False


def tensor_to_image(t):
    """
    Convert torch tensor (B or CxHxW, values in [0,1]) to uint8 HxWxC numpy array.
    """
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().clamp(0.0, 1.0).numpy()
    t = np.transpose(t, (1, 2, 0))  # HWC
    t = (t * 255.0).astype(np.uint8)
    return t


def psnr(pred, target, max_val=1.0):
    """
    pred, target: torch tensors in [0,1], shape (B,C,H,W)
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 99.0
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def simple_ssim(pred, target, max_val=1.0):
    """
    Very simple global SSIM approximation over the whole image.
    pred, target: (B,C,H,W) in [0,1]
    This is NOT a full windowed SSIM implementation, but fine for relative comparison.
    """
    x = pred.detach().cpu().view(-1)
    y = target.detach().cpu().view(-1)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_val.item()


def evaluate_and_save(
    clean_dir="data/clean",
    refl_dir=None,
    base_ckpt="checkpoints/base_unet.pth",
    baseline_ckpt="checkpoints/upsampler_baseline.pth",
    vdesirr_ckpt="checkpoints/upsampler_vdesirr.pth",
    out_dir="results",
    num_samples=10,
    device=None,
):
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    dataset = SyntheticReflectionDataset(
        clean_dir=clean_dir,
        refl_dir=refl_dir,
        hr_size=(512, 512),
        lr_size=(256, 256),
        mode="upsampler",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load models
    base_model = UNetBase().to(device)
    base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
    base_model.eval()

    upsampler_baseline = BaselineUpsampler().to(device)
    upsampler_baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    upsampler_baseline.eval()

    upsampler_vdesirr = VDesirrUpsampler().to(device)
    upsampler_vdesirr.load_state_dict(torch.load(vdesirr_ckpt, map_location=device))
    upsampler_vdesirr.eval()

    # Try to create LPIPS model (but don't die if SSL / download fails)
    lpips_alex = None
    USE_LPIPS = False
    if HAS_LPIPS:
        try:
            lpips_alex = lpips.LPIPS(net="alex").to(device)
            lpips_alex.eval()
            USE_LPIPS = True
            print("LPIPS loaded successfully (AlexNet backbone).")
        except Exception as e:
            print("Could not initialize LPIPS. Skipping LPIPS metric. Error:", e)
            lpips_alex = None
            USE_LPIPS = False
    else:
        print("lpips package not available. Skipping LPIPS metric.")

    psnr_baseline_list, psnr_vdesirr_list = [], []
    ssim_baseline_list, ssim_vdesirr_list = [], []
    lpips_baseline_list, lpips_vdesirr_list = [], []

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        mix_lr, clean_lr, mix_hr, clean_hr = batch
        mix_lr = mix_lr.to(device)
        mix_hr = mix_hr.to(device)
        clean_hr = clean_hr.to(device)

        with torch.no_grad():
            clean_lr_pred = base_model(mix_lr)
            clean_hr_baseline = upsampler_baseline(clean_lr_pred, mix_hr)
            clean_hr_vdesirr = upsampler_vdesirr(clean_lr_pred, mix_hr)

        # PSNR
        psnr_b = psnr(clean_hr_baseline, clean_hr).item()
        psnr_v = psnr(clean_hr_vdesirr, clean_hr).item()
        psnr_baseline_list.append(psnr_b)
        psnr_vdesirr_list.append(psnr_v)

        # SSIM
        ssim_b = simple_ssim(clean_hr_baseline, clean_hr)
        ssim_v = simple_ssim(clean_hr_vdesirr, clean_hr)
        ssim_baseline_list.append(ssim_b)
        ssim_vdesirr_list.append(ssim_v)

        # LPIPS (if available)
        if USE_LPIPS and lpips_alex is not None:
            def to_lpips_range(x):
                return x * 2.0 - 1.0

            b_lpips = (
                lpips_alex(
                    to_lpips_range(clean_hr_baseline),
                    to_lpips_range(clean_hr),
                )
                .mean()
                .item()
            )
            v_lpips = (
                lpips_alex(
                    to_lpips_range(clean_hr_vdesirr),
                    to_lpips_range(clean_hr),
                )
                .mean()
                .item()
            )
            lpips_baseline_list.append(b_lpips)
            lpips_vdesirr_list.append(v_lpips)
            lpips_msg = f"LPIPS baseline={b_lpips:.4f}, vdesirr={v_lpips:.4f}"
        else:
            lpips_msg = "LPIPS: skipped"

        # Save side-by-side comparison image
        img_mix = tensor_to_image(mix_hr)
        img_base = tensor_to_image(clean_hr_baseline)
        img_vdes = tensor_to_image(clean_hr_vdesirr)
        img_gt = tensor_to_image(clean_hr)

        h, w, _ = img_mix.shape
        canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)
        canvas[:, 0:w, :] = img_mix
        canvas[:, w:2*w, :] = img_base
        canvas[:, 2*w:3*w, :] = img_vdes
        canvas[:, 3*w:4*w, :] = img_gt

        Image.fromarray(canvas).save(os.path.join(out_dir, f"sample_{i:02d}.png"))

        print(
            f"[{i+1}/{num_samples}] Saved sample_{i:02d}.png "
            f"(PSNR baseline={psnr_b:.2f}, vdesirr={psnr_v:.2f}; "
            f"SSIM baseline={ssim_b:.4f}, vdesirr={ssim_v:.4f}; "
            f"{lpips_msg})"
        )

    def avg(x):
        return float(np.mean(x)) if len(x) > 0 else float("nan")

    print("\n=== Averages over", len(psnr_baseline_list), "samples ===")
    print(
        f"PSNR  - Baseline: {avg(psnr_baseline_list):.2f} dB, "
        f"V-DESIRR: {avg(psnr_vdesirr_list):.2f} dB"
    )
    print(
        f"SSIM  - Baseline: {avg(ssim_baseline_list):.4f}, "
        f"V-DESIRR: {avg(ssim_vdesirr_list):.4f}"
    )
    if USE_LPIPS and len(lpips_baseline_list) > 0:
        print(
            f"LPIPS - Baseline: {avg(lpips_baseline_list):.4f}, "
            f"V-DESIRR: {avg(lpips_vdesirr_list):.4f} (lower is better)"
        )
    else:
        print("LPIPS - skipped (model not available).")


if __name__ == "__main__":
    evaluate_and_save()
