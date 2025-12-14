import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SyntheticReflectionDataset
from src.models.unet_base import UNetBase
from src.models.upsampler_baseline import BaselineUpsampler
from src.models.upsampler_vdesirr import VDesirrUpsampler




def train_upsampler(
    upsampler_type="baseline",  # "baseline" or "vdesirr"
    clean_dir="data/clean",
    refl_dir=None,
    base_ckpt="checkpoints/base_unet.pth",
    epochs=15,
    batch_size=4,
    lr=1e-4,
    device="cuda",
    save_path_baseline="checkpoints/upsampler_baseline.pth",
    save_path_vdesirr="checkpoints/upsampler_vdesirr.pth",
):
    os.makedirs("checkpoints", exist_ok=True)

    dataset = SyntheticReflectionDataset(
        clean_dir=clean_dir,
        refl_dir=refl_dir,
        hr_size=(512, 512),
        lr_size=(256, 256),
        mode="upsampler",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load frozen base model
    base_model = UNetBase().to(device)
    base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # Choose upsampler
    if upsampler_type == "baseline":
        upsampler = BaselineUpsampler().to(device)
        save_path = save_path_baseline
    elif upsampler_type == "vdesirr":
        upsampler = VDesirrUpsampler().to(device)
        save_path = save_path_vdesirr
    else:
        raise ValueError("upsampler_type must be 'baseline' or 'vdesirr'")

    optimizer = torch.optim.Adam(upsampler.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        upsampler.train()
        epoch_loss = 0.0
        for mix_lr, clean_lr, mix_hr, clean_hr in tqdm(
            loader, desc=f"{upsampler_type} Epoch {epoch+1}/{epochs}"
        ):
            mix_lr = mix_lr.to(device)
            mix_hr = mix_hr.to(device)
            clean_hr = clean_hr.to(device)

            with torch.no_grad():
                clean_lr_pred = base_model(mix_lr)

            optimizer.zero_grad()
            clean_hr_pred = upsampler(clean_lr_pred, mix_hr)
            loss = criterion(clean_hr_pred, clean_hr)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * mix_lr.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"{upsampler_type} Epoch {epoch+1}: loss = {avg_loss:.4f}")

        torch.save(upsampler.state_dict(), save_path)
        print(f"Saved {upsampler_type} upsampler to {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (Optional) retrain baseline if you want, but we can skip:
    # train_upsampler(
    #     upsampler_type="baseline",
    #     device=device,
    #     epochs=20,
    # )

    # Retrain V-DESIRR-like upsampler with more epochs
    train_upsampler(
        upsampler_type="vdesirr",
        device=device,
        epochs=40,
    )
