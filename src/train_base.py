import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SyntheticReflectionDataset
from src.models.unet_base import UNetBase




def train_base(
    clean_dir="data/clean",
    refl_dir=None,
    epochs=20,
    batch_size=8,
    lr=1e-3,
    device="cuda",
    save_path="checkpoints/base_unet.pth",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = SyntheticReflectionDataset(
        clean_dir=clean_dir,
        refl_dir=refl_dir,
        hr_size=(512, 512),
        lr_size=(256, 256),
        mode="base",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = UNetBase().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for mix_lr, clean_lr in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            mix_lr = mix_lr.to(device)
            clean_lr = clean_lr.to(device)

            optimizer.zero_grad()
            clean_pred = model(mix_lr)

            loss = criterion(clean_pred, clean_lr)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * mix_lr.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_base(device=device)
