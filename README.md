# Adv-image-processing-final-project

# Instructions for running the code

1. Clone the Repository
--------------------------------
git clone <your-repo-url>.git
cd <repo-folder>

2. Create and Activate Virtual Environment
--------------------------------
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

3. Install Dependencies
--------------------------------
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio
python3 -m pip install numpy pillow opencv-python tqdm lpips

4. Add Training Images
--------------------------------
Place any .jpg or .png images inside:

data/clean/

Example:
data/clean/img01.jpg
data/clean/img02.png

5. Train the Base Model
--------------------------------
mkdir -p checkpoints
python3 -m src.train_base

This creates:
checkpoints/base_unet.pth

6. Train the Upsamplers (Baseline + V-DESIRR)
--------------------------------
python3 -m src.train_upsampler

This creates:
checkpoints/upsampler_baseline.pth
checkpoints/upsampler_vdesirr.pth

7. Evaluate Models + Save Visual Results
--------------------------------
python3 -m src.eval_visuals

Outputs will be saved to:
results/sample_00.png
results/sample_01.png
...

Console will print PSNR and SSIM (LPIPS if available).

8. Deactivate Environment
--------------------------------
deactivate
