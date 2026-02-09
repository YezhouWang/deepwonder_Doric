import os
import torch
import numpy as np
from skimage import io

from DWonder.RMBG.network import Network_3D_Unet
from DWonder.WF2NoBG_FFD import wf2nobg_ffd

# -----------------------------
# USER SETTINGS
# -----------------------------

# Input TIFF (WITH background)
input_tiff = "datasets/train_RMBG1/Input/mov_w_bg.tiff"

# Output TIFF
output_tiff = "results/mov_w_bg_first128_RMBG.tif"

# Your trained model
model_path = "RMBG_pth/train_RMBG1_dn4_fm32_202602031938/E_02_Iter_4000.pth"

GPU_ID = "0"
if_use_GPU = True

# RMBG parameters (same as training)
RMBG_fmap = 32
RMBG_normalize_factor = 1

RMBG_img_w = 256
RMBG_img_h = 256
RMBG_img_s = 128

RMBG_gap_w = 224
RMBG_gap_h = 224
RMBG_gap_s = 96

# -----------------------------
# Load movie
# -----------------------------

print("Loading TIFF...")

movie = io.imread(input_tiff)

print("Original shape:", movie.shape)

# Take ONLY first 28 frames
movie_128 = movie[:128, :, :]

print("Using first 128 frames:", movie_128.shape)

# Convert datatype if needed
movie_128 = movie_128.astype(np.uint16)

# -----------------------------
# Load RMBG network
# -----------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

RMBG_net = Network_3D_Unet(
    in_channels=4,
    out_channels=4,
    f_maps=RMBG_fmap,
    final_sigmoid=True
)

if if_use_GPU and torch.cuda.is_available():
    RMBG_net.load_state_dict(torch.load(model_path))
    RMBG_net.cuda()
else:
    RMBG_net.load_state_dict(torch.load(model_path, map_location='cpu'))

RMBG_net.eval()

# -----------------------------
# Run Background Removal
# -----------------------------

print("Running RMBG inference...")

movie_RMBG = wf2nobg_ffd(
    RMBG_net,
    movie_128,
    if_use_GPU=if_use_GPU,
    RMBG_GPU=GPU_ID,
    RMBG_batch_size=1,
    RMBG_img_w=RMBG_img_w,
    RMBG_img_h=RMBG_img_h,
    RMBG_img_s=RMBG_img_s,
    RMBG_gap_w=RMBG_gap_w,
    RMBG_gap_h=RMBG_gap_h,
    RMBG_gap_s=RMBG_gap_s,
    RMBG_normalize_factor=RMBG_normalize_factor
)

movie_RMBG = movie_RMBG.astype(np.uint16)

# -----------------------------
# Save Result
# -----------------------------

os.makedirs("results", exist_ok=True)

io.imsave(output_tiff, movie_RMBG)

print("Done.")
print("Saved to:", output_tiff)
