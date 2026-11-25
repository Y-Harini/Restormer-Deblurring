
import os
import cv2
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from basicsr_film.archs.restormer_film_arch import RestormerFiLM


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = "/data/ai23btech11029/Restormer"

CONFIG_PATH = os.path.join(
    PROJECT_DIR,
    "Motion_Deblurring/Options/Deblurring_Restormer_D_FiLM_10.yml"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "experiments/Deblurring_Restormer_D_FiLM_10/models/net_g_latest.pth"
)

INPUT_DIR = "/data/DATASETS/GoPro/test/GoPro/input"

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "experiments/Deblurring_Restormer_D_FiLM_10/visualization/restored/GoPro"
)

WINDOW_SIZE = 8


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Restormer Model D FiLM - GoPro Testing")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(device))


# ============================================================
# Load YAML configuration
# ============================================================

with open(CONFIG_PATH, "r") as f:
    opt = yaml.safe_load(f)

network_opt = opt["network_g"].copy()

# Remove architecture type before passing arguments
network_opt.pop("type")


# ============================================================
# Initialize Model D
# ============================================================

print("\nInitializing RestormerFiLM...")

model = RestormerFiLM(**network_opt).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

if "params" not in checkpoint:
    raise KeyError("Checkpoint does not contain 'params'.")

model.load_state_dict(checkpoint["params"], strict=True)

model.eval()

print("Model D checkpoint loaded successfully.")


# ============================================================
# Prepare output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Image loading
# ============================================================

valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")


def load_image(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Unable to read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1).unsqueeze(0)

    return img


# ============================================================
# Reflection padding
# ============================================================

def pad_image(img, window_size):

    _, _, h, w = img.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h == 0 and pad_w == 0:
        return img, 0, 0

    if pad_h >= h or pad_w >= w:
        raise ValueError(
            f"Image dimensions {(h, w)} are too small for "
            f"reflection padding {(pad_h, pad_w)}."
        )

    img = F.pad(
        img,
        (0, pad_w, 0, pad_h),
        mode="reflect"
    )

    return img, pad_h, pad_w


# ============================================================
# Find test images
# ============================================================

image_paths = []

for root, _, files in os.walk(INPUT_DIR):

    for filename in sorted(files):

        if filename.lower().endswith(valid_extensions):

            image_paths.append(
                os.path.join(root, filename)
            )

image_paths.sort()

if not image_paths:
    raise RuntimeError("No test images found.")

print("\nNumber of test images:", len(image_paths))


# ============================================================
# Neutral FiLM conditioning
# ============================================================

# Validation and testing receive no synthetic augmentation metadata.
# Therefore, use an 8-dimensional zero vector.

aug_params = torch.zeros(
    (1, 8),
    dtype=torch.float32,
    device=device
)


# ============================================================
# Run inference
# ============================================================

print("\nStarting Model D inference...")

processed = 0

with torch.no_grad():

    for image_path in tqdm(image_paths):

        # Preserve relative directory structure
        relative_path = os.path.relpath(
            image_path,
            INPUT_DIR
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            relative_path
        )

        if os.path.exists(output_path):
            continue
        
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        # Load blurry input
        img = load_image(image_path).to(device)

        # Pad to window-size multiple
        img, pad_h, pad_w = pad_image(
            img,
            WINDOW_SIZE
        )

        # Model D forward pass
        restored = model(img, aug_params)

        # Remove padding
        if pad_h > 0:
            restored = restored[:, :, :-pad_h, :]

        if pad_w > 0:
            restored = restored[:, :, :, :-pad_w]

        # Convert output to image
        restored = restored.clamp(0, 1)

        restored = restored.squeeze(0)
        restored = restored.permute(1, 2, 0)

        restored = (
            restored.cpu().numpy() * 255.0
        ).round().astype(np.uint8)

        # RGB to BGR for OpenCV
        restored = cv2.cvtColor(
            restored,
            cv2.COLOR_RGB2BGR
        )

        # Save restored image
        success = cv2.imwrite(
            output_path,
            restored
        )

        if not success:
            raise IOError(
                f"Failed to save image: {output_path}"
            )

        processed += 1


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("Model D Inference Completed")
print("=" * 60)

print("Checkpoint:", CHECKPOINT_PATH)
print("Input images:", len(image_paths))
print("Successfully processed:", processed)
print("Restored images:", OUTPUT_DIR)

print("=" * 60)
