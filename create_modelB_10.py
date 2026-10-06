from pathlib import Path
import random
import os

# Original dataset
INPUT_DIR = Path("/data/DATASETS/GoPro/train/GoPro/input_crops")
TARGET_DIR = Path("/data/DATASETS/GoPro/train/GoPro/target_crops")

# 10% dataset
OUTPUT_INPUT_DIR = Path(
    "/data/ai23btech11029/Restormer/datasets/modelB_GoPro_10/input_crops"
)

OUTPUT_TARGET_DIR = Path(
    "/data/ai23btech11029/Restormer/datasets/modelB_GoPro_10/target_crops"
)

# Reproducible random selection
SEED = 42
PERCENTAGE = 0.10

OUTPUT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Get all input images
input_files = sorted(INPUT_DIR.glob("*.png"))

print("Total input images:", len(input_files))

# Check input-target pairs
valid_pairs = []

for input_file in input_files:
    target_file = TARGET_DIR / input_file.name

    if target_file.exists():
        valid_pairs.append(input_file)
    else:
        print("WARNING: Missing target:", input_file.name)

print("Valid image pairs:", len(valid_pairs))

# Select 10%
random.seed(SEED)

num_samples = int(len(valid_pairs) * PERCENTAGE)

selected_files = random.sample(valid_pairs, num_samples)

print("Selected samples:", len(selected_files))

# Create symbolic links
for input_file in selected_files:

    target_file = TARGET_DIR / input_file.name

    output_input = OUTPUT_INPUT_DIR / input_file.name
    output_target = OUTPUT_TARGET_DIR / input_file.name

    if not output_input.exists():
        os.symlink(input_file, output_input)

    if not output_target.exists():
        os.symlink(target_file, output_target)

print("\n10% Model B dataset created successfully.")
print("Number of selected pairs:", len(selected_files))
print("Random seed:", SEED)
