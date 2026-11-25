
import os
import cv2
import csv
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = "/data/ai23btech11029/Restormer"

INPUT_DIR = "/data/DATASETS/GoPro/test/GoPro/input"

TARGET_DIR = "/data/DATASETS/GoPro/test/GoPro/target"

RESTORED_DIR = os.path.join(
    PROJECT_DIR,
    "experiments/Deblurring_Restormer_D_FiLM_10/visualization/restored/GoPro"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "experiments/Deblurring_Restormer_D_FiLM_10/evaluation"
)

os.makedirs(RESULT_DIR, exist_ok=True)

CSV_PATH = os.path.join(
    RESULT_DIR,
    "metrics.csv"
)

SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    "summary.txt"
)

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


# ============================================================
# Find images recursively
# ============================================================

def get_image_paths(directory):

    image_paths = {}

    for root, _, files in os.walk(directory):

        for filename in files:

            if filename.lower().endswith(VALID_EXTENSIONS):

                full_path = os.path.join(root, filename)

                relative_path = os.path.relpath(
                    full_path,
                    directory
                )

                image_paths[relative_path] = full_path

    return image_paths


print("=" * 65)
print("Restormer Model D FiLM - GoPro Evaluation")
print("=" * 65)

input_images = get_image_paths(INPUT_DIR)
target_images = get_image_paths(TARGET_DIR)
restored_images = get_image_paths(RESTORED_DIR)

print("Input images:", len(input_images))
print("Ground truth images:", len(target_images))
print("Restored images:", len(restored_images))


# ============================================================
# Match images using relative paths
# ============================================================

common_paths = sorted(
    set(input_images.keys())
    & set(target_images.keys())
    & set(restored_images.keys())
)

if not common_paths:
    raise RuntimeError(
        "No matching input, target and restored images found."
    )

missing_restored = sorted(
    set(input_images.keys()) - set(restored_images.keys())
)

missing_target = sorted(
    set(input_images.keys()) - set(target_images.keys())
)

print("\nMatched images:", len(common_paths))
print("Missing restored images:", len(missing_restored))
print("Missing ground truth images:", len(missing_target))


# ============================================================
# Evaluation
# ============================================================

results = []

psnr_values = []
ssim_values = []

skipped = 0

for relative_path in common_paths:

    input_path = input_images[relative_path]
    target_path = target_images[relative_path]
    restored_path = restored_images[relative_path]

    # Read images
    target = cv2.imread(
        target_path,
        cv2.IMREAD_COLOR
    )

    restored = cv2.imread(
        restored_path,
        cv2.IMREAD_COLOR
    )

    if target is None or restored is None:

        print("Skipping unreadable image:", relative_path)
        skipped += 1
        continue

    # Convert BGR to RGB
    target = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2RGB
    )

    restored = cv2.cvtColor(
        restored,
        cv2.COLOR_BGR2RGB
    )

    # Check image dimensions
    if target.shape != restored.shape:

        print(
            "Skipping shape mismatch:",
            relative_path,
            target.shape,
            restored.shape
        )

        skipped += 1
        continue

    # Calculate PSNR
    psnr = peak_signal_noise_ratio(
        target,
        restored,
        data_range=255
    )

    # Calculate SSIM
    ssim = structural_similarity(
        target,
        restored,
        channel_axis=-1,
        data_range=255
    )

    psnr_values.append(psnr)
    ssim_values.append(ssim)

    results.append([
        relative_path,
        round(psnr, 4),
        round(ssim, 6)
    ])


# ============================================================
# Calculate average metrics
# ============================================================

if not results:
    raise RuntimeError(
        "No images could be evaluated. Check image paths and dimensions."
    )

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print("\n" + "=" * 65)
print("MODEL D EVALUATION RESULTS")
print("=" * 65)

print("Successfully evaluated:", len(results))
print("Skipped images:", skipped)

print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.6f}")


# ============================================================
# Save individual image metrics
# ============================================================

with open(CSV_PATH, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Image",
        "PSNR (dB)",
        "SSIM"
    ])

    writer.writerows(results)

    writer.writerow([
        "AVERAGE",
        round(avg_psnr, 4),
        round(avg_ssim, 6)
    ])


# ============================================================
# Save summary
# ============================================================

with open(SUMMARY_PATH, "w") as f:

    f.write("Restormer Model D FiLM - GoPro Evaluation\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Input images: {len(input_images)}\n")
    f.write(f"Ground truth images: {len(target_images)}\n")
    f.write(f"Restored images: {len(restored_images)}\n")
    f.write(f"Successfully evaluated: {len(results)}\n")
    f.write(f"Skipped images: {skipped}\n\n")

    f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
    f.write(f"Average SSIM: {avg_ssim:.6f}\n")

    f.write(f"\nMissing restored images: {len(missing_restored)}\n")
    f.write(f"Missing ground truth images: {len(missing_target)}\n")


# ============================================================
# Final output
# ============================================================

print("\nEvaluation completed.")

print("Metrics CSV:", CSV_PATH)
print("Summary:", SUMMARY_PATH)

print("=" * 65)
