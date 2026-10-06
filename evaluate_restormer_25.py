import os
import csv
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ============================================================
# Paths
# ============================================================

INPUT_DIR = "/data/DATASETS/GoPro/test/GoPro/input"
TARGET_DIR = "/data/DATASETS/GoPro/test/GoPro/target"

RESTORED_DIR = (
    "/data/ai23btech11029/Restormer/"
    "experiments/Deblurring_Restormer_25/"
    "visualization/restored/GoPro"
)

OUTPUT_DIR = (
    "/data/ai23btech11029/Restormer/"
    "experiments/Deblurring_Restormer_25/"
    "visualization"
)

COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary.txt")

os.makedirs(COMPARISON_DIR, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def load_image(path):
    """Load RGB image and convert to float [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = np.asarray(img).astype(np.float32) / 255.0
    return img


def create_comparison(input_img, restored_img, target_img, output_path):
    """
    Create:
    Blurred Input | Restormer Output | Ground Truth
    """

    input_pil = Image.open(input_img).convert("RGB")
    restored_pil = Image.open(restored_img).convert("RGB")
    target_pil = Image.open(target_img).convert("RGB")

    # Make sure all three have the same dimensions
    width, height = input_pil.size

    if restored_pil.size != (width, height):
        restored_pil = restored_pil.resize((width, height))

    if target_pil.size != (width, height):
        target_pil = target_pil.resize((width, height))

    label_height = 45

    comparison = Image.new(
        "RGB",
        (width * 3, height + label_height),
        "white"
    )

    comparison.paste(input_pil, (0, label_height))
    comparison.paste(restored_pil, (width, label_height))
    comparison.paste(target_pil, (width * 2, label_height))

    draw = ImageDraw.Draw(comparison)

    draw.text((width // 2 - 50, 12), "Blurred Input", fill="black")
    draw.text((width + width // 2 - 60, 12), "Restormer Output", fill="black")
    draw.text((2 * width + width // 2 - 50, 12), "Ground Truth", fill="black")

    comparison.save(output_path)


# ============================================================
# Find test images
# ============================================================

files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

print("=" * 60)
print("Restormer Evaluation")
print("=" * 60)
print(f"Number of input images: {len(files)}")
print()

if len(files) == 0:
    raise RuntimeError("No test input images found.")

# ============================================================
# Evaluate
# ============================================================

results = []

total_psnr = 0.0
total_ssim = 0.0
processed = 0

for i, filename in enumerate(files, start=1):

    input_path = os.path.join(INPUT_DIR, filename)
    target_path = os.path.join(TARGET_DIR, filename)
    restored_path = os.path.join(RESTORED_DIR, filename)

    if not os.path.exists(target_path):
        print(f"[WARNING] Missing target: {filename}")
        continue

    if not os.path.exists(restored_path):
        print(f"[WARNING] Missing restored image: {filename}")
        continue

    # Load images
    restored = load_image(restored_path)
    target = load_image(target_path)

    # Make sure dimensions match
    if restored.shape != target.shape:
        print(
            f"[WARNING] Shape mismatch: {filename} "
            f"Restored={restored.shape}, Target={target.shape}"
        )
        continue

    # PSNR
    psnr = peak_signal_noise_ratio(
        target,
        restored,
        data_range=1.0
    )

    # SSIM
    ssim = structural_similarity(
        target,
        restored,
        channel_axis=-1,
        data_range=1.0
    )

    results.append([filename, psnr, ssim])

    total_psnr += psnr
    total_ssim += ssim
    processed += 1

    # Create comparison image
    comparison_path = os.path.join(
        COMPARISON_DIR,
        filename
    )

    create_comparison(
        input_path,
        restored_path,
        target_path,
        comparison_path
    )

    if i % 50 == 0 or i == len(files):
        print(
            f"Processed {i}/{len(files)} "
            f"| PSNR: {psnr:.4f} "
            f"| SSIM: {ssim:.4f}"
        )


# ============================================================
# Average metrics
# ============================================================

if processed == 0:
    raise RuntimeError("No images were successfully evaluated.")

average_psnr = total_psnr / processed
average_ssim = total_ssim / processed


# ============================================================
# Save metrics.csv
# ============================================================

with open(METRICS_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "PSNR",
        "SSIM"
    ])

    for filename, psnr, ssim in results:
        writer.writerow([
            filename,
            f"{psnr:.6f}",
            f"{ssim:.6f}"
        ])


# ============================================================
# Save summary.txt
# ============================================================

with open(SUMMARY_FILE, "w") as f:
    f.write("Restormer GoPro Test Results\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Number of input images: {len(files)}\n")
    f.write(f"Number successfully evaluated: {processed}\n\n")

    f.write(f"Average PSNR: {average_psnr:.6f} dB\n")
    f.write(f"Average SSIM: {average_ssim:.6f}\n")


# ============================================================
# Final output
# ============================================================

print()
print("=" * 60)
print("Evaluation Complete")
print("=" * 60)
print(f"Images evaluated : {processed}")
print(f"Average PSNR     : {average_psnr:.6f} dB")
print(f"Average SSIM     : {average_ssim:.6f}")
print()
print(f"Metrics          : {METRICS_FILE}")
print(f"Summary          : {SUMMARY_FILE}")
print(f"Comparisons      : {COMPARISON_DIR}")
print("=" * 60)
