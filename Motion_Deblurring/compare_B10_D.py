
import os
import csv
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = "/data/DATASETS/GoPro/test/GoPro/input"
TARGET_DIR = "/data/DATASETS/GoPro/test/GoPro/target"

B10_DIR = (
    "/data/ai23btech11029/Restormer/"
    "experiments/Deblurring_Restormer_10/"
    "visualization/restored/GoPro"
)

D_DIR = (
    "/data/ai23btech11029/Restormer/"
    "experiments/Deblurring_Restormer_D_FiLM_10/"
    "visualization/restored/GoPro"
)

OUTPUT_DIR = (
    "/data/ai23btech11029/Restormer/"
    "experiments/B10_vs_D_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPARISON_DIR = os.path.join(OUTPUT_DIR, "top_comparisons")
os.makedirs(COMPARISON_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUTPUT_DIR, "comparison_metrics.csv")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")


def get_image_paths(root):
    """Find images recursively and preserve relative paths."""
    image_paths = {}

    for current_root, _, files in os.walk(root):
        for filename in files:
            if filename.lower().endswith(VALID_EXTENSIONS):
                full_path = os.path.join(current_root, filename)
                relative_path = os.path.relpath(full_path, root)
                image_paths[relative_path] = full_path

    return image_paths


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def create_comparison(input_path, b10_path, d_path,
                      target_path, output_path):

    images = [
        Image.open(p).convert("RGB")
        for p in [input_path, b10_path, d_path, target_path]
    ]

    width, height = images[0].size

    # Ensure dimensions are identical for display.
    images = [
        img if img.size == (width, height)
        else img.resize((width, height))
        for img in images
    ]

    labels = [
        "Blurred Input",
        "Model B-10",
        "Model D",
        "Ground Truth"
    ]

    label_height = 40

    canvas = Image.new(
        "RGB",
        (width * 4, height + label_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * width, label_height))
        draw.text(
            (i * width + 10, 12),
            label,
            fill="black"
        )

    canvas.save(output_path)


# --------------------------------------------------
# Find matching images
# --------------------------------------------------

inputs = get_image_paths(INPUT_DIR)
targets = get_image_paths(TARGET_DIR)
b10_images = get_image_paths(B10_DIR)
d_images = get_image_paths(D_DIR)

common_files = sorted(
    set(inputs) & set(targets) & set(b10_images) & set(d_images)
)

print("=" * 65)
print("MODEL B-10 vs MODEL D COMPARISON")
print("=" * 65)

print("Input images:", len(inputs))
print("Ground truth images:", len(targets))
print("B-10 restored images:", len(b10_images))
print("D restored images:", len(d_images))
print("Common matching images:", len(common_files))

if not common_files:
    raise RuntimeError("No matching images found. Check the paths.")


# --------------------------------------------------
# Calculate PSNR and SSIM for each model
# --------------------------------------------------

results = []

for i, relative_path in enumerate(common_files, start=1):

    input_path = inputs[relative_path]
    target_path = targets[relative_path]
    b10_path = b10_images[relative_path]
    d_path = d_images[relative_path]

    target = load_image(target_path)
    b10 = load_image(b10_path)
    d = load_image(d_path)

    if target.shape != b10.shape or target.shape != d.shape:
        print("[SKIPPED: shape mismatch]", relative_path)
        continue

    psnr_b10 = peak_signal_noise_ratio(
        target, b10, data_range=1.0
    )

    psnr_d = peak_signal_noise_ratio(
        target, d, data_range=1.0
    )

    ssim_b10 = structural_similarity(
        target, b10, channel_axis=-1, data_range=1.0
    )

    ssim_d = structural_similarity(
        target, d, channel_axis=-1, data_range=1.0
    )

    delta_psnr = psnr_d - psnr_b10
    delta_ssim = ssim_d - ssim_b10

    results.append({
        "filename": relative_path,
        "PSNR_B10": psnr_b10,
        "PSNR_D": psnr_d,
        "Delta_PSNR": delta_psnr,
        "SSIM_B10": ssim_b10,
        "SSIM_D": ssim_d,
        "Delta_SSIM": delta_ssim
    })

    if i % 100 == 0:
        print(f"Processed {i}/{len(common_files)}")


# --------------------------------------------------
# Sort by PSNR improvement
# --------------------------------------------------

results.sort(
    key=lambda x: x["Delta_PSNR"],
    reverse=True
)

# Save all image metrics
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "PSNR_B10",
            "PSNR_D",
            "Delta_PSNR",
            "SSIM_B10",
            "SSIM_D",
            "Delta_SSIM"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# Save top 5 images where D improves PSNR
# --------------------------------------------------

positive_results = [
    r for r in results if r["Delta_PSNR"] > 0
]

top_results = positive_results[:5]

for rank, result in enumerate(top_results, start=1):

    relative_path = result["filename"]

    safe_name = relative_path.replace("/", "_").replace("\\", "_")

    output_path = os.path.join(
        COMPARISON_DIR,
        f"Rank_{rank}_{safe_name}"
    )

    create_comparison(
        inputs[relative_path],
        b10_images[relative_path],
        d_images[relative_path],
        targets[relative_path],
        output_path
    )


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 65)
print("TOP 5 IMAGES WHERE MODEL D IMPROVES PSNR")
print("=" * 65)

if not top_results:
    print("No images found where Model D has higher PSNR than B-10.")

for rank, result in enumerate(top_results, start=1):

    print(f"\nRank {rank}: {result['filename']}")
    print(f"B-10 PSNR : {result['PSNR_B10']:.4f} dB")
    print(f"D PSNR    : {result['PSNR_D']:.4f} dB")
    print(f"Difference: {result['Delta_PSNR']:+.4f} dB")
    print(f"B-10 SSIM : {result['SSIM_B10']:.6f}")
    print(f"D SSIM    : {result['SSIM_D']:.6f}")
    print(f"SSIM diff : {result['Delta_SSIM']:+.6f}")

print("\nTotal evaluated:", len(results))
print("Images where D has higher PSNR:", len(positive_results))
print("CSV:", CSV_FILE)
print("Top comparisons:", COMPARISON_DIR)
print("=" * 65)
