
import torch
import sys

# Import the Model D Restormer architecture
from basicsr_film.archs.restormer_film_arch import Restormer

def main():

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create FiLM-enabled Restormer
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ).to(device)

    model.eval()

    # Create a dummy input image
    batch_size = 2
    height = 128
    width = 128

    dummy_image = torch.randn(
        batch_size, 3, height, width
    ).to(device)

    # Create a dummy 8-dimensional augmentation vector
    dummy_params = torch.zeros(
        batch_size, 8
    ).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_image, dummy_params)

    # Print results
    print("\n----- FiLM Restormer Test -----")
    print("Input shape: ", dummy_image.shape)
    print("Condition shape:", dummy_params.shape)
    print("Output shape:", output.shape)

    # Check output dimensions
    assert output.shape == dummy_image.shape, \
        "Output shape does not match input shape!"

    # Check for invalid values
    assert torch.isfinite(output).all(), \
        "Output contains NaN or Inf values!"

    print("\nSUCCESS: FiLM Restormer forward pass works!")


if __name__ == "__main__":
    main()
