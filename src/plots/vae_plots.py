import numpy as np
import matplotlib.pyplot as plt
import os


def plot_vae_reconstructions(vae, images, labels, output_path="vae_reconstruction_comparison.png"):
    """
    Plot original vs reconstructed images for one example of each digit class.
    
    Args:
        vae: Trained VariationalAutoencoder instance
        images: Array of MNIST images
        labels: Array of labels for the images
        output_path: Path to save the figure
    """
    print("Creating reconstruction comparison...")
    
    # Find one example of each digit
    digit_examples = {}
    for i, label in enumerate(labels):
        if label not in digit_examples:
            digit_examples[label] = images[i]
        if len(digit_examples) == 10:
            break
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for digit in range(10):
        original = digit_examples[digit]
        reconstructed = vae.reconstruct(original)
        
        # Original on top row
        axes[0, digit].imshow(original.reshape(28, 28), cmap='gray_r', interpolation='nearest')
        axes[0, digit].set_title(f'Original {digit}')
        axes[0, digit].axis('off')
        
        # Reconstructed on bottom row
        axes[1, digit].imshow(reconstructed.reshape(28, 28), cmap='gray_r', interpolation='nearest')
        axes[1, digit].set_title(f'Reconstructed')
        axes[1, digit].axis('off')
    
    fig.suptitle("VAE Reconstruction: Original vs Reconstructed (One per Digit)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_vae_latent_space(vae, images, labels, sample_size=2000, output_path="vae_latent_space.png"):
    """
    Plot 2D latent space with points colored by digit class.
    
    Args:
        vae: Trained VariationalAutoencoder instance
        images: Array of MNIST images
        labels: Array of labels for the images
        sample_size: Number of points to sample for visualization
        output_path: Path to save the figure
    """
    print("Creating latent space visualization...")
    
    sample_size = min(sample_size, len(images))
    sample_indices = np.random.choice(len(images), sample_size, replace=False)
    
    latent_coords = []
    latent_labels = []
    for idx in sample_indices:
        mu = vae.encode(images[idx])
        latent_coords.append(mu)
        latent_labels.append(labels[idx])
    
    latent_coords = np.array(latent_coords)
    latent_labels = np.array(latent_labels)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(latent_coords[:, 0], latent_coords[:, 1], 
                         c=latent_labels, cmap='tab10', s=10, alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label='Digit')
    plt.title("VAE Latent Space (2D) - MNIST Digits", fontsize=16)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_vae_grid_generation(vae, n_grid=5, grid_range=(-3, 3), output_path="vae_grid_generation.png"):
    """
    Generate and plot digits from a regular grid of latent coordinates.
    
    Args:
        vae: Trained VariationalAutoencoder instance
        n_grid: Number of points along each dimension
        grid_range: Tuple of (min, max) values for the grid
        output_path: Path to save the figure
    """
    print("Creating grid generation visualization...")
    
    grid_values = np.linspace(grid_range[0], grid_range[1], n_grid)
    grid_coords = []
    
    for y_val in grid_values:
        for x_val in grid_values:
            grid_coords.append([x_val, y_val])
    
    grid_coords = np.array(grid_coords)
    grid_images = vae.generate_from_latent(grid_coords)
    
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(12, 12))
    for i in range(n_grid):
        for j in range(n_grid):
            idx = i * n_grid + j
            axes[i, j].imshow(grid_images[idx], cmap='gray_r', interpolation='nearest')
            axes[i, j].set_title(f'({grid_coords[idx][0]:.1f}, {grid_coords[idx][1]:.1f})', fontsize=8)
            axes[i, j].axis('off')
    
    fig.suptitle("VAE Generated Digits from Latent Space Grid", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_vae_random_generation(vae, n_samples=10, output_path="vae_random_generation.png"):
    """
    Generate random digits and display them with their latent coordinates.
    
    Args:
        vae: Trained VariationalAutoencoder instance
        n_samples: Number of random samples to generate
        output_path: Path to save the figure
    """
    print("Creating random generation visualization...")
    
    generated_images, latent_coords = vae.generate(n_samples=n_samples)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(n_samples):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(generated_images[i], cmap='gray_r', interpolation='nearest')
        coord_text = f'({latent_coords[i][0]:.2f}, {latent_coords[i][1]:.2f})'
        axes[row, col].set_title(coord_text, fontsize=10)
        axes[row, col].axis('off')
    
    fig.suptitle("VAE Random Generation with Latent Coordinates", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_all_vae_visualizations(vae, images, labels, output_dir="."):
    """
    Create all VAE visualizations at once.
    
    Args:
        vae: Trained VariationalAutoencoder instance
        images: Array of MNIST images
        labels: Array of labels for the images
        output_dir: Directory to save all figures
    """
    
    print("\n" + "=" * 50)
    print("Generating VAE visualizations...")
    print("=" * 50)
    
    plot_vae_reconstructions(vae, images, labels, 
                            output_path=os.path.join(output_dir, "vae_reconstruction_comparison.png"))
    
    plot_vae_latent_space(vae, images, labels, sample_size=2000,
                         output_path=os.path.join(output_dir, "vae_latent_space.png"))
    
    plot_vae_grid_generation(vae, n_grid=5, grid_range=(-3, 3),
                            output_path=os.path.join(output_dir, "vae_grid_generation.png"))
    
    plot_vae_random_generation(vae, n_samples=10,
                              output_path=os.path.join(output_dir, "vae_random_generation.png"))
    
    print("\n" + "=" * 50)
    print("All VAE visualizations completed!")
    print("=" * 50)
