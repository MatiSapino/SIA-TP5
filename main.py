import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

from src.autoencoder.multi_layer_perceptron import MultiLayerPerceptron
from noise_generator import generate_augmented_dataset
from src.autoencoder.VariationalAutoencoder import VariationalAutoencoder
from src.plots.vae_plots import create_all_vae_visualizations

def load_font_data(path_to_font_h="font.h"):
    with open(path_to_font_h, 'r') as f:
        content = f.read()

    pattern = re.compile(r"\{([\s,0-9xabcdefABCDEF]+)\},\s*//\s*(\S+),\s*(.+)")
    matches = pattern.findall(content)

    if not matches:
        print("No matches found")
        return None

    all_characters = []
    char_names = []

    for match in matches:
        hex_values_str = match[0].split(',')
        char_code = match[1]
        char_name = match[2].strip()

        hex_values = [int(h, 16) for h in hex_values_str]
        binary_char = np.zeros((7, 5), dtype=int)

        for row_idx, hex_val in enumerate(hex_values):
            current_val = hex_val
            for col_idx in range(5):
                binary_char[row_idx, 4 - col_idx] = current_val & 1
                current_val >>= 1

        all_characters.append(binary_char.flatten())
        char_names.append(char_name)

    print(f"Loaded {len(all_characters)} characters from font data.")
    return np.array(all_characters), char_names

def plot_character(character_vector, title="", ax=None):
    if character_vector is None:
        return
    img = character_vector.reshape(7, 5)
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(img, cmap='binary', interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def get_latent_representations(model, data):
    latent_coords = []
    for x in data:
        model.forward(x)
        latent_coords.append(model.outputs[4].copy())
    return np.array(latent_coords)

def generate_from_latent(model, latent_vector):
    if not isinstance(latent_vector, np.ndarray):
        latent_vector = np.array(latent_vector)

    current_output = latent_vector

    for i in range(4, model.layers - 1):
        weights = model.weights[i]
        bias = model.biases[i]
        z = weights @ current_output + bias
        current_output = model.activation(z)

    return (current_output > 0.5).astype(int)

def load_mnist_csv(path_to_csv, n_samples=None):
    data = np.loadtxt(path_to_csv, delimiter=",", skiprows=1)

    if n_samples:
        data = data[:n_samples]

    labels = data[:, 0].astype(int)
    images = data[:, 1:]
    images = images / 255.0

    return images, labels

def plot_digit(digit_vector, title="", ax=None):
    if digit_vector is None:
        return
    img = digit_vector.reshape(28, 28)
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(img, cmap='gray_r', interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])




def run_basic_autoencoder(font_path="src/data/font.h"):
    print("\n" + "=" * 50)
    print("## Part 1a: Basic Autoencoder (AE) ##")
    print("=" * 50)

    font_data, char_names = load_font_data(font_path)
    if font_data is None:
        return

    training_data = [(x, x) for x in font_data]

    ae_sizes = [35, 32, 24, 16, 2, 16, 24, 32, 35]
    ae = MultiLayerPerceptron(
        sizes=ae_sizes,
        learning_rate=0.000985,
        epochs=8000,
        error_threshold=1,
        training_data=training_data,
        activation="sigmoid",
        optimizer="adam",
        task="autoencoder"
    )
    ae.train()

    latent_coords = get_latent_representations(ae, font_data)

    plt.figure(figsize=(12, 8))
    plt.scatter(latent_coords[:, 0], latent_coords[:, 1], s=10)
    for i, name in enumerate(char_names):
        plt.text(latent_coords[i, 0] + 0.01, latent_coords[i, 1] + 0.01, name, fontsize=9)
    plt.title("Latent Space Representation of Font Characters (AE)")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("latent_space_representation_ae.png")


def run_denoising_autoencoder(font_path="src/data/font.h"):
    print("\n" + "=" * 50)
    print("## Part 1b: Denoising Autoencoder (DAE) ##")
    print("=" * 50)

    font_data, char_names = load_font_data(font_path)
    if font_data is None:
        return
    
    ae_sizes = [35, 32, 24, 16, 2, 16, 24, 32, 35]
    copies = 5
    seed = 42

    font_data_list = [x.astype(float) for x in font_data]
    noisy_data_list = generate_augmented_dataset(
        font_data_list,
        copies=copies,
        stddev=0.2,
        include_original=True,
        binarize=True,
        seed=seed,
        force_change=True
    )
    clean_targets = font_data_list * (copies + 1)
    training_data_dae = list(zip(noisy_data_list, clean_targets))

    dae = MultiLayerPerceptron(
        sizes = ae_sizes,
        learning_rate=0.000985,
        epochs=2000,
        error_threshold=5,
        training_data=training_data_dae,
        activation="sigmoid",
        optimizer="adam",
        task="autoencoder"
    )
    dae.train()

    noise_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    samples_per_level = 20   # cantidad de muestras de ruido por nivel
    character_errors = {name: [] for name in char_names}
    noise_samples = {}       # (char_name, noise) -> lista de muestras ruidosas

    for i, name in enumerate(char_names):
        original = font_data_list[i]

        for noise in noise_levels:
            errors = []
            noisy_examples = []

            for s in range(samples_per_level):
                noisy = generate_augmented_dataset(
                    [original],
                    copies=1,
                    stddev=noise,
                    include_original=False,
                    binarize=True,
                    seed=(i * 100 * seed + s),
                    force_change=True
                )[0]

                reconstructed = dae.predict(noisy)
                reconstructed_bin = (reconstructed > 0.5).astype(int)

                err = np.sum(original != reconstructed_bin)
                errors.append(err)
                noisy_examples.append(noisy)

            avg_err = np.mean(errors)
            character_errors[name].append(avg_err)

            key = (name, round(noise, 3))
            noise_samples[key] = noisy_examples

    plt.figure(figsize=(14, 8))

    for name, errors in character_errors.items():
        plt.plot(noise_levels, errors, marker="o", linestyle="-", alpha=0.6, label=name)

    plt.title(f"Per-Character Reconstruction Error vs Noise Level ({samples_per_level} runs)")
    plt.xlabel("Noise Std Dev")
    plt.ylabel("Reconstruction Error (bit differences)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("src/plots/dae_error_vs_noise_avg.png")
    plt.show()

    # Compute global average error per noise level
    avg_error_per_noise = []

    for idx, noise in enumerate(noise_levels):
        # Tomar el error de cada letra para ese nivel de ruido
        errors_at_noise = [character_errors[name][idx] for name in char_names]
        avg_error_per_noise.append(np.mean(errors_at_noise))

    plt.figure(figsize=(7, 5))
    plt.plot(noise_levels, avg_error_per_noise, marker="o", linewidth=2)
    plt.title(f"Average Reconstruction Error vs Noise Level ({samples_per_level} samples)")
    plt.xlabel("Noise Std Dev")
    plt.ylabel("Average Reconstruction Error (bit differences)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("src/plots/dae_avg_error_vs_noise.png")
    plt.show()

    noise_levels = [0.2, 0.3, 0.4]
    indices_ejemplo = [char_names.index('a'), char_names.index('b'), char_names.index('c'),
                       char_names.index('d'), char_names.index('e'), char_names.index('o')]

    for noise in noise_levels:
        for idx in indices_ejemplo:
            name = char_names[idx]
            original = font_data_list[idx]

            noisy = noise_samples[(name, round(noise, 3))][0]

            if np.array_equal(noisy, original):
                print(f"Saltado: '{name}' con σ={noise} — Noisy == Original")
                continue

            reconstructed = dae.predict(noisy)
            reconstructed_bin = (reconstructed > 0.5).astype(int)

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))

            plot_character(original, f"Original '{name}'", ax=axes[0])
            plot_character(noisy, f"Noisy", ax=axes[1])
            plot_character(reconstructed_bin, "Reconstructed", ax=axes[2])

            #fig.suptitle(f"Character '{name}' — Noise σ={noise}", fontsize=15)
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            #plt.show()
            plt.savefig(f"src/plots/dae_{name}_noise_{noise}.png")


def run_variational_autoencoder(mnist_path="data/mnist_train.csv"):
    print("\n" + "=" * 50)
    print("## Part 2: Variational Autoencoder (VAE) ##")
    print("=" * 50)

    x_train, y_train = load_mnist_csv(mnist_path, n_samples=20000)

    vae = VariationalAutoencoder(
        input_dim=784,
        hidden_dims_encoder=[128, 64],
        latent_dim=2,
        hidden_dims_decoder=[64, 128],
        learning_rate=0.001,
        beta=1
    )
    vae.fit(x_train, epochs=200, batch_size=64, loss_threshold=100.0, patience=10)

    create_all_vae_visualizations(vae, x_train, y_train)


def main():
    parser = argparse.ArgumentParser(description="Run Autoencoder Experiments")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["ae", "dae", "vae", "all"],
        default="all",
        help="Choose which model to run"
    )

    args = parser.parse_args()
    np.random.seed(42)

    if args.mode in ["ae", "all"]:
        run_basic_autoencoder()

    if args.mode in ["dae", "all"]:
        run_denoising_autoencoder()

    if args.mode in ["vae", "all"]:
        run_variational_autoencoder()


if __name__ == "__main__":
    main()