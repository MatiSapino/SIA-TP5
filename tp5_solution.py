import numpy as np
import matplotlib.pyplot as plt
import re

from multi_layer_perceptron import MultiLayerPerceptron
from noise_generator import generate_augmented_dataset
from VariationalAutoencoder import VariationalAutoencoder

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
        latent_coords.append(model.outputs[2].copy())
    return np.array(latent_coords)

def generate_from_latent(model, latent_vector):
    if not isinstance(latent_vector, np.ndarray):
        latent_vector = np.array(latent_vector)

    weights_3 = model.weights[2]
    bias_3 = model.biases[2]
    z_3 = weights_3 @ latent_vector + bias_3
    a_3 = model.activation(z_3)

    weights_4 = model.weights[3]
    bias_4 = model.biases[3]
    z_4 = weights_4 @ a_3 + bias_4
    a_4 = model.activation(z_4)

    return (a_4 > 0.5).astype(int)

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

def main():
    np.random.seed(42)
    print("\n" + "="*50)
    print("## Part 1a: Basic Autoencoder (AE) ##")
    print("="*50)

    font_data, char_names = load_font_data("font.h")
    if font_data is None:
        return

    training_data_ae = [(x, x) for x in font_data]
    ae_sizes = [35, 32, 2, 32, 35]

    ae = MultiLayerPerceptron(
        sizes=ae_sizes,
        learning_rate=0.001,
        epochs=20000,
        error_threshold=1,
        training_data=training_data_ae,
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


    idx_a = char_names.index('a')
    idx_b = char_names.index('b')

    vec_a = latent_coords[idx_a]
    vec_b = latent_coords[idx_b]

    mid_point = 0.5 * (vec_a + vec_b)

    new_char_vec = generate_from_latent(ae, mid_point)

    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    plot_character(font_data[idx_a], f"Original '{char_names[idx_a]}'", ax=axes[0])
    plot_character(font_data[idx_b], f"Original '{char_names[idx_b]}'", ax=axes[1])
    plot_character(new_char_vec, "Generated (Midpoint)", ax=axes[2])
    fig.suptitle("Latent Space Representation of Font Characters (AE)", fontsize=14)
    plt.tight_layout()
    plt.savefig("generated_character_ae.png")

    print("\n" + "=" * 50)
    print("## Part 1b: Denoising Autoencoder (DAE) ##")
    print("=" * 50)

    font_data_list = [x.astype(float) for x in font_data]
    noisy_data_list = generate_augmented_dataset(
        font_data_list,
        copies=1,
        stddev=0.4,
        include_original=False,
        binarize=False,
        seed=42
    )

    training_data_dae = list(zip(noisy_data_list, font_data_list))
    dae = MultiLayerPerceptron(
        sizes = ae_sizes,
        learning_rate=0.001,
        epochs=10000,
        error_threshold=0.5 * len(font_data),
        training_data=training_data_dae,
        activation="sigmoid",
        optimizer="adam",
        task="autoencoder"
    )
    dae.train()

    indices_ejemplo = [char_names.index('g'), char_names.index('m'), char_names.index('z')]

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for i, idx in enumerate(indices_ejemplo):
        original = font_data_list[idx]

        ruidoso = generate_augmented_dataset(
            [original],
            copies=1,
            stddev=0.4,
            include_original=False,
            binarize=False,
            seed=i
        )[0]

        reconstruido = dae.predict(ruidoso)
        reconstruido_bin = (reconstruido > 0.5).astype(int)

        plot_character(original, f"Original '{char_names[idx]}'", ax=axes[i, 0])
        plot_character(ruidoso, "Noisy Input", ax=axes[i, 1])
        plot_character(reconstruido_bin, "Reconstructed", ax=axes[i, 2])

    fig.suptitle("Latent Space Representation of Font Characters (DAE)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("denoising_autoencoder_results.png")

    print("\n" + "=" * 50)
    print("## Part 2: Variational Autoencoder (VAE) ##")
    print("=" * 50)

    x_train_mnist, y_train_mnist = load_mnist_csv("data/mnist_train.csv", n_samples=20000)
    input_dim = 784
    hidden_enc = [256, 128]
    latent_dim = 10
    hidden_dec = [128, 256]

    vae = VariationalAutoencoder(
        input_dim=input_dim,
        hidden_dims_encoder=hidden_enc,
        latent_dim=latent_dim,
        hidden_dims_decoder=hidden_dec,
        learning_rate=0.0005,
        beta=0.1
    )
    vae.fit(x_train_mnist, epochs=100, batch_size=64)

    generated_images = vae.generate(n_samples=10)

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i, img in enumerate(generated_images):
        plot_digit(img, ax=axes[i])
    fig.suptitle("Generated Digits from VAE", fontsize=16)
    plt.savefig("generated_digits_ae.png")

if __name__ == "__main__":
    main()