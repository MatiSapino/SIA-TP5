import re
import matplotlib.pyplot as plt

# Archivo a leer
path = "datos.txt"

# Expresiones regulares para extraer los valores numéricos
epoch_pat = re.compile(r"Epoch\s+(\d+)")
error_pat = re.compile(r"Error:\s*([\d.]+)")
max_pixels_pat = re.compile(r"Max_Incorrect_Pixels:\s*(\d+)")

epochs = []
errors = []
max_pixels = []

with open(path, "r") as f:
    for line in f:
        epoch_match = epoch_pat.search(line)
        error_match = error_pat.search(line)
        maxp_match = max_pixels_pat.search(line)

        if epoch_match and error_match and maxp_match:
            epochs.append(int(epoch_match.group(1)))
            errors.append(float(error_match.group(1)))
            max_pixels.append(int(maxp_match.group(1)))

# -----------------------------
# Plot 1: Error vs Epoch
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(epochs, errors, label="Error", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error vs Epoch")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: Max Incorrect Pixels vs Epoch
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(epochs, max_pixels, color="orange", label="Max Incorrect Pixels", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Max Incorrect Pixels")
plt.title("Max Incorrect Pixels vs Epoch")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
