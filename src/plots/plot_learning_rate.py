import glob
import re
import matplotlib.pyplot as plt


error_regex = re.compile(r"Error:\s*([0-9.]+)")
epoch_regex = re.compile(r"Epoch\s+(\d+)/")

files = sorted(glob.glob("eta_*.txt"))

plt.figure(figsize=(10, 6))

for filename in files:

    lr_str = filename.split("_")[1].replace(".txt", "")
    learning_rate = float(lr_str)

    epochs = []
    errors = []

    with open(filename, "r") as f:
        for line in f:
            e_epoch = epoch_regex.search(line)
            e_error = error_regex.search(line)

            if e_epoch and e_error:
                epochs.append(int(e_epoch.group(1)))
                errors.append(float(e_error.group(1)))

    plt.plot(epochs, errors, label=f"eta = {learning_rate}")

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error vs Epoch para distintos Learning Rates")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
