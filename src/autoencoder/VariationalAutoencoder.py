# VariationalAutoencoder.py
# Implementación "a mano" (NumPy) para la Parte 2 del TP5

import numpy as np
import time


# --- Funciones de Activación y sus Derivadas ---

def sigmoid(z):
    """Aplica la función sigmoide, con clipping para estabilidad numérica."""
    z = np.clip(z, -500, 500)  # Evita overflow en np.exp
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(a):
    """Derivada de la función sigmoide (asumiendo que 'a' es la salida de la sigmoide)."""
    return a * (1 - a)


def relu(z):
    """Función de activación ReLU (Rectified Linear Unit)."""
    return np.maximum(0, z)


def relu_prime(a):
    """Derivada de la función ReLU (asumiendo que 'a' es la salida de ReLU)."""
    return (a > 0).astype(float)


class VariationalAutoencoder:
    """
    Implementación de un Autoencoder Variacional (VAE) desde cero usando solo NumPy.

    Basado en la teoría de 'Autoencoders.pdf' (págs. [cite_start]73-89) [cite: 580-681].
    """

    def __init__(self, input_dim, hidden_dims_encoder, latent_dim, hidden_dims_decoder, learning_rate=1e-3, beta=1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.beta = beta

        # --- Arquitectura del Encoder ---
        # El Encoder genera [mu, log_var], por eso la salida es latent_dim * 2
        self.encoder_sizes = [input_dim] + hidden_dims_encoder + [latent_dim * 2]
        self.encoder_weights = []
        self.encoder_biases = []

        sizes = self.encoder_sizes
        for i in range(len(sizes) - 1):
            # Inicialización Xavier/Glorot
            std_dev = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            self.encoder_weights.append(np.random.randn(sizes[i + 1], sizes[i]) * std_dev)
            self.encoder_biases.append(np.zeros(sizes[i + 1]))

        # --- Arquitectura del Decoder ---
        self.decoder_sizes = [latent_dim] + hidden_dims_decoder + [input_dim]
        self.decoder_weights = []
        self.decoder_biases = []

        sizes = self.decoder_sizes
        for i in range(len(sizes) - 1):
            std_dev = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            self.decoder_weights.append(np.random.randn(sizes[i + 1], sizes[i]) * std_dev)
            self.decoder_biases.append(np.zeros(sizes[i + 1]))

        # --- Funciones de Activación ---
        self.hidden_activation = relu
        self.hidden_activation_prime = relu_prime
        self.output_activation = sigmoid  # Sigmoide para la salida (píxeles [0, 1])

        # Almacenamiento temporal para backpropagation
        self.epsilon = None  # Ruido aleatorio usado en el reparameterization trick

    def _encoder_forward(self, x):
        """Pase forward solo a través del Encoder."""
        activations = [x]
        zs = []  # Almacena las entradas pre-activación (z = Wx + b)
        a = x

        # Capas ocultas del encoder
        for i in range(len(self.encoder_weights) - 1):
            z = self.encoder_weights[i] @ a + self.encoder_biases[i]
            a = self.hidden_activation(z)
            zs.append(z)
            activations.append(a)

        # Capa de salida del encoder (lineal, sin activación)
        z_out = self.encoder_weights[-1] @ a + self.encoder_biases[-1]
        zs.append(z_out)
        activations.append(z_out)  # Almacena la salida lineal

        # Dividir la salida en mu y log_var
        mu = z_out[:self.latent_dim]
        log_var = z_out[self.latent_dim:]

        return mu, log_var, activations, zs

    def _reparameterize(self, mu, log_var):
        """
        [cite_start]Aplica el "Reparameterization Trick" [cite: 365-369, 683-704].
        z = mu + epsilon * std
        """
        std = np.exp(0.5 * log_var)
        self.epsilon = np.random.randn(self.latent_dim)  # Almacena epsilon para backprop
        z = mu + self.epsilon * std
        return z

    def _decoder_forward(self, z):
        """Pase forward solo a través del Decoder."""
        activations = [z]
        zs = []
        a = z

        # Capas ocultas del decoder
        for i in range(len(self.decoder_weights) - 1):
            z_layer = self.decoder_weights[i] @ a + self.decoder_biases[i]
            a = self.hidden_activation(z_layer)
            zs.append(z_layer)
            activations.append(a)

        # Capa de salida del decoder (con activación sigmoide)
        z_out = self.decoder_weights[-1] @ a + self.decoder_biases[-1]
        x_recon = self.output_activation(z_out)
        zs.append(z_out)
        activations.append(x_recon)
        return x_recon, activations, zs

    def forward(self, x):
        """Pase forward completo del VAE."""
        mu, log_var, enc_activations, enc_zs = self._encoder_forward(x)
        z = self._reparameterize(mu, log_var)
        x_recon, dec_activations, dec_zs = self._decoder_forward(z)
        return x_recon, mu, log_var, z, enc_activations, enc_zs, dec_activations, dec_zs

    def _compute_loss(self, x, x_recon, mu, log_var):
        """
        [cite_start]Calcula la pérdida total del VAE = Reconstruction_Loss + KL_Loss[cite: 624].
        """
        # 1. Reconstruction Loss (Binary Cross-Entropy)
        # Se añade epsilon para evitar log(0)
        epsilon = 1e-7
        recon_loss = -np.sum(x * np.log(x_recon + epsilon) + (1 - x) * np.log(1 - x_recon + epsilon))

        # [cite_start]2. KL Loss (Divergencia KL) [cite: 618]
        kl_loss = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def backward(self, x, x_recon, mu, log_var, z, enc_activations, enc_zs, dec_activations, dec_zs):
        """
        Pase backward completo (Retropropagación) para el VAE.
        Calcula los gradientes para todos los pesos y biases.
        [cite_start][cite: 786-825]
        """

        # --- 1. Gradientes para el Decoder ---
        grad_decoder_weights = [np.zeros_like(w) for w in self.decoder_weights]
        grad_decoder_biases = [np.zeros_like(b) for b in self.decoder_biases]

        # Gradiente inicial: dLoss / dx_recon
        # Para BCE (recon_loss) y Sigmoid (output_activation), el gradiente
        # dLoss / dz_out (pre-activación) es simplemente (x_recon - x).
        delta = x_recon - x

        # Gradientes para la última capa del decoder
        grad_decoder_biases[-1] = delta
        grad_decoder_weights[-1] = np.outer(delta, dec_activations[-2])  # dec_activations[-2] es la 'a' de la capa anterior

        # Retropropagar a través de las capas ocultas del decoder
        for l in range(2, len(self.decoder_sizes)):
            a_prime = self.hidden_activation_prime(dec_activations[-l])
            delta = (self.decoder_weights[-l + 1].T @ delta) * a_prime

            grad_decoder_biases[-l] = delta
            grad_decoder_weights[-l] = np.outer(delta, dec_activations[-l - 1])

        # Gradiente de la pérdida (de reconstrucción) con respecto a 'z'
        # Este es el gradiente que fluye de vuelta al "Reparameterization Trick"
        dLoss_dz = self.decoder_weights[0].T @ delta

        # --- 2. Gradientes para el Encoder ---
        grad_encoder_weights = [np.zeros_like(w) for w in self.encoder_weights]
        grad_encoder_biases = [np.zeros_like(b) for b in self.encoder_biases]

        # [cite_start]Gradientes de la Pérdida KL con respecto a mu y log_var [cite: 799]
        dKL_dmu = self.beta * mu
        dKL_dlog_var = self.beta * (0.5 * (np.exp(log_var) - 1))

        # [cite_start]Gradientes del "Reparameterization Trick" [cite: 792, 799]
        std = np.exp(0.5 * log_var)
        dz_dmu = 1.0
        dz_dlog_var = self.epsilon * 0.5 * std

        # Gradiente total con respecto a mu y log_var
        # dLoss/dmu = dReconLoss/dmu + dKLoss/dmu
        # dLoss/dmu = (dReconLoss/dz) * (dz/dmu) + dKLoss/dmu
        dLoss_dmu = dLoss_dz * dz_dmu + dKL_dmu
        dLoss_dlog_var = dLoss_dz * dz_dlog_var + dKL_dlog_var

        # 'delta' para la capa de salida del encoder (la capa [mu, log_var])
        delta_encoder_out = np.concatenate((dLoss_dmu, dLoss_dlog_var))

        # Gradientes para la última capa del encoder
        grad_encoder_biases[-1] = delta_encoder_out
        grad_encoder_weights[-1] = np.outer(delta_encoder_out, enc_activations[-2])

        # Retropropagar a través de las capas ocultas del encoder
        delta = delta_encoder_out
        for l in range(2, len(self.encoder_sizes)):
            a_prime = self.hidden_activation_prime(enc_activations[-l])
            delta = (self.encoder_weights[-l + 1].T @ delta) * a_prime

            grad_encoder_biases[-l] = delta
            grad_encoder_weights[-l] = np.outer(delta, enc_activations[-l - 1])

        return grad_encoder_weights, grad_encoder_biases, grad_decoder_weights, grad_decoder_biases

    def fit(self, X_train, epochs, batch_size=64):
        """Entrena el VAE usando mini-batch gradient descent."""
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss, epoch_recon_loss, epoch_kl_loss = 0, 0, 0

            # Mezclar los datos al inicio de cada época
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]

            for i in range(0, num_samples, batch_size):
                batch = X_train_shuffled[i:i + batch_size]

                # Acumuladores de gradientes para el batch
                sum_grad_enc_w = [np.zeros_like(w) for w in self.encoder_weights]
                sum_grad_enc_b = [np.zeros_like(b) for b in self.encoder_biases]
                sum_grad_dec_w = [np.zeros_like(w) for w in self.decoder_weights]
                sum_grad_dec_b = [np.zeros_like(b) for b in self.decoder_biases]

                batch_loss, batch_recon_loss, batch_kl_loss = 0, 0, 0

                for x in batch:
                    # 1. Forward pass
                    x_recon, mu, log_var, z, enc_acts, enc_zs, dec_acts, dec_zs = self.forward(x)

                    # 2. Compute loss
                    loss, recon_loss, kl_loss = self._compute_loss(x, x_recon, mu, log_var)
                    batch_loss += loss
                    batch_recon_loss += recon_loss
                    batch_kl_loss += kl_loss

                    # 3. Backward pass (compute gradients)
                    g_ew, g_eb, g_dw, g_db = self.backward(x, x_recon, mu, log_var, z, enc_acts, enc_zs, dec_acts,
                                                           dec_zs)

                    # 4. Acumular gradientes
                    for j in range(len(g_ew)):
                        sum_grad_enc_w[j] += g_ew[j]
                        sum_grad_enc_b[j] += g_eb[j]
                    for j in range(len(g_dw)):
                        sum_grad_dec_w[j] += g_dw[j]
                        sum_grad_dec_b[j] += g_db[j]

                # 5. Actualizar pesos (usando el gradiente promedio del batch)
                batch_len = len(batch)
                for j in range(len(self.encoder_weights)):
                    self.encoder_weights[j] -= self.lr * (sum_grad_enc_w[j] / batch_len)
                    self.encoder_biases[j] -= self.lr * (sum_grad_enc_b[j] / batch_len)
                for j in range(len(self.decoder_weights)):
                    self.decoder_weights[j] -= self.lr * (sum_grad_dec_w[j] / batch_len)
                    self.decoder_biases[j] -= self.lr * (sum_grad_dec_b[j] / batch_len)

                epoch_loss += batch_loss
                epoch_recon_loss += batch_recon_loss
                epoch_kl_loss += batch_kl_loss

            # Imprimir estadísticas de la época
            avg_loss = epoch_loss / num_samples
            avg_recon_loss = epoch_recon_loss / num_samples
            avg_kl_loss = epoch_kl_loss / num_samples
            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} | Tiempo: {elapsed:.2f}s | Loss: {avg_loss:.2f} | Recon Loss: {avg_recon_loss:.2f} | KL Loss: {avg_kl_loss:.2f}")

    def generate(self, n_samples=1):
        """Genera nuevas muestras desde el espacio latente (Punto 2c)."""
        generated_images = []
        for _ in range(n_samples):
            # Muestrea un z aleatorio de una N(0, I)
            z_sample = np.random.randn(self.latent_dim)
            # Pasa z solo por el decoder
            x_generated, _, _ = self._decoder_forward(z_sample)
            generated_images.append(x_generated.reshape(28, 28))  # Reformatea a 28x28
        return generated_images