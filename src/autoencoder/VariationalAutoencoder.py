import numpy as np
import time

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Evita overflow en np.exp
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_prime(a):
    return (a > 0).astype(float)


class VariationalAutoencoder:

    def __init__(self, input_dim, hidden_dims_encoder, latent_dim, hidden_dims_decoder, learning_rate=1e-3, beta=1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.beta = beta

        # Arquitectura del Encoder: genera [mu, log_var], por eso la salida es latent_dim * 2
        self.encoder_sizes = [input_dim] + hidden_dims_encoder + [latent_dim * 2]
        self.encoder_weights = []
        self.encoder_biases = []
        sizes = self.encoder_sizes

        for i in range(len(sizes) - 1):
            std_dev = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            self.encoder_weights.append(np.random.randn(sizes[i + 1], sizes[i]) * std_dev)
            self.encoder_biases.append(np.zeros(sizes[i + 1]))

        # Arqui del Decoder
        self.decoder_sizes = [latent_dim] + hidden_dims_decoder + [input_dim]
        self.decoder_weights = []
        self.decoder_biases = []

        sizes = self.decoder_sizes
        for i in range(len(sizes) - 1):
            std_dev = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            self.decoder_weights.append(np.random.randn(sizes[i + 1], sizes[i]) * std_dev)
            self.decoder_biases.append(np.zeros(sizes[i + 1]))

        # Activations
        self.hidden_activation = relu
        self.hidden_activation_prime = relu_prime
        self.output_activation = sigmoid  # Sigmoide para la salida (píxeles [0, 1])
        self.epsilon = None               # Ruido aleatorio usado en el reparameterization trick
        
        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon_adam = 1e-8
        
        # Initialize Adam momentum and velocity for encoder
        self.m_enc_w = [np.zeros_like(w) for w in self.encoder_weights]
        self.v_enc_w = [np.zeros_like(w) for w in self.encoder_weights]
        self.m_enc_b = [np.zeros_like(b) for b in self.encoder_biases]
        self.v_enc_b = [np.zeros_like(b) for b in self.encoder_biases]
        
        # Initialize Adam momentum and velocity for decoder
        self.m_dec_w = [np.zeros_like(w) for w in self.decoder_weights]
        self.v_dec_w = [np.zeros_like(w) for w in self.decoder_weights]
        self.m_dec_b = [np.zeros_like(b) for b in self.decoder_biases]
        self.v_dec_b = [np.zeros_like(b) for b in self.decoder_biases]
        
        self.t = 0  # Time step for Adam

    def _encoder_forward(self, x):
        activations = [x]
        zs = []         # Almacenar las entradas preactivación (z = Wx + b)
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
        activations.append(z_out)           # almacena la salida lineal
        # Dividir la salida en mu y log_var
        mu = z_out[:self.latent_dim]
        log_var = z_out[self.latent_dim:]

        return mu, log_var, activations, zs

    def _reparameterize(self, mu, log_var): # Reparameterization Trick (z = mu + epsilon * std)
        std = np.exp(0.5 * log_var)
        self.epsilon = np.random.randn(self.latent_dim)  # Almacena epsilon para backprop
        z = mu + self.epsilon * std

        return z

    def _decoder_forward(self, z):

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

    def forward(self, x):   # Forward completo del VAE
        mu, log_var, enc_activations, enc_zs = self._encoder_forward(x)
        z = self._reparameterize(mu, log_var)
        x_recon, dec_activations, dec_zs = self._decoder_forward(z)

        return x_recon, mu, log_var, z, enc_activations, enc_zs, dec_activations, dec_zs

    def _compute_loss(self, x, x_recon, mu, log_var):   # Calcula la pérdida total del VAE = Reconstruction_Loss + KL_Loss
        # Reconstruction Loss (Binary Cross-Entropy)
        epsilon = 1e-7  # para evitar log(0)!!!
        recon_loss = -np.sum(x * np.log(x_recon + epsilon) + (1 - x) * np.log(1 - x_recon + epsilon))

        # KL Loss (Divergencia KL)
        kl_loss = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def backward(self, x, x_recon, mu, log_var, z, enc_activations, enc_zs, dec_activations, dec_zs): # Backward completo (Retropropagación): calcula los gradientes para todos los pesos y biases

        # Gradientes para el Decoder
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

        # Gradientes para el Encoder
        grad_encoder_weights = [np.zeros_like(w) for w in self.encoder_weights]
        grad_encoder_biases = [np.zeros_like(b) for b in self.encoder_biases]

        # Gradientes de la Pérdida KL con respecto a mu y log_var
        dKL_dmu = self.beta * mu
        dKL_dlog_var = self.beta * (0.5 * (np.exp(log_var) - 1))

        # Gradientes del "Reparameterization Trick"
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


    def fit(self, X_train, epochs, batch_size=64, loss_threshold=None, patience=5):  # Entrena el VAE usando mini-batch gradient descent

        num_samples = X_train.shape[0]
        best_loss = float('inf')
        patience_counter = 0

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
                    # 1) Forward pass
                    x_recon, mu, log_var, z, enc_acts, enc_zs, dec_acts, dec_zs = self.forward(x)

                    # 2) Compute loss
                    loss, recon_loss, kl_loss = self._compute_loss(x, x_recon, mu, log_var)
                    batch_loss += loss
                    batch_recon_loss += recon_loss
                    batch_kl_loss += kl_loss

                    # 3) Backward pass (compute gradients)
                    g_ew, g_eb, g_dw, g_db = self.backward(x, x_recon, mu, log_var, z, enc_acts, enc_zs, dec_acts, dec_zs)

                    # 4) Acumular gradientes
                    for j in range(len(g_ew)):
                        sum_grad_enc_w[j] += g_ew[j]
                        sum_grad_enc_b[j] += g_eb[j]
                    for j in range(len(g_dw)):
                        sum_grad_dec_w[j] += g_dw[j]
                        sum_grad_dec_b[j] += g_db[j]

                # 5) Actualizar pesos usando Adam optimizer
                batch_len = len(batch)
                self.t += 1
                
                # Update encoder weights and biases with Adam
                for j in range(len(self.encoder_weights)):
                    grad_w = sum_grad_enc_w[j] / batch_len
                    grad_b = sum_grad_enc_b[j] / batch_len
                    
                    # Update biased first moment estimate
                    self.m_enc_w[j] = self.beta1 * self.m_enc_w[j] + (1 - self.beta1) * grad_w
                    self.m_enc_b[j] = self.beta1 * self.m_enc_b[j] + (1 - self.beta1) * grad_b
                    
                    # Update biased second raw moment estimate
                    self.v_enc_w[j] = self.beta2 * self.v_enc_w[j] + (1 - self.beta2) * (grad_w ** 2)
                    self.v_enc_b[j] = self.beta2 * self.v_enc_b[j] + (1 - self.beta2) * (grad_b ** 2)
                    
                    # Compute bias-corrected first moment estimate
                    m_hat_w = self.m_enc_w[j] / (1 - self.beta1 ** self.t)
                    m_hat_b = self.m_enc_b[j] / (1 - self.beta1 ** self.t)
                    
                    # Compute bias-corrected second raw moment estimate
                    v_hat_w = self.v_enc_w[j] / (1 - self.beta2 ** self.t)
                    v_hat_b = self.v_enc_b[j] / (1 - self.beta2 ** self.t)
                    
                    # Update parameters
                    self.encoder_weights[j] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon_adam)
                    self.encoder_biases[j] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon_adam)

                # Update decoder weights and biases with Adam
                for j in range(len(self.decoder_weights)):
                    grad_w = sum_grad_dec_w[j] / batch_len
                    grad_b = sum_grad_dec_b[j] / batch_len
                    
                    # Update biased first moment estimate
                    self.m_dec_w[j] = self.beta1 * self.m_dec_w[j] + (1 - self.beta1) * grad_w
                    self.m_dec_b[j] = self.beta1 * self.m_dec_b[j] + (1 - self.beta1) * grad_b
                    
                    # Update biased second raw moment estimate
                    self.v_dec_w[j] = self.beta2 * self.v_dec_w[j] + (1 - self.beta2) * (grad_w ** 2)
                    self.v_dec_b[j] = self.beta2 * self.v_dec_b[j] + (1 - self.beta2) * (grad_b ** 2)
                    
                    # Compute bias-corrected first moment estimate
                    m_hat_w = self.m_dec_w[j] / (1 - self.beta1 ** self.t)
                    m_hat_b = self.m_dec_b[j] / (1 - self.beta1 ** self.t)
                    
                    # Compute bias-corrected second raw moment estimate
                    v_hat_w = self.v_dec_w[j] / (1 - self.beta2 ** self.t)
                    v_hat_b = self.v_dec_b[j] / (1 - self.beta2 ** self.t)
                    
                    # Update parameters
                    self.decoder_weights[j] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon_adam)
                    self.decoder_biases[j] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon_adam)

                epoch_loss += batch_loss
                epoch_recon_loss += batch_recon_loss
                epoch_kl_loss += batch_kl_loss

            # Printtt: estadísticas de la epoch
            avg_loss = epoch_loss / num_samples
            avg_recon_loss = epoch_recon_loss / num_samples
            avg_kl_loss = epoch_kl_loss / num_samples
            
            # Normalize by number of pixels for easier interpretation
            per_pixel_loss = avg_loss / self.input_dim
            per_pixel_recon = avg_recon_loss / self.input_dim
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} | Tiempo: {elapsed:.2f}s | Loss: {avg_loss:.2f} ({per_pixel_loss:.4f}/pixel) | Recon Loss: {avg_recon_loss:.2f} ({per_pixel_recon:.4f}/pixel) | KL Loss: {avg_kl_loss:.2f}")
            
            # Early stopping based on loss threshold
            if loss_threshold is not None and avg_loss <= loss_threshold:
                print(f"\n✓ Reached loss threshold {loss_threshold:.2f}. Stopping early at epoch {epoch + 1}.")
                break
            
            # Early stopping based on patience (no improvement)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n✓ No improvement for {patience} epochs. Stopping early at epoch {epoch + 1}.")
                    break

    def generate(self, n_samples=1):    
        generated_images = []   # Nuevas muestras desde el espacio latente
        latent_coords = []

        for _ in range(n_samples):
            # Muestrea un z aleatorio de una N(0, I)
            z_sample = np.random.randn(self.latent_dim)
            # Pasa z solo por el decoder
            x_generated, _, _ = self._decoder_forward(z_sample)
            generated_images.append(x_generated.reshape(28, 28))
            latent_coords.append(z_sample)

        return generated_images, latent_coords
    
    def generate_from_latent(self, latent_points):
        """Generate images from specific latent coordinates"""
        generated_images = []
        
        for z in latent_points:
            x_generated, _, _ = self._decoder_forward(z)
            generated_images.append(x_generated.reshape(28, 28))
        
        return generated_images
    
    def encode(self, x):
        """Encode an image to latent space (returns mu only, no sampling)"""
        mu, log_var, _, _ = self._encoder_forward(x)
        return mu
    
    def reconstruct(self, x):
        """Reconstruct an image"""
        x_recon, _, _, _, _, _, _, _ = self.forward(x)
        return x_recon