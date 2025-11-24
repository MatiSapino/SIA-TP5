import time
import numpy as np
import os
import json
import pandas as pd

from src.plots.plot_utils import plot_mlp_error_vs_epochs, plot_mlp_predicted_vs_expected, get_run_output_path, compute_classification_metrics, plot_confusion_matrix, plot_metric_vs_epochs, collect_classification_metrics_over_epochs, save_run_config


ACTIVATIONS = {
    "sigmoid": (
        lambda z: 1 / (1 + np.exp(-2*z)),
        lambda a: 2*a * (1 - a) # Derivada de sigmoid
    ),
    "tanh": (
        lambda z: np.tanh(z),
        lambda a: 1 - a**2  # Derivada de tanh
    )
}


class MultiLayerPerceptron:


    def __init__(self, sizes : list[int], learning_rate: float, epochs: int,
                 error_threshold: float, training_data: list[tuple[np.ndarray, np.ndarray]], activation="sigmoid", mlp_task="mlp", optimizer="gradient_descent",
                 perceptron_type="MLP", task="mlp", config=None):
        """ Sizes es un arreglo que contiene el numero de neuronas por capa,
        incluyendo la capa de entrada y la capa de salida."""
        self.sizes = sizes
        self.layers = len(sizes)
        self.epochs = epochs
        self.error_threshold = error_threshold
        self.error = 1.0
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.outputs = [np.zeros(l) for l in sizes]
        self.training_data = training_data
        self.activation_name = activation
        self.activation, self.activation_prime = ACTIVATIONS[activation]
        self.mlp_task = mlp_task
        self.optimizer = optimizer.lower()
        self.perceptron_type = perceptron_type
        self.task = task
        self.config = config
        # Momentum parameters
        self.momentum_beta = 0.9
        self.velocities_w = [np.zeros_like(w) for w in self.weights]
        self.velocities_b = [np.zeros_like(b) for b in self.biases]
        # Adam parameters
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.adam_t = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.outputs[0] = x
        for l in range(1, self.layers):
            # busco obtener z=W⋅x+b con W la matriz de pesos, x la entrada y b el bias
            z = self.weights[l - 1] @ self.outputs[l - 1] + self.biases[l - 1]  # @ es como np.dot
            self.outputs[l] = self.activation(z)
        return self.outputs[-1]

    def backpropagation(self, y: np.ndarray):
        #array con lista de deltas de cada capa (arrancan en 0) menos la capa de input
        deltas = [np.zeros_like(n) for n in self.outputs[1:]]

        #calculo el delta de la capa de salida usando δ_L = (ζ - O) * g'(z)
        deltas[-1] = (y - self.outputs[-1]) * self.activation_prime(self.outputs[-1])

        #Ahora calculo los deltas para las capas ocultas arrancando desde la penultima capa
        # uso la formula δ_l = (W_(l+1)^T * δ_(l+1)) * g'(z_l)
        for l in range(2, self.layers):
            derivadas = self.activation_prime(self.outputs[-l])
            deltas[-l] = (self.weights[-l + 1].T @ deltas[-l + 1]) * derivadas

        # Actualizo pesos y biases segun el optimizador
        if self.optimizer == "gradient_descent":
            for l in range(self.layers - 1):
                self.weights[l] += self.learning_rate * np.outer(deltas[l], self.outputs[l])
                self.biases[l] += self.learning_rate * deltas[l]
        elif self.optimizer == "momentum":
            for l in range(self.layers - 1):
                grad_w = np.outer(deltas[l], self.outputs[l])
                grad_b = deltas[l]
                self.velocities_w[l] = self.momentum_beta * self.velocities_w[l] + self.learning_rate * grad_w
                self.velocities_b[l] = self.momentum_beta * self.velocities_b[l] + self.learning_rate * grad_b
                self.weights[l] += self.velocities_w[l]
                self.biases[l] += self.velocities_b[l]
        elif self.optimizer == "adam":
            self.adam_t += 1
            for l in range(self.layers - 1):
                grad_w = np.outer(deltas[l], self.outputs[l])
                grad_b = deltas[l]
                # Update first moment estimate
                self.m_w[l] = self.adam_beta1 * self.m_w[l] + (1 - self.adam_beta1) * grad_w
                self.m_b[l] = self.adam_beta1 * self.m_b[l] + (1 - self.adam_beta1) * grad_b
                # Update second moment estimate
                self.v_w[l] = self.adam_beta2 * self.v_w[l] + (1 - self.adam_beta2) * (grad_w ** 2)
                self.v_b[l] = self.adam_beta2 * self.v_b[l] + (1 - self.adam_beta2) * (grad_b ** 2)
                # Bias correction
                m_w_hat = self.m_w[l] / (1 - self.adam_beta1 ** self.adam_t)
                m_b_hat = self.m_b[l] / (1 - self.adam_beta1 ** self.adam_t)
                v_w_hat = self.v_w[l] / (1 - self.adam_beta2 ** self.adam_t)
                v_b_hat = self.v_b[l] / (1 - self.adam_beta2 ** self.adam_t)
                # Update weights and biases
                self.weights[l] += self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.adam_epsilon)
                self.biases[l] += self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.adam_epsilon)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")



    def train(self):
        start_time = time.time()
        e_condition_reached = False
        epoch_e_reached = 0
        error_history = []
        y_true_epochs = []
        y_pred_epochs = []
        # treat 'mnist' like 'digit' (multiclass classification)
        is_classification = (self.mlp_task in ["xor", "parity", "digit", "mnist"])
        # Always set output_path at the start for this run
        output_path = get_run_output_path(self.perceptron_type, self.task, self.activation_name, self.optimizer)
        for epoch in range(self.epochs):
            epoch_error = 0
            y_true = []
            y_pred = []

            total_incorrect_pixels = 0
            max_incorrect_pixels = 0

            for x, y in self.training_data:
                output = self.forward(x)
                epoch_error += 0.5 * np.sum((y - output) ** 2)
                self.backpropagation(y)

                if self.task == "autoencoder":
                    binarized_output = (output > 0.5).astype(int)
                    incorrect_count = np.sum(np.abs(binarized_output - y))
                    total_incorrect_pixels += incorrect_count
                    if incorrect_count > max_incorrect_pixels:
                        max_incorrect_pixels = (int)(incorrect_count)

                if is_classification:
                    # For multiclass tasks (digit, mnist): use argmax; for xor/parity: binary
                    if self.mlp_task in ("digit", "mnist"):
                        y_true.append(np.argmax(y))
                        y_pred.append(np.argmax(output))
                    else:
                        y_true.append(int(y[0]))
                        y_pred.append(int(np.round(output[0])))
            self.error = epoch_error
            error_history.append(self.error)
            if is_classification:
                y_true_epochs.append(np.array(y_true))
                y_pred_epochs.append(np.array(y_pred))
            # Print progress every epoch and on last epoch
            msg = f"Epoch {epoch+1}/{self.epochs} | Error: {self.error:.4f}"
            if is_classification and len(y_true) > 0:
                acc = np.mean(np.array(y_true) == np.array(y_pred))
                msg += f" | Accuracy: {acc:.4f}"

            if self.task == "autoencoder":
                avg_incorrect = total_incorrect_pixels / len(self.training_data)
                msg += f" | Avg_Incorrect_Pixels: {avg_incorrect:.2f} | Max_Incorrect_Pixels: {max_incorrect_pixels} | Self.error: {self.error:.2f} | Self.error_threshold: {self.error_threshold:.2f}"

            print(msg)

            if self.task == "autoencoder" and max_incorrect_pixels <= 1:
                e_condition_reached = True
                epoch_e_reached = epoch
                print(f"Target (Max Incorrect Pixels <= 1) reached at epoch {epoch+1}. Stopping training.")
                break

            if self.error <= self.error_threshold:
                e_condition_reached = True
                epoch_e_reached = epoch
                break
        self.__print_results(start_time, e_condition_reached, epoch_e_reached)
        plot_mlp_error_vs_epochs(error_history, self.perceptron_type, self.task, self.activation_name, self.optimizer, self.activation_name, self.config)
        self.save_results(error_history=error_history)

        if is_classification:
            plot_mlp_predicted_vs_expected(self.training_data, self, epoch,
                                       perceptron_type=self.perceptron_type,
                                       task=self.task,
                                       activation=self.activation_name,
                                       optimizer=self.optimizer,
                                       config=self.config)
            # Compute and plot metrics over epochs
            # use macro averaging for multiclass digit/mnist, binary otherwise
            avg = 'macro' if self.mlp_task in ("digit", "mnist") else 'binary'
            acc_hist, prec_hist, rec_hist, f1_hist = collect_classification_metrics_over_epochs(y_true_epochs, y_pred_epochs, average=avg)
            # All metric plots now go to the metrics subfolder (handled by plot_utils)
            plot_metric_vs_epochs(acc_hist, "accuracy", output_path)
            plot_metric_vs_epochs(prec_hist, "precision", output_path)
            plot_metric_vs_epochs(rec_hist, "recall", output_path)
            plot_metric_vs_epochs(f1_hist, "f1", output_path)
            # Plot confusion matrix for last epoch
            classes = list(range(10)) if self.mlp_task in ("digit", "mnist") else [0,1]
            metrics = compute_classification_metrics(y_true_epochs[-1], y_pred_epochs[-1], average=avg)
            plot_confusion_matrix(metrics['confusion_matrix'], classes=classes, output_path=output_path)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def evaluate_generalization(self, test_data: list[tuple[np.ndarray, np.ndarray]]):
        """
        Evaluate the model on a test set. For regression tasks return MSE (float).
        For classification tasks return computed metrics dict and also produce plots.
        """
        if not test_data:
            print("No test data provided for generalization evaluation.")
            return None

        is_classification = (self.mlp_task in ["xor", "parity", "digit", "mnist"])

        if is_classification:
            y_true = []
            y_pred = []
            for x, y in test_data:
                out = self.predict(x)
                # For multiclass tasks (digit, mnist) use argmax; for xor/parity treat as binary
                if self.mlp_task in ("digit", "mnist"):
                    y_true.append(np.argmax(y))
                    y_pred.append(np.argmax(out))
                else:
                    y_true.append(int(y[0]))
                    y_pred.append(int(np.round(out[0])))

            # compute metrics and plot
            output_path = get_run_output_path(self.perceptron_type, self.task, self.activation_name, self.optimizer)
            # use a subfolder for generalization/test outputs to separate from training plots
            gen_output_path = os.path.join(output_path, "generalization")
            os.makedirs(gen_output_path, exist_ok=True)
            # save config first (ensures folder exists)
            save_run_config(gen_output_path, self.config or {})
            avg = 'macro' if self.mlp_task in ("digit", "mnist") else 'binary'
            metrics = compute_classification_metrics(np.array(y_true), np.array(y_pred), average=avg)

            # save predicted vs expected for test set (created from test data)
            plot_mlp_predicted_vs_expected(test_data, self, epoch=None, perceptron_type=self.perceptron_type,
                                           task=self.task, activation=self.activation_name, optimizer=self.optimizer, config=self.config, output_path=gen_output_path)

            # Compute test error using the same loss used in training (0.5 * sum((y - out)^2) per sample)
            test_error_sum = 0.0
            for x, y in test_data:
                out = self.predict(x)
                test_error_sum += 0.5 * np.sum((y - out) ** 2)
            test_error = float(test_error_sum / len(test_data))

            # Diagnostic: print and capture distribution of true/pred labels for the test set
            try:
                unique_true, counts_true = np.unique(np.array(y_true), return_counts=True)
                unique_pred, counts_pred = np.unique(np.array(y_pred), return_counts=True)
                print("Generalization label distribution (y_true):", dict(zip(unique_true.tolist(), counts_true.tolist())))
                print("Generalization label distribution (y_pred):", dict(zip(unique_pred.tolist(), counts_pred.tolist())))
            except Exception:
                unique_true, counts_true, unique_pred, counts_pred = [], [], [], []

            # Plot confusion matrix for test set
            plot_confusion_matrix(metrics['confusion_matrix'], classes=list(range(10)) if self.mlp_task in ('digit', 'mnist') else [0,1], output_path=gen_output_path)

            # Save metrics and test error as JSON (serialize arrays/lists)
            try:
                metrics_serializable = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in metrics.items()}
                metrics_serializable['test_error'] = test_error
                # attach distribution info if available
                if len(unique_true) > 0:
                    metrics_serializable['y_true_distribution'] = {int(k): int(v) for k, v in zip(unique_true.tolist(), counts_true.tolist())}
                if len(unique_pred) > 0:
                    metrics_serializable['y_pred_distribution'] = {int(k): int(v) for k, v in zip(unique_pred.tolist(), counts_pred.tolist())}
                metrics_file = os.path.join(gen_output_path, "generalization_metrics.json")
                with open(metrics_file, 'w') as mf:
                    json.dump({**(self.config or {}), "generalization_metrics": metrics_serializable}, mf, indent=2)
            except Exception:
                pass
            return metrics
        else:
            # Regression: compute MSE
            squared_error_sum = 0.0
            for x, y in test_data:
                out = self.predict(x)
                squared_error_sum += np.sum((y - out) ** 2)
            mse = squared_error_sum / len(test_data)
            # plot predicted vs expected using existing utility
            output_path = get_run_output_path(self.perceptron_type, self.task, self.activation_name, self.optimizer)
            gen_output_path = os.path.join(output_path, "generalization")
            os.makedirs(gen_output_path, exist_ok=True)
            plot_mlp_predicted_vs_expected(test_data, self, epoch=None, perceptron_type=self.perceptron_type,
                                           task=self.task, activation=self.activation_name, optimizer=self.optimizer, config=self.config, output_path=gen_output_path)
            # also save mse in a json file
            try:
                metrics_file = os.path.join(gen_output_path, "generalization_metrics.json")
                with open(metrics_file, 'w') as mf:
                    json.dump({**(self.config or {}), "generalization_mse": float(mse)}, mf, indent=2)
            except Exception:
                pass
            return mse

    def __print_results(self, start_time, e_condition_reached, epoch_e_reached):
           print("\n--- Stop Condition Reached ---")
           if e_condition_reached:
               print(f"Executed Epochs {epoch_e_reached} : Convergence")
           else:
               print(f"Executed Epochs {self.epochs} : No Convergence")
           print(f"Total Time: {time.time() - start_time:.2f} seconds")
           print(f"Error: {self.error:.4f}")

    def save_results(self, error_history, results_file="./output/mlp_error_history.csv"):

        sizes_str = "[" + ",".join(map(str, self.sizes)) + "]"
        column_name = f"{self.perceptron_type}-lr:{self.learning_rate}-act:{self.activation_name}-opt:{self.optimizer}-sizes:{sizes_str}"

        df_new = pd.DataFrame({
            "epoch": list(range(len(error_history))),
            column_name: error_history
        })

        if os.path.exists(results_file):
            df_existing = pd.read_csv(results_file)

            same_prefix_cols = [col for col in df_existing.columns if col.startswith(column_name)]
            suffix = len(same_prefix_cols) + 1
            column_name_unique = f"{column_name}-col:{suffix}"

            df_new.rename(columns={column_name: column_name_unique}, inplace=True)
            df_merged = pd.merge(df_existing, df_new, on="epoch", how="outer").sort_values(by="epoch")
        else:
            column_name_unique = f"{column_name}-col:1"
            df_new.rename(columns={column_name: column_name_unique}, inplace=True)
            df_merged = df_new

        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        df_merged.to_csv(results_file, index=False)
