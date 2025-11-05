import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import re
import ast

def compute_classification_metrics(y_true, y_pred, average='binary'):
    # y_true and y_pred should be 1D arrays of labels
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return metrics

def get_metrics_output_path(output_path):
    metrics_path = os.path.join(output_path, "metrics")
    os.makedirs(metrics_path, exist_ok=True)
    return metrics_path

def plot_confusion_matrix(cm, classes, output_path, title="Confusion Matrix"):
    metrics_path = get_metrics_output_path(output_path)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_path, "confusion_matrix.png"))
    plt.close()

def plot_metric_vs_epochs(metric_history, metric_name, output_path):
    metrics_path = get_metrics_output_path(output_path)
    plt.figure(figsize=(7, 5))
    plt.plot(range(len(metric_history)), metric_history, marker='o', linestyle='-', color='blue')
    plt.title(f"{metric_name.title()} vs Epochs")
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.title())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_path, f"{metric_name}_vs_epochs.png"))
    plt.close()
    
def collect_classification_metrics_over_epochs(y_true_epochs, y_pred_epochs, average='binary'):
    # y_true_epochs and y_pred_epochs are lists of label arrays for each epoch
    acc_hist, prec_hist, rec_hist, f1_hist = [], [], [], []
    for y_true, y_pred in zip(y_true_epochs, y_pred_epochs):
        metrics = compute_classification_metrics(y_true, y_pred, average=average)
        acc_hist.append(metrics['accuracy'])
        prec_hist.append(metrics['precision'])
        rec_hist.append(metrics['recall'])
        f1_hist.append(metrics['f1'])
    return acc_hist, prec_hist, rec_hist, f1_hist


def get_run_output_path(perceptron_type, task, activation, optimizer):
    output_dir = "./output"
    # For regression perceptrons, avoid repeating perceptron_type and remove optimizer from folder name
    regression_types = [
        "Simple Linear Perceptron",
        "Simple Non-Linear Tanh Perceptron",
        "Simple Non-Linear Logistics Perceptron"
    ]
    if perceptron_type in regression_types:
        run_folder = f"{activation}"
        output_path = os.path.join(output_dir, perceptron_type, run_folder)
    else:
        run_folder = f"{task}_{activation}_{optimizer}"
        output_path = os.path.join(output_dir, perceptron_type, run_folder)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def save_run_config(output_path, config):
    config_file = os.path.join(output_path, "config.json")
    # ensure output folder exists
    os.makedirs(output_path, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def plot_simple_step_perceptron(elements, weights, epoch, function_name, error, perceptron_type, task, activation, optimizer, config):
    output_path = get_run_output_path(perceptron_type, task, activation, optimizer)
    save_run_config(output_path, config)
    x1 = [e.get_entries()[1] for e in elements]
    x2 = [e.get_entries()[2] for e in elements]
    y_expected = [e.get_expected_value() for e in elements]
    plt.figure(figsize=(6, 6))
    colors = ['red' if y == -1 else 'blue' for y in y_expected]
    plt.scatter(x1, x2, c=colors, s=100)
    plt.scatter([], [], c='red', s=100, label='Output -1')
    plt.scatter([], [], c='blue', s=100, label='Output 1')
    w0, w1, w2 = weights[0], weights[1], weights[2]
    padding = 0.5
    x_min, x_max = min(x1) - padding, max(x1) + padding
    y_min, y_max = min(x2) - padding, max(x2) + padding
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.suptitle(f'{perceptron_type} - {task} - {activation} - {optimizer}')
    plt.title(f'Epoch {epoch}')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if w2 != 0:
        x_plot = np.array(plt.xlim())
        y_plot = (-w0 - w1 * x_plot) / w2
        plt.plot(x_plot, y_plot, 'k--', label='Decision Boundary')
    else:
        x_line_val = -w0 / w1 if w1 != 0 else 0
        plt.axvline(x=x_line_val, color='k', linestyle='--', label='Decision Boundary')
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.text(0.8, 0.95, f'Error: {error:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=1, lw=1.5))
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_path, f"epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()

def plot_simple_regression_perceptron(elements, epoch, error, model_type, activation_function, task, activation, optimizer, config):
    # Unify perceptron_type naming for all outputs
    if model_type == "simple_linear":
        perceptron_type = "Simple Linear Perceptron"
    elif model_type == "simple_no_linear_tanh":
        perceptron_type = "Simple Non-Linear Tanh Perceptron"
    elif model_type == "simple_no_linear_logistics":
        perceptron_type = "Simple Non-Linear Logistics Perceptron"
    else:
        perceptron_type = model_type.replace("_", " ").title()
    output_path = get_run_output_path(perceptron_type, task, activation, None)
    save_run_config(output_path, config)

    y_expected_list = []
    z_predicted_list = []

    for e in elements:
        entries = np.array(e.get_entries(), dtype=float)
        y_expected_list.append(e.get_expected_value())

        z_predicted = activation_function(entries)
        z_predicted_list.append(z_predicted)

    plt.figure(figsize=(7, 7))

    plt.scatter(y_expected_list, z_predicted_list, c='blue', s=80, label='Input Points')

    min_val = min(min(y_expected_list), min(z_predicted_list))
    max_val = max(max(y_expected_list), max(z_predicted_list))

    range_val = max_val - min_val
    padding = range_val * 0.1 if range_val > 0 else 5.0

    ideal_range = np.linspace(min_val - padding, max_val + padding, 100)

    plt.xlim(ideal_range.min(), ideal_range.max())
    plt.ylim(ideal_range.min(), ideal_range.max())

    plt.plot(ideal_range, ideal_range, 'r--', label='Regression (Y = Z)')

    plt.xlabel('Expected Output (Y)')
    plt.ylabel('Predicted Output (Z)')
    # Simpler, non-repetitive title for regression perceptrons
    plt.suptitle(f'{perceptron_type}')
    if epoch is not None:
        plt.title(f'Epoch {epoch}')
    plt.text(0.95, 0.05, f'MSE: {error:.4f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=1))
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()

    if epoch is None:
        filename = os.path.join(output_path, f"generalization.png")
    else:
        filename = os.path.join(output_path, f"epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()

def plot_projections_simple_regression_perceptron(perceptron, elements, epoch, error, model_type, activation_function):
    # Unify perceptron_type naming and projections folder
    if model_type == "simple_linear":
        perceptron_type = "Simple Linear Perceptron"
    elif model_type == "simple_no_linear_tanh":
        perceptron_type = "Simple Non-Linear Tanh Perceptron"
    elif model_type == "simple_no_linear_logistics":
        perceptron_type = "Simple Non-Linear Logistics Perceptron"
    else:
        perceptron_type = model_type.replace("_", " ").title()
    # Projections go in a subfolder of the current run folder (same as other outputs)
    # Find the latest run folder for this perceptron_type (by mtime)
    base_dir = os.path.join("./output", perceptron_type)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    run_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if run_folders:
        # Use the most recently modified run folder
        run_folders.sort(key=lambda f: os.path.getmtime(os.path.join(base_dir, f)), reverse=True)
        run_folder = run_folders[0]
    else:
        # Fallback: create a default run folder
        run_folder = "default_run"
        os.makedirs(os.path.join(base_dir, run_folder), exist_ok=True)
    output_path = os.path.join(base_dir, run_folder, "projections")
    os.makedirs(output_path, exist_ok=True)
    model_name = perceptron_type

    num_inputs = len(perceptron.weights) - 1

    all_x = np.array([e.get_entries() for e in elements])
    y_expected = np.array([e.get_expected_value() for e in elements])

    weights = perceptron.weights

    fig, axes = plt.subplots(1, num_inputs, figsize=(5 * num_inputs, 6))
    fig.suptitle(f'{model_name}', fontsize=14)
    if epoch is not None:
        fig.text(0.5, 0.92, f'Epoch {epoch}', ha='center', fontsize=14, transform=fig.transFigure)

    if num_inputs == 1:
        axes = [axes]

    x_means = np.mean(all_x, axis=0)
    mean_entries = x_means.copy()

    for i in range(1, num_inputs + 1):
        ax = axes[i - 1]
        x_i = all_x[:, i]
        w_i = weights[i]

        ax.scatter(x_i, y_expected, c='blue', alpha=0.7, label='Input Points')

        x_line = np.linspace(x_i.min() - 0.5, x_i.max() + 0.5, 100)
        y_reg_line = np.zeros_like(x_line)

        if model_type == "simple_linear":
            intercept_correction = np.dot(weights, x_means) - (w_i * x_means[i])
            y_reg_line = w_i * x_line + intercept_correction
            plot_label = f'Regression (W{i}={w_i:.4f})'
            line_style = 'r-'
        else:
            tita = weights[0] * mean_entries[0]

            for j in range(1, num_inputs + 1):
                if j != i:
                    tita += weights[j] * mean_entries[j]

            simulated_entries = np.tile(mean_entries, (len(x_line), 1))
            simulated_entries[:, i] = x_line

            for k, sim_entry in enumerate(simulated_entries):
                y_reg_line[k] = activation_function(sim_entry)

            plot_label = f'Regression (W{i}={w_i:.4f})'
            line_style = 'r-'

        ax.plot(x_line, y_reg_line, line_style, linewidth=2, label=plot_label)

        ax.set_xlabel(f'X{i}')
        if model_type == "simple_linear":
            ax.set_ylabel('Y')
        else:
            ax.set_ylabel('Y (Normalized Output)')
        ax.set_title(f'Projection X{i}')
        if model_type == "simple_no_linear_logistics":
            ax.set_ylim(-0.1, 1.1)
        elif model_type == "simple_no_linear_tanh":
            ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.text(0.85, 0.92, f'MSE: {error:.4f}', transform=fig.transFigure, fontsize=12, verticalalignment='bottom', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=1))
    plt.subplots_adjust(top=0.85, bottom=0.20, wspace=0.3)

    if epoch is not None:
        filename = os.path.join(output_path, f"epoch_{epoch}.png")
    else:
        filename = os.path.join(output_path, f"generalization.png")
    plt.savefig(filename)
    plt.close()

def plot_error_vs_epochs(error_history, perceptron_type, task, activation, optimizer, title, ylabel, config):
    # Unify perceptron_type naming for error_vs_epochs as well
    pt = perceptron_type
    regression_types = [
        "Simple Linear Perceptron",
        "Simple Non-Linear Tanh Perceptron",
        "Simple Non-Linear Logistics Perceptron"
    ]
    if pt.lower() == "simple linear" or pt.lower() == "simple_linear":
        pt = "Simple Linear Perceptron"
    elif pt.lower() == "simple no linear tanh" or pt.lower() == "simple_no_linear_tanh":
        pt = "Simple Non-Linear Tanh Perceptron"
    elif pt.lower() == "simple no linear logistics" or pt.lower() == "simple_no_linear_logistics":
        pt = "Simple Non-Linear Logistics Perceptron"
    # For regression, remove optimizer from folder name
    if pt in regression_types:
        output_path = get_run_output_path(pt, task, activation, None)
        plot_title = f"{pt}"
    else:
        output_path = get_run_output_path(pt, task, activation, optimizer)
        plot_title = title
    save_run_config(output_path, config)
    plt.figure(figsize=(7, 5))
    plt.plot(range(len(error_history)), error_history, marker='o', linestyle='-', color='blue')
    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_path, "error_vs_epochs.png")
    plt.savefig(filename)
    plt.close()

def plot_mlp_error_vs_epochs(error_history, perceptron_type, task, activation, optimizer, activation_name, config):
    title = f'{perceptron_type} - {task} - {activation} - {optimizer} - {activation_name}'
    plot_error_vs_epochs(error_history, perceptron_type, task, activation, optimizer, title, "Error", config)

def plot_mlp_predicted_vs_expected(training_data, mlp, epoch=None, perceptron_type="MLP", task="mlp", activation="sigmoid", optimizer="gradient_descent", config=None, output_path: str = None):
    # Allow caller to override the output path (useful for separating training vs test/generalization plots)
    if output_path is None:
        output_path = get_run_output_path(perceptron_type, task, activation, optimizer)
    if config is not None:
        save_run_config(output_path, config)
    y_expected_list = []
    z_predicted_list = []
    for x, y in training_data:
        y_expected = y
        z_predicted = mlp.predict(x)
        # For classification, use argmax; for regression, use value
        if len(y_expected.shape) > 0 and y_expected.shape[0] > 1:
            y_expected_list.append(np.argmax(y_expected))
            z_predicted_list.append(np.argmax(z_predicted))
        else:
            y_expected_list.append(float(y_expected[0]))
            z_predicted_list.append(float(z_predicted[0]))
    plt.figure(figsize=(7, 7))
    plt.scatter(y_expected_list, z_predicted_list, c='blue', s=80, label='Predicted vs Expected')
    min_val = min(min(y_expected_list), min(z_predicted_list))
    max_val = max(max(y_expected_list), max(z_predicted_list))
    range_val = max_val - min_val
    padding = range_val * 0.1 if range_val > 0 else 5.0
    ideal_range = np.linspace(min_val - padding, max_val + padding, 100)
    plt.xlim(ideal_range.min(), ideal_range.max())
    plt.ylim(ideal_range.min(), ideal_range.max())
    plt.plot(ideal_range, ideal_range, 'r--', label='Ideal (Y = Z)')
    plt.xlabel('Expected Output (Y)')
    plt.ylabel('Predicted Output (Z)')
    activation_name = getattr(mlp, 'activation_name', activation)
    plt.suptitle(f'{perceptron_type} - {task} - {activation} - {optimizer}')
    if epoch is not None:
        plt.title(f'Epoch {epoch}')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    filename = os.path.join(output_path, f"epoch_{epoch if epoch is not None else 'final'}.png")
    plt.savefig(filename)
    plt.close()

def plot_dataset(df):

    cols = [col for col in df.columns if col != 'y']
    if len(cols) == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]],
                             c=df['y'], cmap='viridis', s=60)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_zlabel(cols[2])
        fig.colorbar(scatter, ax=ax, label='y')
        plt.title("Dataset visualization")
        plt.show()

def plot_all_runs(results_file="./output/error_history.csv", title="", model_prefix=None):

    if not os.path.exists(results_file):
        print(f"File not found: {results_file}")
        return

    df = pd.read_csv(results_file)
    epoch = df["epoch"]

    if model_prefix:
        cols = [col for col in df.columns if col.startswith(model_prefix)]
    else:
        cols = [col for col in df.columns if col != "epoch"]

    if not cols:
        print(f"Prefix not found: {model_prefix}")
        return

    plt.figure(figsize=(10, 6))
    for col in cols:
        parts = col.split("-")
        if model_prefix:
            perceptron_name = "" if parts[0] == model_prefix else parts[0]
        else:
            perceptron_name = parts[0]

        lr = None
        beta = None
        for p in parts[1:]:
            if p.startswith("lr:"):
                lr = p.split(":")[1]
            elif p.startswith("b:"):
                beta = p.split(":")[1]

        legend_name = perceptron_name.strip()
        if lr:
            legend_name += f"  LR: {lr}"
        if beta:
            legend_name += f"  Beta: {beta}"
        legend_name = legend_name.strip()

        plt.plot(epoch, df[col], label=legend_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_average_runs(csv_path="./output/mlp_error_history.csv", title="MLP", log_scale=True):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        print("Missing 'epoch' column.")
        return

    # Agrupar por prefijo (todo antes de "-col:X")
    pattern = re.compile(r'^(.*)-col:\d+$')
    groups = {}

    for col in df.columns:
        if col == "epoch":
            continue
        match = pattern.match(col)
        if match:
            prefix = match.group(1)
            groups.setdefault(prefix, []).append(col)

    if not groups:
        print("No matching columns found.")
        return

    plt.figure(figsize=(10, 6))

    for prefix, cols in groups.items():
        subset = df[cols].copy()

        # Rellenar valores faltantes con forward-fill y luego último valor
        valid_cols = []
        for c in cols:
            series = subset[c].copy()
            if series.notna().any():
                last_valid = series.dropna().iloc[-1]
                series = series.ffill().fillna(last_valid)
                valid_cols.append(series)
        if not valid_cols:
            continue

        subset = pd.concat(valid_cols, axis=1)
        avg = subset.mean(axis=1)
        std = subset.std(axis=1)
        std = np.clip(std, 0, avg * 0.5)

        lr_match = re.search(r'lr:([\d\.eE]+)', prefix)
        lr = lr_match.group(1) if lr_match else "?"

        act_match = re.search(r'act:([a-zA-Z0-9_]+)', prefix)
        act = act_match.group(1) if act_match else "?"

        opt_match = re.search(r'opt:([a-zA-Z0-9_]+)', prefix)
        opt = opt_match.group(1) if opt_match else "?"

        sizes_match = re.search(r'sizes:(\[[^\]]+\])', prefix)
        try:
            sizes = ast.literal_eval(sizes_match.group(1)) if sizes_match else []
        except Exception:
            sizes = []

        n_samples = len(valid_cols)
        sizes_str = f"[{','.join(map(str, sizes))}]" if sizes else "?"

        label = f"LR:{lr}, act:{act}, opt:{opt}, sizes:{sizes_str}  (n={n_samples})"

        plt.plot(df["epoch"], avg, label=label, linewidth=2)
        plt.fill_between(df["epoch"], avg - std, avg + std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    if log_scale:
        plt.yscale("log")

    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
