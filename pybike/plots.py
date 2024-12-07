import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def plot_losses(loss_dict, save_path=None):
    """
    Creates a figure with 5 subplots displaying training and validation loss curves.
    The layout is arranged in 3 rows and 2 columns, with the last subplot centered.
    Subplot titles are larger and bolded for better visibility.

    Parameters:
    - loss_dict (defaultdict(list) or dict): Dictionary containing loss lists for different metrics.
    - save_path (str, optional): Path to save the image. If not provided, the figure is displayed.
    """

    metrics = {
        "start_station_loss": {
            "train": "train_loss_start_station",
            "val": "val_loss_start_station",
            "title": "Start Station Loss",
        },
        "end_station_loss": {
            "train": "train_loss_end_station",
            "val": "val_loss_end_station",
            "title": "End Station Loss",
        },
        "start_time_loss": {
            "train": "train_loss_start_time",
            "val": "val_loss_start_time",
            "title": "Start Time Loss",
        },
        "end_time_loss": {
            "train": "train_loss_end_time",
            "val": "val_loss_end_time",
            "title": "End Time Loss",
        },
        "total_loss": {
            "train": "train_total_loss",
            "val": "val_total_loss",
            "title": "Total Loss",
        },
    }

    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

    axes = []

    for idx, (metric_key, metric_info) in enumerate(metrics.items()):
        if idx < 4:
            ax = fig.add_subplot(gs[idx])
        else:
            ax = fig.add_subplot(gs[2, :])
        axes.append(ax)

        train_key = metric_info["train"]
        val_key = metric_info["val"]

        train_values = loss_dict.get(train_key, [])
        val_values = loss_dict.get(val_key, [])

        epochs = range(1, len(train_values) + 1)

        ax.plot(epochs, train_values, label="Training", marker="o")
        ax.plot(epochs, val_values, label="Validation", marker="x")

        ax.set_title(metric_info["title"], fontsize=16, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)

    fig.tight_layout()

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def plot_metrics(metrics_dict, save_path=None):
    """
    Creates a figure with 6 subplots displaying training and validation metric curves.
    The layout is arranged in 3 rows and 2 columns.
    Subplot titles are larger and bolded for better visibility.

    Parameters:
    - metrics_dict (defaultdict(list) or dict): Dictionary containing metric lists for different metrics.
    - save_path (str, optional): Path to save the image. If not provided, the figure is displayed.
    """

    metrics = {
        "start_station_acc": {
            "train": "train_start_station_acc",
            "val": "val_start_station_acc",
            "title": "Start Station Accuracy",
        },
        "end_station_acc": {
            "train": "train_end_station_acc",
            "val": "val_end_station_acc",
            "title": "End Station Accuracy",
        },
        "start_time_mae": {
            "train": "train_start_time_mae",
            "val": "val_start_time_mae",
            "title": "Start Time MAE",
        },
        "end_time_mae": {
            "train": "train_end_time_mae",
            "val": "val_end_time_mae",
            "title": "End Time MAE",
        },
        "start_time_r2": {
            "train": "train_start_time_r2",
            "val": "val_start_time_r2",
            "title": "Start Time R² Score",
        },
        "end_time_r2": {
            "train": "train_end_time_r2",
            "val": "val_end_time_r2",
            "title": "End Time R² Score",
        },
    }

    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    axes = []

    for idx, (metric_key, metric_info) in enumerate(metrics.items()):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        train_key = metric_info["train"]
        val_key = metric_info["val"]

        train_values = metrics_dict.get(train_key, [])
        val_values = metrics_dict.get(val_key, [])

        epochs = range(1, len(train_values) + 1)

        ax.plot(epochs, train_values, label="Training", marker="o")
        ax.plot(epochs, val_values, label="Validation", marker="x")

        ax.set_title(metric_info["title"], fontsize=16, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=14)

        if "acc" in metric_key:
            ax.set_ylabel("Accuracy", fontsize=14)
        elif "mae" in metric_key:
            ax.set_ylabel("MAE", fontsize=14)
        elif "r2" in metric_key:
            ax.set_ylabel("R² Score", fontsize=14)
        else:
            ax.set_ylabel("Metric", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
