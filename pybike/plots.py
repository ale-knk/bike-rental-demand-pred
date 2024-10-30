import matplotlib.pyplot as plt
from collections import defaultdict
import os
import matplotlib.gridspec as gridspec


def plot_losses(loss_dict, save_path=None):
    """
    Creates a figure with 5 subplots displaying training and validation loss curves.
    The layout is arranged in 3 rows and 2 columns, with the last subplot centered.
    Subplot titles are larger and bolded for better visibility.

    Parameters:
    - loss_dict (defaultdict(list) or dict): Dictionary containing loss lists for different metrics.
    - save_path (str, optional): Path to save the image. If not provided, the figure is displayed.
    """

    # Define the metrics and their corresponding keys in the loss dictionary
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

    num_metrics = len(metrics)

    # Create a figure with increased height to accommodate larger titles
    fig = plt.figure(figsize=(15, 18))

    # Define GridSpec with 3 rows and 2 columns
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

    # Initialize a list to hold the axes
    axes = []

    for idx, (metric_key, metric_info) in enumerate(metrics.items()):
        if idx < 4:
            ax = fig.add_subplot(gs[idx])
        else:
            # For the last subplot, span both columns to center it
            ax = fig.add_subplot(gs[2, :])
        axes.append(ax)

        # Retrieve training and validation loss values
        train_key = metric_info["train"]
        val_key = metric_info["val"]

        train_values = loss_dict.get(train_key, [])
        val_values = loss_dict.get(val_key, [])

        epochs = range(1, len(train_values) + 1)

        # Plot training and validation losses
        ax.plot(epochs, train_values, label="Training", marker="o")
        ax.plot(epochs, val_values, label="Validation", marker="x")

        # Set plot title and labels with enhanced font size and bold weight
        ax.set_title(metric_info["title"], fontsize=16, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)

    # Adjust layout for better spacing
    fig.tight_layout()

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved at {save_path}")
    else:
        plt.show()
