import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from pybike.plots import plot_losses


def setup_training(model, lr=0.0001, out_dir="./output"):
    """Setup training directory and return necessary components for training."""
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "log": os.path.join(out_dir, "log.json"),
        "model": os.path.join(out_dir, "model.pth"),
        "config": os.path.join(out_dir, "config.json"),
        "losses_plot": os.path.join(out_dir, "losses_plot.png"),
    }

    criterion = {"station": nn.CrossEntropyLoss(), "time": nn.MSELoss()}

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return paths, criterion, optimizer


def process_batch(model, batch, criterion, optimizer, is_training=True):
    """Process a single batch of data."""
    inputs, targets = batch

    # Unpack inputs and targets
    start_stations, start_coords, start_times, end_stations, end_coords, end_times = (
        inputs.values()
    )
    next_start_station, next_start_time, next_end_station, next_end_time = (
        targets.values()
    )

    if is_training:
        optimizer.zero_grad()

    # Forward pass
    start_station_pred, start_time_pred, end_station_pred, end_time_pred = model(
        start_stations, start_coords, start_times, end_stations, end_coords, end_times
    )

    # Calculate losses
    losses = {
        "start_station": criterion["station"](
            start_station_pred, next_start_station.squeeze()
        ),
        "start_time": criterion["time"](start_time_pred, next_start_time),
        "end_station": criterion["station"](
            end_station_pred, next_end_station.squeeze()
        ),
        "end_time": criterion["time"](end_time_pred, next_end_time),
    }

    total_loss = sum(losses.values())

    if is_training:
        total_loss.backward()
        optimizer.step()

    return {k: v.item() for k, v in losses.items()}


def train_epoch(model, dataloader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    epoch_losses = defaultdict(float)
    num_batches = 0

    for i, batch in enumerate(dataloader):
        batch_losses = process_batch(model, batch, criterion, optimizer)

        for k, v in batch_losses.items():
            epoch_losses[k] += v
        num_batches += 1

    return {k: v / num_batches for k, v in epoch_losses.items()}, num_batches


def validate(model, val_dataloader, criterion):
    """Perform validation."""
    model.eval()
    val_losses = defaultdict(float)
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            batch_losses = process_batch(
                model, batch, criterion, None, is_training=False
            )

            for k, v in batch_losses.items():
                val_losses[k] += v
            num_batches += 1

    return {k: v / num_batches for k, v in val_losses.items()}, num_batches


def save_training_artifacts(model, paths, loss_history):
    """Save model, config, and training history."""
    # Save model state
    torch.save(model.state_dict(), paths["model"])

    # Save model config
    config = {
        "n_stations": model.n_stations,
        "d_model": model.d_model,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
    }
    with open(paths["config"], "w") as f:
        json.dump(config, f)

    # Save loss history
    with open(paths["log"], "w") as f:
        json.dump(loss_history, f)

    # Generate loss plots
    plot_losses(loss_history, paths["losses_plot"])


def format_losses(train_losses, val_losses=None):
    """Format loss information for printing."""
    components = [
        ("S_Station", "start_station"),
        ("S_Time", "start_time"),
        ("E_Station", "end_station"),
        ("E_Time", "end_time"),
    ]

    info = []
    for name, key in components:
        train_val = f"{train_losses[key]:.4f}"
        if val_losses:
            train_val += f"/{val_losses[key]:.4f}"
        info.append(f"{name}_Loss (T/V): {train_val}")

    total_train = sum(train_losses.values())
    total_val = sum(val_losses.values()) if val_losses else None

    train_val = f"{total_train:.4f}"
    if total_val:
        train_val += f"/{total_val:.4f}"
    info.append(f"Total_Loss (T/V): {train_val}")

    return " | ".join(info)


def train(
    model,
    dataloader,
    val_dataloader=None,
    num_epochs=10,
    lr=0.0001,
    out_dir="./output",
    verbose=True,
):
    """Main training function."""
    paths, criterion, optimizer = setup_training(model, lr, out_dir)
    loss_history = defaultdict(list)

    for epoch in range(num_epochs):
        # Training phase
        train_losses, _ = train_epoch(model, dataloader, criterion, optimizer)

        # Record training losses
        for k, v in train_losses.items():
            loss_history[f"train_loss_{k}"].append(v)
        loss_history["train_total_loss"].append(sum(train_losses.values()))

        # Validation phase
        val_losses = None
        if val_dataloader is not None:
            val_losses, _ = validate(model, val_dataloader, criterion)

            # Record validation losses
            for k, v in val_losses.items():
                loss_history[f"val_loss_{k}"].append(v)
            loss_history["val_total_loss"].append(sum(val_losses.values()))

        # Print progress
        if verbose:
            train_info = format_losses(train_losses, val_losses)
            print(f"Epoch [{epoch+1}/{num_epochs}] {train_info}")

    save_training_artifacts(model, paths, loss_history)
    return loss_history
