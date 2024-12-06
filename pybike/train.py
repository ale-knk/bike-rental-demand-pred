import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from pybike.config import TrainConfig, OptimizerConfig, ModelConfig
from pybike.metrics import calculate_accuracy, calculate_mae, calculate_r2
from pybike.model import BikeRoutePredictor
from pybike.plots import plot_losses, plot_metrics
from pybike.utils import save_dict_to_json

def setup_model(model_config: ModelConfig) -> BikeRoutePredictor:
    """
    Initialize the BikeRoutePredictor model with the given parameters.

    Args:
        model_params (dict): Dictionary containing model parameters.

    Returns:
        BikeRoutePredictor: An instance of the BikeRoutePredictor model.
    """
    model = BikeRoutePredictor(**model_config.model_dump())
    return model


def setup_optimizer(
    model: nn.Module, optimizer_config: OptimizerConfig, optimizer_type: str = "adam"
) -> optim.Optimizer:
    """
    Initialize the optimizer for the given model.

    Args:
        model (nn.Module): The model to optimize.
        optimizer_params (dict): Dictionary containing optimizer parameters.
        optimizer_type (str): Type of optimizer to use ("adam", "sgd", "rmsprop").

    Returns:
        optim.Optimizer: An instance of the optimizer.

    Raises:
        ValueError: If an unsupported optimizer type is provided.
    """
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_config.model_dump())
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), **optimizer_config.model_dump())
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), **optimizer_config.model_dump())
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    return optimizer

def setup_criterion() -> dict[str, nn.Module]:
    
    criterion = {
        "station": nn.CrossEntropyLoss(),
        "time": nn.MSELoss()
    }
    return criterion

def process_batch(
    model: nn.Module,
    batch: tuple[dict, dict],
    criterion: dict,
    optimizer: optim.Optimizer | None = None,
    is_training: bool = True,
) -> tuple[dict, dict]:
    """
    Process a single batch of data.

    Args:
        model (nn.Module): The model to train or evaluate.
        batch (tuple[dict, dict]): A tuple containing inputs and targets.
        criterion (dict): Dictionary containing loss functions.
        optimizer (optim.Optimizer, optional): The optimizer for training. Defaults to None.
        is_training (bool, optional): Flag indicating if the model is in training mode. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - losses (dict): Dictionary with calculated losses.
            - metrics (dict): Dictionary with calculated metrics.
    """
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
    losses = {k: v.item() for k, v in losses.items()}

    metrics = {
        "start_station_acc": calculate_accuracy(start_station_pred, next_start_station),
        "end_station_acc": calculate_accuracy(end_station_pred, next_end_station),
        "start_time_mae": calculate_mae(start_time_pred, next_start_time),
        "end_time_mae": calculate_mae(end_time_pred, next_end_time),
        "start_time_r2": calculate_r2(start_time_pred, next_start_time),
        "end_time_r2": calculate_r2(end_time_pred, next_end_time),
    }

    if is_training:
        total_loss.backward()
        optimizer.step()

    return losses, metrics


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: dict,
    optimizer: optim.Optimizer,
) -> tuple[dict, dict, int]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (dict): Dictionary containing loss functions.
        optimizer (optim.Optimizer): The optimizer for training.

    Returns:
        tuple: A tuple containing:
            - epoch_losses (dict): Dictionary with average losses for the epoch.
            - epoch_metrics (dict): Dictionary with average metrics for the epoch.
            - num_batches (int): Number of batches processed.
    """
    model.train()
    epoch_losses = defaultdict(float)
    epoch_metrics = defaultdict(float)

    num_batches = 0

    for i, batch in enumerate(dataloader):
        batch_losses, batch_metrics = process_batch(model, batch, criterion, optimizer)

        for k, v in batch_losses.items():
            epoch_losses[k] += v
        for k, v in batch_metrics.items():
            epoch_metrics[k] += v
        num_batches += 1

    epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

    return epoch_losses, epoch_metrics, num_batches


def validate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: dict
) -> tuple[dict, dict]:
    """
    Perform validation on the model.

    Args:
        model (nn.Module): The model to validate.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (dict): Dictionary containing loss functions.

    Returns:
        tuple: A tuple containing:
            - val_losses (dict): Dictionary with average validation losses.
            - val_metrics (dict): Dictionary with average validation metrics.
    """
    model.eval()
    val_losses = defaultdict(float)
    val_metrics = defaultdict(float)

    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_losses, batch_metrics = process_batch(
                model, batch, criterion, None, is_training=False
            )

            for k, v in batch_losses.items():
                val_losses[k] += v

            for k, v in batch_metrics.items():
                val_metrics[k] += v

            num_batches += 1

    val_losses = {k: v / num_batches for k, v in val_losses.items()}
    val_metrics = {k: v / num_batches for k, v in val_metrics.items()}

    return val_losses, val_metrics


def save_training_artifacts(
    model: nn.Module,
    optimizer: optim.Optimizer,
    out_dir: str,
    loss_history: list,
    metrics_history: list,
) -> None:
    """
    Save model, config, and training history.

    Args:
        model (nn.Module): The trained model.
        optimizer (optim.Optimizer): The optimizer used for training.
        paths (dict): Dictionary with paths for saving logs, model, optimizer, etc.
        loss_history (list): List of loss values over epochs.
        metrics_history (list): List of metric values over epochs.
    """
    # Save model state
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pth"))

    # Save loss history
    with open(os.path.join(out_dir, "log.log"), "w") as f:
        json.dump(loss_history, f)

    # Generate loss plots
    plot_losses(loss_history, os.path.join(out_dir, "losses.png"))
    plot_metrics(metrics_history, os.path.join(out_dir, "metrics.png"))

    # Save loss and metrics jsons
    save_dict_to_json(loss_history, os.path.join(out_dir, "losses.json"))
    save_dict_to_json(metrics_history, os.path.join(out_dir, "metrics.json"))


def format_losses(train_losses: dict, val_losses: dict | None = None) -> str:
    """
    Format loss information for printing.

    Args:
        train_losses (dict): Dictionary with training losses.
        val_losses (dict, optional): Dictionary with validation losses. Defaults to None.

    Returns:
        str: Formatted string with loss information.
    """
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
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader | None = None,
    num_epochs: int = 10,
    out_dir: str = "./outputs",
    verbose=True,
):
    """
    Train the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        optimizer (optim.Optimizer): The optimizer for training.
        num_epochs (int): Number of epochs to train the model.
        out_dir (str): Output directory for saving training artifacts.
        verbose (bool): Flag to print training progress. Defaults to True.
    """

    
    criterion = setup_criterion()
    loss_history = defaultdict(list)
    metrics_history = defaultdict(list)

    for epoch in range(num_epochs):
        # Training phase
        train_losses, train_metrics, _ = train_epoch(
            model, train_dataloader, criterion, optimizer
        )

        # Record training losses
        for k, v in train_losses.items():
            loss_history[f"train_loss_{k}"].append(v)
        loss_history["train_total_loss"].append(sum(train_losses.values()))

        for k, v in train_metrics.items():
            metrics_history[f"train_{k}"].append(v)

        # Validation phase
        val_losses = None
        if val_dataloader is not None:
            val_losses, val_metrics = validate(model, val_dataloader, criterion)

            # Record validation losses
            for k, v in val_losses.items():
                loss_history[f"val_loss_{k}"].append(v)
            loss_history["val_total_loss"].append(sum(val_losses.values()))

            for k, v in val_metrics.items():
                metrics_history[f"val_{k}"].append(v)

        # Print progress
        if verbose:
            train_info = format_losses(train_losses, val_losses)
            print(f"Epoch [{epoch+1}/{num_epochs}] {train_info}")

    save_training_artifacts(model, optimizer, out_dir, loss_history, metrics_history)
    return loss_history, metrics_history
