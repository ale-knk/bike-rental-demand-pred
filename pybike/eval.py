from torch import nn, optim 

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

def validate(
    model: nn.Module, val_dataloader: torch.utils.data.DataLoader
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
        for batch in val_dataloader:
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
