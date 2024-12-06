import torch
from torch import Tensor


def calculate_accuracy(predictions: Tensor, targets: Tensor) -> float:
    """
    Calculate the accuracy of the predictions.

    Args:
        predictions (Tensor): The predicted values.
        targets (Tensor): The true values.

    Returns:
        float: The accuracy of the predictions.
    """
    _, predicted_classes = torch.max(predictions, 1)
    correct = (predicted_classes == targets.squeeze()).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()


def calculate_mae(predictions: Tensor, targets: Tensor) -> float:
    """
    Calculate the Mean Absolute Error (MAE) of the predictions.

    Args:
        predictions (Tensor): The predicted values.
        targets (Tensor): The true values.

    Returns:
        float: The MAE of the predictions.
    """
    return torch.mean(torch.abs(predictions - targets)).item()


def calculate_r2(predictions: Tensor, targets: Tensor) -> float:
    """
    Calculate the R-squared (coefficient of determination) of the predictions.

    Args:
        predictions (Tensor): The predicted values.
        targets (Tensor): The true values.

    Returns:
        float: The R-squared value of the predictions.
    """
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()
