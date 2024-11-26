import torch
import argparse
import os

from pybike.config import EvalConfig
from pybike.dataloader import set_dataloaders
from pybike.preprocessing import CoordinateNormalizer
from pybike.train import setup_criterion, setup_model, validate
from pybike.utils import save_dict_to_json


def main():
    """
    Main function to evaluate a bike rental demand prediction model.
    This function parses the configuration file, sets up the model and dataloaders,
    performs evaluation, and saves the evaluation metrics and configuration.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a bike rental demand prediction model."
    )
    parser.add_argument(
        "--config_path", "-c", type=str, help="Path to the configuration file."
    )
    args = parser.parse_args()

    # Load evaluation configuration from JSON file
    eval_config = EvalConfig.from_json(args.config_path)

    # Create output directory if it doesn't exist
    out_dir = eval_config.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load the coordinate normalizer
    scaler = CoordinateNormalizer()
    scaler.load(eval_config.config_train.scaler_path)

    # Setup the model and load its weights
    model = setup_model(eval_config.config_train.config_model)
    model.load_state_dict(
        torch.load(eval_config.config_train.model_path, weights_only=True)
    )
    model.eval()

    # Setup the loss criterion
    criterion = setup_criterion()

    # Setup the test dataloader
    test_dataloader = set_dataloaders(
        splits=["test"], scaler=scaler, **eval_config.config_dataloader.model_dump()
    )

    # Validate the model on the test dataset
    _, metrics = validate(model, test_dataloader, criterion)

    # Save the evaluation configuration and metrics to the output directory
    out_config_path = os.path.join(out_dir, "eval_config.json")
    eval_config.save(out_config_path)
    save_dict_to_json(metrics, os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
