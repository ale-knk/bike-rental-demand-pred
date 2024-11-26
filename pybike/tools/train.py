import os
import argparse

from pybike.dataloader import set_dataloaders
from pybike.train import setup_model, setup_optimizer, train
from pybike.preprocessing import CoordinateNormalizer
from pybike.config import TrainConfig


def main():
    """
    Main function to train a bike rental demand prediction model.

    This function performs the following steps:
    1. Parses command-line arguments to get the path to the configuration file.
    2. Loads the training configuration from a JSON file.
    3. Sets the output directory for saving the model and other artifacts.
    4. Initializes the model and optimizer based on the configuration.
    5. Prepares data loaders for training and validation.
    6. Trains the model.
    7. Saves the training configuration and scaler.

    Command-line arguments:
    --config_path, -c: Path to the configuration file (str).
    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a bike rental demand prediction model."
    )
    parser.add_argument(
        "--config_path", "-c", type=str, help="Path to the configuration file."
    )
    args = parser.parse_args()

    # Load training configuration from JSON file
    train_config = TrainConfig.from_json(args.config_path)

    # Set output directory for saving model and other artifacts
    out_dir = train_config.out_dir

    # Create out dir if necessary
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize model and optimizer based on configuration
    model = setup_model(train_config.config_model)
    optimizer = setup_optimizer(model, train_config.config_optimizer)
    scaler = CoordinateNormalizer()

    # Prepare data loaders for training and validation
    train_dataloader, val_dataloader = set_dataloaders(
        splits=["train", "val"],
        scaler=scaler,
        **train_config.config_dataloader.model_dump(),
    )

    # Train the model
    train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=5,
        out_dir=out_dir,
    )

    # Save the training configuration and scaler
    out_config_path = os.path.join(out_dir, "train_config.json")
    train_config.save(out_config_path)
    scaler.save(train_config.scaler_path)


if __name__ == "__main__":
    main()
