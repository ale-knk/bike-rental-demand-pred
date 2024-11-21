import argparse
from pybike.utils import read_json_to_dict
from pybike.train import setup_model, setup_optimizer, train
from pybike.dataloader import set_dataloaders

def main():
    parser = argparse.ArgumentParser(
        description="Train a bike rental demand prediction model."
    )
    parser.add_argument("--config_path", "-c", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    config = read_json_to_dict(args.config_path)

    model = setup_model(config["model"])
    optimizer = setup_optimizer(model, config["optimizer"])
    train_dataloader, val_dataloader = set_dataloaders(
        splits=["train", "val"], **config["dataloader"]
    )
    train(
        model, 
        optimizer, 
        train_dataloader, 
        val_dataloader=val_dataloader, 
        num_epochs=5,
        out_dir=config["out_dir"],
    )

if __name__ == "__main__":
    main()