import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


class Config:
    STATIONS_CSV_PATH = os.getenv(
        "STATIONS_CSV_PATH", BASE_DIR.parent / "data/stations.csv"
    )
    TRIPS_CSV_PATH = os.getenv("TRIPS_CSV_PATH", BASE_DIR.parent / "data/trips.csv")
    STATUS_CSV_PATH = os.getenv("STATUS_CSV_PATH", BASE_DIR.parent / "data/status.csv")


class DataloaderConfig(BaseModel):
    seq_len: int = Field(default=41, description="Sequence length")
    batch_size: int = Field(default=300, description="Batch size")
    shuffle: bool = Field(default=True, description="Whether to shuffle the data")


class ModelConfig(BaseModel):
    n_stations: int = Field(..., description="Number of stations")
    d_model: int = Field(..., description="Model dimensionality")
    num_layers: int = Field(..., description="Number of layers")
    num_heads: int = Field(..., description="Number of attention heads")
    ids_embedding_dim: int = Field(..., description="Dimension of ID embeddings")
    coords_embedding_dim: int = Field(
        ..., description="Dimension of coordinate embeddings"
    )
    time_embedding_dim: int = Field(..., description="Dimension of time embeddings")
    dropout: float = Field(default=0.1, description="Dropout rate")

class OptimizerConfig(BaseModel):
    lr: float = Field(..., description="Learning rate")
    betas: List[float] = Field(..., description="Betas for optimizer")
    eps: float = Field(..., description="Epsilon for optimizer")
    weight_decay: float = Field(..., description="Weight decay")


class TrainConfig(BaseModel):
    config_dataloader: DataloaderConfig
    config_model: ModelConfig
    config_optimizer: OptimizerConfig

    out_dir: str = Field(default="./outputs", description="Output directory")

    def save(self, path: str) -> None:
        """Saves the configuration to a JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        """Loads the configuration from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
            data["config_dataloader"] = DataloaderConfig(**data["config_dataloader"])
            data["config_model"] = ModelConfig(**data["config_model"])
            data["config_optimizer"] = OptimizerConfig(**data["config_optimizer"])
        return cls(**data)

    @property
    def model_path(self) -> str:
        return os.path.join(self.out_dir, "model.pth")

    @property
    def scaler_path(self) -> str:
        return os.path.join(self.out_dir, "scaler.pkl")

    @property
    def optimizer_path(self) -> str:
        return os.path.join(self.out_dir, "optimizer.pth")

    @property
    def metrics_plots_path(self) -> str:
        return os.path.join(self.out_dir, "metrics.png")

    @property
    def metrics_json_path(self) -> str:
        return os.path.join(self.out_dir, "metrics.json")

    @property
    def losses_plots_path(self) -> str:
        return os.path.join(self.out_dir, "loss.png")

    @property
    def losses_json_path(self) -> str:
        return os.path.join(self.out_dir, "losses.json")
    
class EvalConfig(BaseModel):
    config_train: TrainConfig = Field(..., description="Training configuration")
    out_dir: str = Field(default="./outputs/eval", description="Output directory")
    config_dataloader: DataloaderConfig

    @classmethod
    def from_json(cls, path: str) -> "EvalConfig":
        """Loads the configuration from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
            data["config_train"] = TrainConfig.from_json(os.path.join(data["train_dir"],"train_config.json"))
            data["config_dataloader"] = DataloaderConfig(
                seq_len=data["config_train"].config_dataloader.seq_len,
                shuffle=False,
                **data["config_dataloader"])
            data.pop("train_dir")
        return cls(**data)
    
    def save(self, path: str) -> None:
        """Saves the configuration to a JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))



