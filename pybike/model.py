import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the paper
    "Attention Is All You Need" by Vaswani et al.

    This module injects some information about the relative or absolute position
    of the tokens in the sequence. The positional encodings have the same dimension
    as the embeddings so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    Attributes:
        dropout (nn.Dropout): Dropout layer to apply after adding positional encoding.
        pe (Tensor): Positional encoding matrix of shape (max_len, 1, d_model).

    Methods:
        forward(x: Tensor) -> Tensor:
            Adds positional encoding to the input tensor and applies dropout.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model (embedding size).
            dropout (float): Dropout rate to apply after adding positional encoding.
            max_len (int): The maximum length of the input sequences.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to the input tensor and applies dropout.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, embedding_dim).

        Returns:
            Tensor: Output tensor with positional encoding added and dropout applied.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CyclicEmbedding(nn.Module):
    """
    Implements a cyclic embedding layer that projects cyclic features (e.g., time of day)
    into a higher-dimensional space using sine and cosine transformations.

    Attributes:
        projection (nn.Linear): Linear layer to project the concatenated sine and cosine embeddings.
        relu (nn.ReLU): ReLU activation function applied after the projection.

    Methods:
        forward(x: Tensor) -> Tensor:
            Applies the cyclic embedding transformation to the input tensor.
    """

    def __init__(self, embedding_dim: int):
        """
        Initializes the CyclicEmbedding module.

        Args:
            embedding_dim (int): The dimension of the output embedding.
        """
        super(CyclicEmbedding, self).__init__()
        self.projection = nn.Linear(2, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the cyclic embedding transformation to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, 1) with values scaled to [0, 1].

        Returns:
            Tensor: Output tensor with cyclic embeddings of shape (batch_size, seq_len, embedding_dim).
        """
        x = x * 2 * math.pi  # Scale to [0, 2π]
        sin_emb = torch.sin(x)
        cos_emb = torch.cos(x)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        emb = self.projection(emb)
        # emb = self.relu(emb)
        return emb


class BikeRoutePredictor(nn.Module):
    """
    Predicts bike routes using embeddings for station IDs, coordinates, and cyclic time features.

    Attributes:
        n_stations (int): Number of bike stations.
        d_model (int): Dimension of the model.
        num_layers (int): Number of layers in the model.
        num_heads (int): Number of attention heads.
        ids_embedding_dim (int): Dimension of the station ID embeddings.
        coords_embedding_dim (int): Dimension of the coordinate embeddings.
        time_embedding_dim (int): Dimension of the time embeddings.
        dropout (float): Dropout rate.
        max_len (int): Maximum length of the input sequences.
        start_station_embedding (nn.Embedding): Embedding layer for start station IDs.
        start_coords_embedding (nn.Linear): Linear layer for start coordinates.
        end_station_embedding (nn.Embedding): Embedding layer for end station IDs.
        end_coords_embedding (nn.Linear): Linear layer for end coordinates.
        cyclic_embedding_start (CyclicEmbedding): Cyclic embedding layer for start times.
        cyclic_embedding_end (CyclicEmbedding): Cyclic embedding layer for end times.
        ff_reduce_dim (nn.Linear): Linear layer to reduce concatenated embeddings to model dimension.

    Methods:
        forward(x: Tensor) -> Tensor:
            Forward pass of the model.
    """

    def __init__(
        self,
        n_stations: int = 70,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ids_embedding_dim: int = 16,
        coords_embedding_dim: int = 32,
        time_embedding_dim: int = 32,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        """
        Initializes the BikeRoutePredictor module.

        Args:
            n_stations (int): Number of bike stations.
            d_model (int): Dimension of the model.
            num_layers (int): Number of layers in the model.
            num_heads (int): Number of attention heads.
            ids_embedding_dim (int): Dimension of the station ID embeddings.
            coords_embedding_dim (int): Dimension of the coordinate embeddings.
            time_embedding_dim (int): Dimension of the time embeddings.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of the input sequences.
        """
        super(BikeRoutePredictor, self).__init__()
        self.n_stations = n_stations
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ids_embedding_dim = ids_embedding_dim
        self.coords_embedding_dim = coords_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.dropout = dropout
        self.max_len = max_len

        self.start_station_embedding = nn.Embedding(n_stations, ids_embedding_dim)
        self.start_coords_embedding = nn.Linear(2, coords_embedding_dim)
        self.end_station_embedding = nn.Embedding(n_stations, ids_embedding_dim)
        self.end_coords_embedding = nn.Linear(2, coords_embedding_dim)

        self.cyclic_embedding_start = CyclicEmbedding(embedding_dim=time_embedding_dim)
        self.cyclic_embedding_end = CyclicEmbedding(embedding_dim=time_embedding_dim)

        concat_dim = (
            ids_embedding_dim * 2 + coords_embedding_dim * 2 + time_embedding_dim * 2
        )
        self.ff_reduce_dim = nn.Linear(concat_dim, d_model)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, batch_first=True
            ),
            num_layers=num_layers,
        )

        self.fc_start_station = nn.Linear(d_model, n_stations)
        self.fc_end_station = nn.Linear(d_model, n_stations)
        self.fc_start_time = nn.Linear(d_model, 1)
        self.fc_end_time = nn.Linear(d_model, 1)

    def forward(
        self,
        start_stations: Tensor,  # shape: (batch_size, seq_len)
        start_coords: Tensor,  # shape: (batch_size, seq_len, 2)
        start_times: Tensor,  # shape: (batch_size, seq_len)
        end_stations: Tensor,  # shape: (batch_size, seq_len)
        end_coords: Tensor,  # shape: (batch_size, seq_len, 2)
        end_times: Tensor,  # shape: (batch_size, seq_len)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            start_stations (Tensor): Tensor containing the start station IDs.
            start_coords (Tensor): Tensor containing the start coordinates.
            start_times (Tensor): Tensor containing the start times.
            end_stations (Tensor): Tensor containing the end station IDs.
            end_coords (Tensor): Tensor containing the end coordinates.
            end_times (Tensor): Tensor containing the end times.

        Returns:
            Tensor: Output tensor.
        """
        start_stations_embeds = self.start_station_embedding(start_stations)
        start_coords_embeds = self.start_coords_embedding(start_coords)
        end_stations_embeds = self.end_station_embedding(end_stations)
        end_coords_embeds = self.end_coords_embedding(end_coords)

        start_times_embeds = self.cyclic_embedding_start(start_times)
        end_times_embeds = self.cyclic_embedding_end(end_times)

        embeddings = torch.cat(
            (
                start_stations_embeds,
                start_coords_embeds,
                start_times_embeds,
                end_stations_embeds,
                end_coords_embeds,
                end_times_embeds,
            ),
            dim=-1,
        )
        embeddings = self.ff_reduce_dim(
            embeddings
        )  # shape: (batch_size, seq_len, d_model)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.relu(embeddings)
        # embeddings = embeddings.transpose(0, 1)  # shape: (seq_len, batch_size, d_model)
        embeddings = self.positional_encoding(embeddings)
        # embeddings = embeddings.transpose(0, 1)

        transformer_output = self.transformer(embeddings)
        transformer_output = transformer_output[
            :, -1, :
        ]  # shape: (batch_size, d_model)

        start_station_pred = self.fc_start_station(transformer_output)
        start_time_pred = self.fc_start_time(transformer_output)
        end_station_pred = self.fc_end_station(transformer_output)
        end_time_pred = self.fc_end_time(transformer_output)

        start_station_pred = self.softmax(start_station_pred)
        end_station_pred = self.softmax(end_station_pred)
        start_time_pred = self.sigmoid(start_time_pred)
        end_time_pred = self.sigmoid(end_time_pred)

        return start_station_pred, start_time_pred, end_station_pred, end_time_pred

    def get_config(self):
        """
        Returns the configuration of the BikeRoutePredictor model.

        Returns:
            dict: A dictionary containing the configuration parameters of the model.
        """
        return {
            "n_stations": self.n_stations,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ids_embedding_dim": self.ids_embedding_dim,
            "coords_embedding_dim": self.coords_embedding_dim,
            "time_embedding_dim": self.time_embedding_dim,
            "dropout": self.dropout,
            "max_len": self.max_len,
        }
