import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CyclicEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super(CyclicEmbedding, self).__init__()
        self.projection = nn.Linear(2, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = x * 2 * math.pi  # Escalar a [0, 2π]
        sin_emb = torch.sin(x)
        cos_emb = torch.cos(x)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        emb = self.projection(emb)
        emb = self.relu(emb)
        return emb


class BikeRoutePredictor(nn.Module):
    def __init__(
        self,
        n_stations,
        d_model=128,
        num_layers=4,
        num_heads=8,
        ids_embedding_dim=16,
        coords_embedding_dim=32,
        time_embedding_dim=32,
        dropout=0.1,
        max_len=5000,
    ):
        super(BikeRoutePredictor, self).__init__()
        self.n_stations, self.d_model, self.num_layers, self.num_heads = (
            n_stations,
            d_model,
            num_layers,
            num_heads,
        )

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
        start_stations,
        start_coords,
        start_times,
        end_stations,
        end_coords,
        end_times,
    ):
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
        embeddings = self.ff_reduce_dim(embeddings)
        embeddings = self.relu(embeddings)
        embeddings = embeddings.transpose(0, 1)
        embeddings = self.positional_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)

        transformer_output = self.transformer(embeddings)
        transformer_output = transformer_output.mean(dim=1)

        start_station_pred = self.fc_start_station(transformer_output)
        start_time_pred = self.fc_start_time(transformer_output)
        start_time_pred = self.relu(start_time_pred)
        end_station_pred = self.fc_end_station(transformer_output)
        end_time_pred = self.fc_end_time(transformer_output)
        end_time_pred = self.relu(end_time_pred)

        return start_station_pred, start_time_pred, end_station_pred, end_time_pred
