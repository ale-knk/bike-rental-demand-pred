import joblib
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from pybike.preprocessing import set_trips_df
from pybike.trip import Trips, CoordinateNormalizer



class TripsGen:
    def __init__(self, trips: Trips, seq_len: int = 15):
        self.seq_len = seq_len
        self.trips = trips

    def __len__(self):
        return len(self.trips) - self.seq_len + 1

    def __getitem__(self, idx):
        trip_seq = self.trips[idx : (idx + self.seq_len)]
        return Trips(trips=trip_seq)


class TripsDataset(Dataset):
    def __init__(self, trips_gen: TripsGen):
        self.trips_gen = trips_gen

    def __len__(self):
        return len(self.trips_gen)

    def __getitem__(self, idx):
        trip_sequence = self.trips_gen[idx]
        input_trips = Trips(trips=trip_sequence.trips[:-1])
        target_trip = trip_sequence.trips[-1]

        # Convertir los datos en tensores
        inputs = {
            "start_station_ids": self._get_tensor(
                input_trips, "get_start_stations_ids_tensor"
            ),
            "start_coords": self._get_tensor(
                input_trips, "get_start_coords_norm_tensor"
            ),
            "start_times": self._get_tensor(input_trips, "get_start_times_tensor"),
            "end_station_ids": self._get_tensor(
                input_trips, "get_end_stations_ids_tensor"
            ),
            "end_coords": self._get_tensor(input_trips, "get_end_coords_norm_tensor"),
            "end_times": self._get_tensor(input_trips, "get_end_times_tensor"),
        }

        targets = {
            "start_station_id": target_trip.start_station_id,
            "start_time": torch.tensor(target_trip.start_time_norm).unsqueeze(-1),
            "end_station_id": target_trip.end_station_id,
            "end_time": torch.tensor(target_trip.end_time_norm).unsqueeze(-1),
        }

        return inputs, targets

    def _get_tensor(self, trips, method_name):
        method = getattr(trips, method_name)
        return method()


def create_dataloader(
    trips: Trips, seq_len: int = 15, batch_size: int = 32, shuffle: bool = True
):
    trips_gen = TripsGen(trips, seq_len)
    dataset = TripsDataset(trips_gen)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def set_dataloaders(
    splits: list[str],
    seq_len: int = 15,
    batch_size: int = 32,
    shuffle: bool = True,
    scaler: CoordinateNormalizer | None = None,
):
    dataloaders = []
    for split in splits:
        trips_df = set_trips_df(split=split).iloc[:1000]
        trips = Trips.from_df(trips_df)
        if split == "train":
            scaler = CoordinateNormalizer()
            trips.normalize_coords(scaler, fit=True)
        else:
            if scaler is None:
                raise Exception("Scaler must be provided for validation and test sets")
            trips.normalize_coords(scaler, fit=False)

        dataloaders.append(create_dataloader(trips, seq_len, batch_size, shuffle))
    
    if len(dataloaders) == 1:
        return dataloaders[0]
    return dataloaders
