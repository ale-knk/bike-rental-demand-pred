from datetime import datetime

import joblib
import pandas as pd
import torch
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import MinMaxScaler

from pybike.station import Station
from pybike.utils import normalize_time


class CoordinateNormalizer:
    def __init__(self):
        self.feature_names = ["start_lat", "start_long", "end_lat", "end_long"]
        self.scaler = MinMaxScaler()

    def _split_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        df_split = df.copy()
        df_split[["start_lat", "start_long"]] = pd.DataFrame(
            df_split["start_station_coords"].tolist(), index=df.index
        )
        df_split[["end_lat", "end_long"]] = pd.DataFrame(
            df_split["end_station_coords"].tolist(), index=df.index
        )
        return df_split

    def fit(self, df: pd.DataFrame):
        df_split = self._split_coords(df)
        self.scaler.fit(df_split[self.feature_names])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_split = self._split_coords(df)
        df_split[self.feature_names] = self.scaler.transform(
            df_split[self.feature_names]
        )
        df_copy = df.copy()
        df_copy["start_station_coords"] = list(
            zip(df_split["start_lat"], df_split["start_long"])
        )
        df_copy["end_station_coords"] = list(
            zip(df_split["end_lat"], df_split["end_long"])
        )
        return df_copy

    def save(self, filepath: str):
        joblib.dump(self.scaler, filepath)

    def load(self, filepath: str):
        self.scaler = joblib.load(filepath)


class Trip(BaseModel):
    start_date: datetime = Field(..., description="Fecha y hora de inicio del viaje")
    end_date: datetime = Field(..., description="Fecha y hora de fin del viaje")
    duration: int = Field(None, description="Duration of the trip in seconds")
    start_station: Station = Field(None, description="Estación de inicio del viaje")
    end_station: Station = Field(None, description="Estación de fin del viaje")
    split: str = Field(None, description="Train, validation or test split")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}

    @validator("start_date", "end_date", pre=True, always=True)
    def parse_dates(cls, value):
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
            except ValueError:
                raise ValueError(f"Invalid date format: {value}")
        return value

    @classmethod
    def from_series(cls, series: pd.Series):
        return cls(
            start_date=series["start_date"],
            end_date=series["end_date"],
            duration=series["duration"],
            start_station=Station.from_id(series["start_station_id"]),
            end_station=Station.from_id(series["end_station_id"]),
            split=series["split"],
        )

    def to_dict(self, model_shape: bool = False) -> dict:
        if not model_shape:
            return self.model_dump()
        return {
            "start_station_id": self.start_station_id,
            "start_station_coords": self.start_station_coords,
            "start_time_norm": self.start_time_norm,
            "end_station_id": self.end_station_id,
            "end_station_coords": self.end_station_coords,
            "end_time_norm": self.end_time_norm,
        }

    @property
    def start_station_coords(self):
        return self.start_station.coords

    @property
    def start_station_coords_norm(self):
        return self.start_station.coords_norm

    @property
    def start_station_id(self):
        return self.start_station.id

    @property
    def start_time_norm(self):
        return normalize_time(self.start_date)

    @property
    def end_station_coords(self):
        return self.end_station.coords

    @property
    def end_station_coords_norm(self):
        return self.end_station.coords_norm

    @property
    def end_station_id(self):
        return self.end_station.id

    @property
    def end_time_norm(self):
        return normalize_time(self.end_date)


class Trips(BaseModel):
    trips: list[Trip] = Field(..., description="List of Trip objects")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}

    def __len__(self):
        return len(self.trips) if self.trips else 0

    def __getitem__(self, i: int):
        if not self.trips:
            raise IndexError("Trips list is empty")
        return self.trips[i]

    def __iter__(self):
        return iter(self.trips)

    @classmethod
    def from_df(cls, trips_df: pd.DataFrame):
        trips = [Trip.from_series(row) for _, row in trips_df.iterrows()]
        return cls(trips=trips)

    def to_dicts(self, model_shape: bool = False) -> list[dict]:
        return [trip.to_dict(model_shape) for trip in self.trips]

    def to_df(self, model_shape: bool = False) -> pd.DataFrame:
        return pd.DataFrame(self.to_dicts(model_shape))

    def get_start_stations_ids_tensor(self, add_batch_dim=False):
        return self._get_station_ids_tensor("start_station_id", add_batch_dim)

    def get_end_stations_ids_tensor(self, add_batch_dim=False):
        return self._get_station_ids_tensor("end_station_id", add_batch_dim)

    def get_start_coords_norm_tensor(self, add_batch_dim=False):
        return self._get_coords_tensor("start_station_coords_norm", add_batch_dim)

    def get_end_coords_norm_tensor(self, add_batch_dim=False):
        return self._get_coords_tensor("end_station_coords_norm", add_batch_dim)

    def get_start_times_tensor(self, add_batch_dim=False):
        return self._get_times_tensor("start_time_norm", add_batch_dim)

    def get_end_times_tensor(self, add_batch_dim=False):
        return self._get_times_tensor("end_time_norm", add_batch_dim)

    def _get_station_ids_tensor(self, attr: str, add_batch_dim: bool):
        stations_ids = [getattr(trip, attr) for trip in self.trips]
        stations_ids = torch.tensor(stations_ids)
        if add_batch_dim:
            stations_ids = stations_ids.unsqueeze(0)
        return stations_ids

    def _get_coords_tensor(self, method: str, add_batch_dim: bool):
        stations_coords = [getattr(trip, method) for trip in self.trips]
        stations_coords = torch.tensor(stations_coords)
        if add_batch_dim:
            stations_coords = stations_coords.unsqueeze(0)
        return stations_coords

    def _get_times_tensor(self, attr: str, add_batch_dim: bool):
        times = [getattr(trip, attr) for trip in self.trips]
        times = torch.tensor(times).unsqueeze(-1)
        if add_batch_dim:
            times = times.unsqueeze(0)
        return times

    def normalize_coords(self, scaler: CoordinateNormalizer, fit: bool = False):
        if fit:
            scaler.fit(self.to_df(model_shape=True))
        df_norm = scaler.transform(self.to_df(model_shape=True))

        for trip, (_, row) in zip(self.trips, df_norm.iterrows()):
            trip.start_station.lat_norm = row["start_station_coords"][0]
            trip.start_station.long_norm = row["start_station_coords"][1]
            trip.end_station.lat_norm = row["end_station_coords"][0]
            trip.end_station.long_norm = row["end_station_coords"][1]
