from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import torch
from pydantic import BaseModel, Field, validator

from pybike.preprocessing import CoordinateNormalizer
from pybike.station import Station
from pybike.utils import normalize_time


class Trip(BaseModel):
    """
    Represents a bike-sharing trip with various attributes.

    This class uses Pydantic's BaseModel to enforce type validations and constraints
    on trip data. It includes methods for creating instances from pandas Series
    and converting trip data to different formats.

    Attributes:
        start_date (datetime): The start date and time of the trip.
        end_date (datetime): The end date and time of the trip.
        duration (Optional[int]): Duration of the trip in seconds.
        start_station (Optional[Station]): The station where the trip started.
        end_station (Optional[Station]): The station where the trip ended.
        split (Optional[str]): Indicates whether the trip belongs to the train, validation, or test split.
    """

    start_date: datetime = Field(..., description="Start date and time of the trip")
    end_date: datetime = Field(..., description="End date and time of the trip")
    duration: Optional[int] = Field(None, description="Duration of the trip in seconds")
    start_station: Optional["Station"] = Field(
        None, description="Start station of the trip"
    )
    end_station: Optional["Station"] = Field(
        None, description="End station of the trip"
    )
    split: Optional[str] = Field(None, description="Train, validation, or test split")

    class Config:
        """
        Configuration for the Trip model.

        - `arbitrary_types_allowed`: Allows types not natively supported by Pydantic.
        - `json_encoders`: Custom JSON encoders for specific types.
        """

        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}

    @validator("start_date", "end_date", pre=True, always=True)
    def parse_dates(cls, value: Any) -> datetime:
        """
        Validates and parses the start_date and end_date fields.

        Args:
            value (Any): The date value to validate and parse.

        Returns:
            datetime: The parsed datetime object.

        Raises:
            ValueError: If the date string is not in the expected format.
        """

        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
            except ValueError:
                raise ValueError(f"Invalid date format: {value}")
        return value

    @classmethod
    def from_series(cls, series: pd.Series) -> "Trip":
        """
        Creates a Trip instance from a pandas Series.

        Args:
            series (pd.Series): A pandas Series containing trip data.

        Returns:
            Trip: An instance of the Trip class.
        """
        return cls(
            start_date=series["start_date"],
            end_date=series["end_date"],
            duration=series["duration"],
            start_station=Station.from_id(series["start_station_id"]),
            end_station=Station.from_id(series["end_station_id"]),
            split=series["split"],
        )

    def to_dict(self, model_shape: bool = False) -> Dict[str, Any]:
        """
        Converts the Trip instance to a dictionary.

        Args:
            model_shape (bool, optional): Determines the shape of the output dictionary.
                - If `False`, returns the standard model dictionary.
                - If `True`, returns a dictionary tailored for model input.
                Defaults to `False`.

        Returns:
            Dict[str, Any]: A dictionary representation of the Trip instance.
        """
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
    def start_station_coords(self) -> Optional[tuple[float, float]]:
        """
        Retrieves the geographic coordinates of the start station.

        Returns:
            Optional[tuple[float, float]]: A tuple containing latitude and longitude,
                                           or `None` if start_station is not set.
        """
        return self.start_station.coords if self.start_station else None

    @property
    def start_station_coords_norm(self) -> Optional[tuple[float, float]]:
        """
        Retrieves the normalized geographic coordinates of the start station.

        Returns:
            Optional[tuple[float, float]]: A tuple containing normalized latitude and longitude,
                                           or `None` if normalization has not been performed or start_station is not set.
        """
        return self.start_station.coords_norm if self.start_station else None

    @property
    def start_station_id(self) -> Optional[int]:
        """
        Retrieves the ID of the start station.

        Returns:
            Optional[int]: The ID of the start station, or `None` if start_station is not set.
        """
        return self.start_station.id if self.start_station else None

    @property
    def start_time_norm(self) -> Optional[float]:
        """
        Retrieves the normalized start time of the trip.

        Returns:
            Optional[float]: The normalized start time, or `None` if start_date is not set.
        """
        return normalize_time(self.start_date) if self.start_date else None

    @property
    def end_station_coords(self) -> Optional[tuple[float, float]]:
        """
        Retrieves the geographic coordinates of the end station.

        Returns:
            Optional[tuple[float, float]]: A tuple containing latitude and longitude,
                                           or `None` if end_station is not set.
        """
        return self.end_station.coords if self.end_station else None

    @property
    def end_station_coords_norm(self) -> Optional[tuple[float, float]]:
        """
        Retrieves the normalized geographic coordinates of the end station.

        Returns:
            Optional[tuple[float, float]]: A tuple containing normalized latitude and longitude,
                                           or `None` if normalization has not been performed or end_station is not set.
        """
        return self.end_station.coords_norm if self.end_station else None

    @property
    def end_station_id(self) -> Optional[int]:
        """
        Retrieves the ID of the end station.

        Returns:
            Optional[int]: The ID of the end station, or `None` if end_station is not set.
        """
        return self.end_station.id if self.end_station else None

    @property
    def end_time_norm(self) -> Optional[float]:
        """
        Retrieves the normalized end time of the trip.

        Returns:
            Optional[float]: The normalized end time, or `None` if end_date is not set.
        """
        return normalize_time(self.end_date) if self.end_date else None

    def to_series(self) -> pd.Series:
        """
        Converts the Trip instance to a pandas Series.

        Returns:
            pd.Series: A pandas Series representation of the Trip instance.
        """
        return pd.Series(self.to_dict())


class Trips(BaseModel):
    """
    Represents a collection of bike-sharing trips.

    This class uses Pydantic's BaseModel to enforce type validations and constraints
    on a list of Trip objects. It provides methods to interact with the collection
    similar to a standard Python list, including indexing, iteration, and length retrieval.

    Attributes:
        trips (List[Trip]): A list of Trip objects.
    """

    trips: List[Trip] = Field(..., description="List of Trip objects")

    class Config:
        """
        Configuration for the Trips model.

        - `arbitrary_types_allowed`: Allows types not natively supported by Pydantic.
        - `json_encoders`: Custom JSON encoders for specific types.
        """

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def __len__(self) -> int:
        """
        Returns the number of trips in the collection.

        Returns:
            int: The number of Trip objects in the `trips` list. Returns 0 if the list is empty.
        """
        return len(self.trips) if self.trips else 0

    def __getitem__(self, index: int) -> Trip:
        """
        Retrieves a Trip object by its index in the collection.

        Args:
            index (int): The index of the Trip to retrieve.

        Returns:
            Trip: The Trip object at the specified index.

        Raises:
            IndexError: If the `trips` list is empty or the index is out of range.
        """
        if not self.trips:
            raise IndexError("Trips list is empty")
        return self.trips[index]

    def __iter__(self) -> Iterator[Trip]:
        """
        Returns an iterator over the Trip objects in the collection.

        Returns:
            Iterator[Trip]: An iterator over the `trips` list.
        """
        return iter(self.trips)

    @classmethod
    def from_df(cls, trips_df: pd.DataFrame) -> "Trips":
        """
        Creates a TripCollection instance from a DataFrame.

        Args:
            trips_df (pd.DataFrame): DataFrame containing trip data.

        Returns:
            TripCollection: An instance of TripCollection with trips initialized from the DataFrame.
        """
        trips = [Trip.from_series(row) for _, row in trips_df.iterrows()]
        return cls(trips=trips)

    def to_dicts(self, model_shape: bool = False) -> List[Dict]:
        """
        Converts the trips in the collection to a list of dictionaries.

        Args:
            model_shape (bool, optional): Whether to use the model shape for the dictionaries. Defaults to False.

        Returns:
            list[dict]: A list of dictionaries representing the trips.
        """
        return [trip.to_dict(model_shape) for trip in self.trips]

    def to_df(self, model_shape: bool = False) -> pd.DataFrame:
        """
        Converts the trips in the collection to a DataFrame.

        Args:
            model_shape (bool, optional): Whether to use the model shape for the DataFrame. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame representing the trips.
        """
        return pd.DataFrame(self.to_dicts(model_shape))

    def get_start_stations_ids_tensor(
        self, add_batch_dim: bool = False
    ) -> torch.Tensor:
        """
        Returns a tensor of start station IDs.

        Args:
            add_batch_dim (bool, optional): Whether to add a batch dimension to the tensor. Defaults to False.

        Returns:
            torch.Tensor: A tensor of start station IDs.
        """
        return self._get_station_ids_tensor("start_station_id", add_batch_dim)

    def get_end_stations_ids_tensor(self, add_batch_dim: bool = False) -> torch.Tensor:
        """
        Returns a tensor of end station IDs.

        Args:
            add_batch_dim (bool, optional): Whether to add a batch dimension to the tensor. Defaults to False.

        Returns:
            torch.Tensor: A tensor of end station IDs.
        """
        return self._get_station_ids_tensor("end_station_id", add_batch_dim)

    def get_start_coords_norm_tensor(self, add_batch_dim: bool = False) -> torch.Tensor:
        """
        Returns a tensor of normalized start station coordinates.

        Args:
            add_batch_dim (bool, optional): Whether to add a batch dimension to the tensor. Defaults to False.

        Returns:
            torch.Tensor: A tensor of normalized start station coordinates.
        """
        return self._get_coords_tensor("start_station_coords_norm", add_batch_dim)

    def get_end_coords_norm_tensor(self, add_batch_dim: bool = False) -> torch.Tensor:
        """
        Returns a tensor of normalized end station coordinates.

        Args:
            add_batch_dim (bool, optional): Whether to add a batch dimension to the tensor. Defaults to False.

        Returns:
            torch.Tensor: A tensor of normalized end station coordinates.
        """
        return self._get_coords_tensor("end_station_coords_norm", add_batch_dim)

    def get_start_times_tensor(self, add_batch_dim: bool = False) -> torch.Tensor:
        """
        Returns a tensor of normalized start times.

        Args:
            add_batch_dim (bool, optional): Whether to add a batch dimension to the tensor. Defaults to False.

        Returns:
            torch.Tensor: A tensor of normalized start times.
        """
        return self._get_times_tensor("start_time_norm", add_batch_dim)

    def get_end_times_tensor(self, add_batch_dim: bool = False) -> torch.Tensor:
        """
        Returns a tensor of normalized end times.

        Args:
            add_batch_dim (bool, optional): Whether to add a batch dimension to the tensor. Defaults to False.

        Returns:
            torch.Tensor: A tensor of normalized end times.
        """
        return self._get_times_tensor("end_time_norm", add_batch_dim)

    def _get_station_ids_tensor(self, attr: str, add_batch_dim: bool) -> torch.Tensor:
        """
        Helper method to get a tensor of station IDs.

        Args:
            attr (str): The attribute name for station IDs.
            add_batch_dim (bool): Whether to add a batch dimension to the tensor.

        Returns:
            torch.Tensor: A tensor of station IDs.
        """
        stations_ids = [getattr(trip, attr) for trip in self.trips]
        stations_ids = torch.tensor(stations_ids)
        if add_batch_dim:
            stations_ids = stations_ids.unsqueeze(0)
        return stations_ids

    def _get_coords_tensor(self, method: str, add_batch_dim: bool) -> torch.Tensor:
        """
        Helper method to get a tensor of coordinates.

        Args:
            method (str): The method name for coordinates.
            add_batch_dim (bool): Whether to add a batch dimension to the tensor.

        Returns:
            torch.Tensor: A tensor of coordinates.
        """
        stations_coords = [getattr(trip, method) for trip in self.trips]
        stations_coords = torch.tensor(stations_coords)
        if add_batch_dim:
            stations_coords = stations_coords.unsqueeze(0)
        return stations_coords

    def _get_times_tensor(self, attr: str, add_batch_dim: bool) -> torch.Tensor:
        """
        Helper method to get a tensor of times.

        Args:
            attr (str): The attribute name for times.
            add_batch_dim (bool): Whether to add a batch dimension to the tensor.

        Returns:
            torch.Tensor: A tensor of times.
        """
        times = [getattr(trip, attr) for trip in self.trips]
        times = torch.tensor(times).unsqueeze(-1)
        if add_batch_dim:
            times = times.unsqueeze(0)
        return times

    def normalize_coords(
        self, scaler: "CoordinateNormalizer", fit: bool = False
    ) -> None:
        """
        Normalizes the coordinates of the trips in the collection using the provided scaler.

        Args:
            scaler (CoordinateNormalizer): The scaler to use for normalizing the coordinates.
            fit (bool, optional): Whether to fit the scaler to the data before transforming. Defaults to False.

        Returns:
            None
        """
        if fit:
            scaler.fit(self.to_df(model_shape=True))
        df_norm = scaler.transform(self.to_df(model_shape=True))

        for trip, (_, row) in zip(self.trips, df_norm.iterrows()):
            trip.start_station.lat_norm = row["start_station_coords"][0]
            trip.start_station.long_norm = row["start_station_coords"][1]
            trip.end_station.lat_norm = row["end_station_coords"][0]
            trip.end_station.long_norm = row["end_station_coords"][1]
