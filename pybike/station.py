from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, validator

from pybike.preprocessing import set_stations_df

stations_df = set_stations_df()


class Station(BaseModel):
    """
    Represents a bike-sharing station with various attributes.

    This class uses Pydantic's BaseModel to enforce type validations and constraints
    on station data. It includes methods for creating instances from pandas Series
    and retrieving station data by ID.

    Attributes:
        id (int): ID of the station (must be between 0 and 69).
        id_unmapped (int): Original ID of the station before preprocessing (must be between 2 and 84).
        name (str): Name of the station (maximum length 255 characters).
        long (float): Longitude of the station.
        lat (float): Latitude of the station.
        dock_count (int): Number of docks at the station (must be non-negative).
        city (str): City where the station is located (maximum length 255 characters).
        cluster (int): Cluster number of the station (must be non-negative).
        installation_date (datetime): Date when the station was installed.
        lat_norm (Optional[float]): Normalized latitude of the station.
        long_norm (Optional[float]): Normalized longitude of the station.
    """

    id: int = Field(
        ..., ge=0, le=69, description="ID of the station (must be between 0 and 69)"
    )
    id_unmapped: int = Field(
        ...,
        ge=2,
        le=84,
        description="Original ID of the station before preprocessing (must be between 2 and 84)",
    )
    name: str = Field(..., max_length=255, description="Name of the station")
    long: float = Field(..., description="Longitude of the station")
    lat: float = Field(..., description="Latitude of the station")
    dock_count: int = Field(..., ge=0, description="Number of docks at the station")
    city: str = Field(
        ..., max_length=255, description="City where the station is located"
    )
    cluster: int = Field(..., ge=0, description="Cluster number of the station")
    installation_date: datetime = Field(
        ..., description="Date when the station was installed"
    )
    lat_norm: Optional[float] = Field(
        None, description="Normalized latitude of the station"
    )
    long_norm: Optional[float] = Field(
        None, description="Normalized longitude of the station"
    )

    class Config:
        """
        Configuration for the Station model.

        - `arbitrary_types_allowed`: Allows types not natively supported by Pydantic.
        - `json_encoders`: Custom JSON encoders for specific types.
        """

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    @validator("installation_date", pre=True, always=True)
    def parse_installation_date(cls, value: str) -> datetime:
        """
        Validates and parses the installation_date field.

        Args:
            value (str): The installation date as a string.

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
    def from_series(cls, series: pd.Series) -> "Station":
        """
        Creates a Station instance from a pandas Series.

        Args:
            series (pd.Series): A pandas Series containing station data.

        Returns:
            Station: An instance of the Station class.
        """
        data = series.to_dict()
        return cls(**data)

    @classmethod
    def from_id(cls, id: int) -> "Station":
        """
        Retrieves a Station instance by its ID.

        Args:
            id (int): The ID of the station to retrieve.

        Returns:
            Station: An instance of the Station class corresponding to the given ID.

        Raises:
            IndexError: If no station with the specified ID exists.
        """
        try:
            station_row = stations_df.loc[stations_df["id"] == id].iloc[0]
        except IndexError:
            raise IndexError(f"No station found with id: {id}")
        return cls(**station_row.to_dict())

    @property
    def coords(self) -> tuple[float, float]:
        """
        Retrieves the geographic coordinates of the station.

        Returns:
            tuple[float, float]: A tuple containing latitude and longitude.
        """
        return self.lat, self.long

    @property
    def coords_norm(self) -> tuple[float, float]:
        """
        Retrieves the normalized geographic coordinates of the station.

        Returns:
            Optional[tuple[float, float]]: A tuple containing normalized latitude and longitude,
                                           or None if normalization has not been performed.
        """
        return self.lat_norm, self.long_norm

    def to_dict(self) -> dict:
        """
        Converts the Station instance to a dictionary.

        Returns:
            dict: A dictionary representation of the Station instance.
        """
        return self.model_dump()

    def to_series(self) -> pd.Series:
        """
        Converts the Station instance to a pandas Series.

        Returns:
            pd.Series: A pandas Series representation of the Station instance.
        """
        return pd.Series(self.to_dict())
