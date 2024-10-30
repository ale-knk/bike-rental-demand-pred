from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, validator

from pybike.preprocessing import set_stations_df

stations_df = set_stations_df()


class Station(BaseModel):
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
    lat_norm: Optional[float] = Field(None, description="Normalized latitude of the station")
    long_norm: Optional[float] = Field(None, description="Normalized longitude of the station")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    @validator("installation_date", pre=True, always=True)
    def parse_installation_date(cls, value):
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
            except ValueError:
                raise ValueError(f"Invalid date format: {value}")
        return value

    @classmethod
    def from_series(cls, series: pd.Series):
        data = series.to_dict()
        return cls(**data)

    @classmethod
    def from_id(cls, id: int):
        station_row = stations_df.loc[stations_df["id"] == id, :].iloc[0,:]
        return cls(**station_row.to_dict())
    
    @property
    def coords(self):
        return self.lat, self.long

    @property
    def coords_norm(self):
        return self.lat_norm, self.long_norm
    
    def to_dict(self):
        return self.model_dump()

    def to_series(self):
        return pd.Series(self.to_dict())
