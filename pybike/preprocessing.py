from datetime import datetime
from typing import List

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .config import Config


def normalize_hour(dt: datetime) -> float:
    """
    Normalize the hour and minute of a datetime object to a value between 0 and 1.

    Parameters:
    dt (datetime): The datetime object to normalize.

    Returns:
    float: The normalized hour and minute.
    """
    return dt.hour / 24.0 + dt.minute / 1440.0


class CoordinateNormalizer:
    """
    A utility class for normalizing geographic coordinates using Min-Max scaling.

    This class provides methods to fit a scaler on coordinate data, transform the data
    using the fitted scaler, and persist the scaler for future use. It is designed to
    handle coordinate normalization for both training and inference datasets.

    Attributes:
        feature_names (List[str]): List of feature names to be normalized.
        scaler (MinMaxScaler): The scaler used for normalization.
    """

    def __init__(self) -> None:
        """
        Initializes the CoordinateNormalizer with default feature names and a MinMaxScaler.
        """
        self.feature_names: List[str] = [
            "start_lat",
            "start_long",
            "end_lat",
            "end_long",
        ]
        self.scaler: MinMaxScaler = MinMaxScaler()

    def _split_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the coordinate tuples into separate latitude and longitude columns.

        This private helper method takes a DataFrame with 'start_station_coords' and
        'end_station_coords' columns, which contain tuples of (latitude, longitude),
        and splits them into individual columns for each component.

        Args:
            df (pd.DataFrame): The input DataFrame containing coordinate tuples.

        Returns:
            pd.DataFrame: A DataFrame with separate columns for start and end latitudes and longitudes.
        """
        df_split: pd.DataFrame = df.copy()
        df_split[["start_lat", "start_long"]] = pd.DataFrame(
            df_split["start_station_coords"].tolist(), index=df.index
        )
        df_split[["end_lat", "end_long"]] = pd.DataFrame(
            df_split["end_station_coords"].tolist(), index=df.index
        )
        return df_split

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits the MinMaxScaler on the provided DataFrame's coordinate features.

        This method extracts the relevant coordinate features from the DataFrame,
        splits them into separate columns, and fits the scaler to these features.

        Args:
            df (pd.DataFrame): The input DataFrame containing coordinate data.

        Raises:
            ValueError: If the DataFrame does not contain the required coordinate columns.
        """
        df_split: pd.DataFrame = self._split_coords(df)
        if not all(feature in df_split.columns for feature in self.feature_names):
            missing = set(self.feature_names) - set(df_split.columns)
            raise ValueError(f"Missing coordinate columns: {missing}")
        self.scaler.fit(df_split[self.feature_names])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the coordinate features in the DataFrame using the fitted scaler.

        This method normalizes the coordinate features and updates the original DataFrame
        with the normalized coordinates.

        Args:
            df (pd.DataFrame): The input DataFrame containing coordinate data.

        Returns:
            pd.DataFrame: A DataFrame with normalized coordinate features.

        Raises:
            ValueError: If the scaler has not been fitted or if the DataFrame lacks required columns.
        """
        if not hasattr(self.scaler, "scale_"):
            raise ValueError(
                "Scaler has not been fitted. Call `fit` with appropriate data before `transform`."
            )

        df_split: pd.DataFrame = self._split_coords(df)
        if not all(feature in df_split.columns for feature in self.feature_names):
            missing = set(self.feature_names) - set(df_split.columns)
            raise ValueError(f"Missing coordinate columns: {missing}")

        # Apply scaling to the feature columns
        df_split[self.feature_names] = self.scaler.transform(
            df_split[self.feature_names]
        )

        # Reconstruct the coordinate tuples with normalized values
        df_copy: pd.DataFrame = df.copy()
        df_copy["start_station_coords"] = list(
            zip(df_split["start_lat"], df_split["start_long"])
        )
        df_copy["end_station_coords"] = list(
            zip(df_split["end_lat"], df_split["end_long"])
        )
        return df_copy

    def save(self, filepath: str) -> None:
        """
        Saves the fitted scaler to the specified file path using joblib.

        This allows the scaler to be persisted and reused for future transformations.

        Args:
            filepath (str): The path where the scaler will be saved.

        Raises:
            IOError: If the scaler cannot be saved to the specified filepath.
        """
        try:
            joblib.dump(self.scaler, filepath)
        except Exception as e:
            raise IOError(f"Failed to save scaler to {filepath}: {e}")

    def load(self, filepath: str) -> None:
        """
        Loads a scaler from the specified file path using joblib.

        This method replaces the current scaler with the one loaded from the file.

        Args:
            filepath (str): The path from where the scaler will be loaded.

        Raises:
            IOError: If the scaler cannot be loaded from the specified filepath.
        """
        try:
            self.scaler = joblib.load(filepath)
        except Exception as e:
            raise IOError(f"Failed to load scaler from {filepath}: {e}")


def set_stations_df() -> pd.DataFrame:
    """
    Load and transform station data from a CSV file into a structured DataFrame.

    This function performs the following steps:
    1. **Load Data**: Reads station data from a CSV file specified by `Config.STATIONS_CSV_PATH`.
    2. **Sort Data**: Sorts the DataFrame based on the `id` column to ensure consistent ordering.
    3. **Create Unmapped ID**: Creates a new column `id_unmapped` as a copy of the original `id` to preserve the original identifiers.
    4. **Reassign IDs**: Reassigns the `id` column to be a sequential index starting from 0, replacing the original IDs.
    5. **Cluster Stations**: Applies K-Means clustering to the longitude (`long`) and latitude (`lat`) coordinates to categorize stations into 3 clusters. The resulting cluster labels are incremented by 1 to start from 1 instead of 0.
    6. **Parse Installation Dates**: Converts the `installation_date` column from string format to `datetime` objects for easier date manipulation and analysis.

    Returns:
        pd.DataFrame: A transformed DataFrame containing the following columns:
            - **id** (`int`): The new sequential identifier for each station.
            - **id_unmapped** (`int`): The original identifier from the CSV, preserved for reference.
            - **name** (`str`): The name of the station.
            - **lat** (`float`): The latitude coordinate of the station.
            - **long** (`float`): The longitude coordinate of the station.
            - **dock_count** (`int`): The number of docks available at the station.
            - **city** (`str`): The city where the station is located.
            - **installation_date** (`datetime`): The date when the station was installed.
            - **cluster** (`int`): The cluster number assigned to the station based on its geographic location.
    """

    stations_df = pd.read_csv(Config.STATIONS_CSV_PATH)
    stations_df.sort_values("id", inplace=True)
    stations_df["id_unmapped"] = stations_df["id"].copy()
    stations_df["id"] = list(stations_df.index)
    kmeans = KMeans(n_clusters=3, random_state=0)
    stations_df["cluster"] = kmeans.fit_predict(stations_df[["long", "lat"]].values)
    stations_df["cluster"] += 1
    stations_df["installation_date"] = pd.to_datetime(
        stations_df["installation_date"], format="%m/%d/%Y"
    )

    return stations_df[
        [
            "id",
            "id_unmapped",
            "name",
            "lat",
            "long",
            "dock_count",
            "city",
            "installation_date",
            "cluster",
        ]
    ]


def set_trips_df(split: str | None = None, **kwargs) -> pd.DataFrame:
    """
    Load and transform trip data from a CSV file into a structured DataFrame with optional filtering and splitting.

    This function performs the following steps:
    1. **Load Data**: Reads trip data from a CSV file specified by `Config.TRIPS_CSV_PATH`.
    2. **Data Type Conversion**: Converts the `zip_code` column to string type to ensure consistency.
    3. **Parse Dates**: Converts the `start_date` and `end_date` columns from string format to `datetime` objects for easier date and time manipulation.
    4. **Sort and Reset Index**: Sorts the DataFrame based on the `start_date` column and resets the index to maintain sequential ordering.
    5. **Map Station IDs**:
        - Calls `set_stations_df()` to obtain the stations DataFrame.
        - Creates a mapping from the original `id_unmapped` to the new `id`.
        - Replaces the `start_station_id` and `end_station_id` in the trips DataFrame with the new sequential IDs.
    6. **Normalize Time**:
        - Applies the `normalize_hour` function to the `start_date` and `end_date` columns to create `start_time_norm` and `end_time_norm`, representing normalized time values between 0 and 1.
    7. **Calculate Duration**: Computes the trip duration in seconds by calculating the difference between `end_date` and `start_date`.
    8. **Apply Filters**: Applies optional filtering based on parameters provided via `**kwargs`:
        - `min_start_date` and `max_start_date`: Filters trips that start within the specified date range.
        - `min_start_time` and `max_start_time`: Filters trips that start within the specified normalized time range.
        - `min_duration` and `max_duration`: Filters trips based on the trip duration.
    9. **Split Data**: Splits the DataFrame into training, validation, and testing sets with proportions 70%, 15%, and 15% respectively by adding a `split` column.
    10. **Filter by Split**: If the `split` parameter is provided (`'train'`, `'val'`, or `'test'`), the DataFrame is filtered to include only the specified split.

    Parameters:
        split (str, optional): If specified, filters the DataFrame to include only the given split (`'train'`, `'val'`, `'test'`). Defaults to `None`.
        **kwargs: Additional filtering parameters:
            - `min_start_date` (str, optional): Include trips that start on or after this datetime (`'%m/%d/%Y %H:%M:%S'`).
            - `max_start_date` (str, optional): Include trips that start on or before this datetime (`'%m/%d/%Y %H:%M:%S'`).
            - `min_start_time` (float, optional): Include trips that start on or after this normalized time (0-1).
            - `max_start_time` (float, optional): Include trips that start on or before this normalized time (0-1).
            - `min_duration` (float, optional): Include trips with duration greater than or equal to this value (seconds).
            - `max_duration` (float, optional): Include trips with duration less than or equal to this value (seconds).

    Returns:
        pd.DataFrame: A transformed DataFrame containing the following columns:
            - **start_date** (`datetime`): The start date and time of the trip.
            - **end_date** (`datetime`): The end date and time of the trip.
            - **start_station_id** (`int`): The sequential identifier of the start station.
            - **end_station_id** (`int`): The sequential identifier of the end station.
            - **start_time_norm** (`float`): The normalized start time of the trip (0-1).
            - **end_time_norm** (`float`): The normalized end time of the trip (0-1).
            - **duration** (`float`): The duration of the trip in seconds.
            - **start_station** (`str`): The name of the start station.
            - **end_station** (`str`): The name of the end station.
            - **split** (`str`): The data split category (`'train'`, `'val'`, `'test'`).
    """

    trips_df = pd.read_csv(Config.TRIPS_CSV_PATH)
    trips_df["zip_code"] = trips_df["zip_code"].astype(str)
    trips_df["start_date"] = pd.to_datetime(
        trips_df["start_date"], format="%m/%d/%Y %H:%M"
    )
    trips_df["end_date"] = pd.to_datetime(trips_df["end_date"], format="%m/%d/%Y %H:%M")

    trips_df.sort_values(by="start_date", inplace=True)
    trips_df.reset_index(drop=True, inplace=True)

    stations_df = set_stations_df()
    ids_mapping = {
        id: i for i, id in enumerate(stations_df["id_unmapped"].values.tolist())
    }

    trips_df["start_station_id"] = trips_df["start_station_id"].replace(ids_mapping)
    trips_df["end_station_id"] = trips_df["end_station_id"].replace(ids_mapping)
    trips_df["start_time_norm"] = trips_df["start_date"].apply(normalize_hour)
    trips_df["end_time_norm"] = trips_df["end_date"].apply(normalize_hour)
    trips_df["duration"] = (
        trips_df["end_date"] - trips_df["start_date"]
    ).dt.total_seconds()

    # Apply filters from kwargs
    if "min_start_date" in kwargs:
        trips_df = trips_df[
            trips_df["start_date"]
            >= pd.to_datetime(kwargs["min_start_date"], format="%m/%d/%Y %H:%M")
        ]
    if "max_start_date" in kwargs:
        trips_df = trips_df[
            trips_df["start_date"]
            <= pd.to_datetime(kwargs["max_start_date"], format="%m/%d/%Y %H:%M")
        ]
    if "min_start_time" in kwargs:
        trips_df = trips_df[trips_df["start_time_norm"] >= kwargs["min_start_time"]]
    if "max_start_time" in kwargs:
        trips_df = trips_df[trips_df["start_time_norm"] <= kwargs["max_start_time"]]
    if "min_duration" in kwargs:
        trips_df = trips_df[trips_df["duration"] >= kwargs["min_duration"]]
    if "max_duration" in kwargs:
        trips_df = trips_df[trips_df["duration"] <= kwargs["max_duration"]]

    # Splitting into train, val, and test
    total_len = len(trips_df)
    train_end = int(0.7 * total_len)
    val_end = int(0.85 * total_len)

    trips_df["split"] = "train"
    trips_df.iloc[train_end:val_end, trips_df.columns.get_loc("split")] = "val"
    trips_df.iloc[val_end:, trips_df.columns.get_loc("split")] = "test"

    if split is not None:
        trips_df = trips_df[trips_df["split"] == split]

    return trips_df
