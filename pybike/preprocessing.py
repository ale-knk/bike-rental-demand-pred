from .config import Config
import pandas as pd
from sklearn.cluster import KMeans
import polars as pl

def normalize_hour(dt):
    """
    Normalize the hour and minute of a datetime object to a value between 0 and 1.
    
    Parameters:
    dt (datetime): The datetime object to normalize.

    Returns:
    float: The normalized hour and minute.
    """
    return dt.hour / 24.0 + dt.minute / 1440.0

def set_stations_df():
    """
    Read a CSV file containing station data, perform some transformations and return a DataFrame.
    
    Parameters:
    stations_df_path (str): The path to the CSV file.

    Returns:
    DataFrame: The transformed DataFrame with the following columns:
        - id
        - id_unmapped
        - name
        - lat
        - long
        - dock_count
        - city
        - installation_date
        - cluster
    """

    stations_df = pd.read_csv(Config.STATIONS_CSV_PATH)
    stations_df.sort_values("id", inplace=True)
    stations_df["id_unmapped"] = stations_df["id"].copy()
    stations_df["id"] = list(stations_df.index)
    kmeans = KMeans(n_clusters=3, random_state=0)
    stations_df['cluster'] = kmeans.fit_predict(stations_df[["long", "lat"]].values)
    stations_df["cluster"] += 1
    stations_df['installation_date'] = pd.to_datetime(stations_df['installation_date'], format='%m/%d/%Y')
    
    return stations_df[["id","id_unmapped","name","lat","long","dock_count","city","installation_date","cluster"]]

def set_trips_df(split: str | None = None, **kwargs):
    """
    Read a CSV file containing trip data, perform some transformations and return a DataFrame.
    
    Parameters:
    split (str, optional): If specified, filter the DataFrame by the given split ('train', 'val', 'test').
    **kwargs: Additional filtering parameters:
        - min_start_date (str, optional): Filter trips that start on or after this datetime ('%m/%d/%Y %H:%M:%S').
        - max_start_date (str, optional): Filter trips that start on or before this datetime ('%m/%d/%Y %H:%M:%S').
        - min_start_time (float, optional): Filter trips that start on or after this normalized time (0-1).
        - max_start_time (float, optional): Filter trips that start on or before this normalized time (0-1).
        - min_duration (float, optional): Filter trips with duration greater than or equal to this value (seconds).
        - max_duration (float, optional): Filter trips with duration less than or equal to this value (seconds).

    Returns:
    DataFrame: The transformed DataFrame with the following columns:
        - start_date
        - end_date
        - start_station_id
        - end_station_id
        - start_time_norm
        - end_time_norm
        - duration
        - start_station
        - end_station
        - split
    """
    
    trips_df = pd.read_csv(Config.TRIPS_CSV_PATH)
    trips_df["zip_code"] = trips_df["zip_code"].astype(str)
    trips_df['start_date'] = pd.to_datetime(trips_df['start_date'], format='%m/%d/%Y %H:%M')
    trips_df['end_date'] = pd.to_datetime(trips_df['end_date'], format='%m/%d/%Y %H:%M')

    trips_df.sort_values(by="start_date", inplace=True)
    trips_df.reset_index(drop=True, inplace=True) 

    stations_df = set_stations_df()
    ids_mapping = {id:i for i,id in enumerate(stations_df["id_unmapped"].values.tolist())}

    trips_df["start_station_id"] = trips_df["start_station_id"].replace(ids_mapping)
    trips_df["end_station_id"] = trips_df["end_station_id"].replace(ids_mapping)
    trips_df['start_time_norm'] = trips_df['start_date'].apply(normalize_hour)
    trips_df['end_time_norm'] = trips_df['end_date'].apply(normalize_hour)
    trips_df['duration'] = (trips_df['end_date'] - trips_df['start_date']).dt.total_seconds()

    # Apply filters from kwargs
    if "min_start_date" in kwargs:
        trips_df = trips_df[trips_df['start_date'] >= pd.to_datetime(kwargs["min_start_date"], format='%m/%d/%Y %H:%M')]
    if "max_start_date" in kwargs:
        trips_df = trips_df[trips_df['start_date'] <= pd.to_datetime(kwargs["max_start_date"], format='%m/%d/%Y %H:%M')]
    if "min_start_time" in kwargs:
        trips_df = trips_df[trips_df['start_time_norm'] >= kwargs["min_start_time"]]
    if "max_start_time" in kwargs:
        trips_df = trips_df[trips_df['start_time_norm'] <= kwargs["max_start_time"]]
    if "min_duration" in kwargs:
        trips_df = trips_df[trips_df['duration'] >= kwargs["min_duration"]]
    if "max_duration" in kwargs:
        trips_df = trips_df[trips_df['duration'] <= kwargs["max_duration"]]

    # Splitting into train, val, and test
    total_len = len(trips_df)
    train_end = int(0.7 * total_len)
    val_end = int(0.85 * total_len)

    trips_df['split'] = 'train'
    trips_df.iloc[train_end:val_end, trips_df.columns.get_loc('split')] = 'val'
    trips_df.iloc[val_end:, trips_df.columns.get_loc('split')] = 'test'

    if split is not None:
        trips_df = trips_df[trips_df["split"] == split]
    
    return trips_df

def set_status_df():
    """
    Read a CSV file containing status data, perform some transformations and return a DataFrame.
    
    Parameters:
    status_df_path (str): The path to the CSV file.

    Returns:
    DataFrame: The transformed DataFrame with the following columns:
        - station_id
        - bikes_available
        - docks_available
        - time
    """
    
    status_df = pl.read_csv(Config.STATUS_CSV_PATH)
    df_slash = status_df.filter(pl.col("time").str.contains(r"\d{4}/\d{2}/\d{2}"))
    df_dash = status_df.filter(pl.col("time").str.contains(r"\d{4}-\d{2}-\d{2}"))
    df_slash = df_slash.with_columns(
        pl.col("time").str.strptime(pl.Datetime, format="%Y/%m/%d %H:%M:%S").alias("time")
    )
    df_dash = df_dash.with_columns(
        pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S").alias("time")
    )
    df_final = pl.concat([df_slash, df_dash]).sort("time")

    status_df = df_final.with_columns(
        pl.col("time").dt.truncate("5m").alias("time")
    ).group_by("time").agg(pl.col("*").first())

    status_df = status_df.to_pandas()

    stations_df = set_stations_df()
    stations = list(db["stations"].find({}))
    ids_mapping = {id:i for i,id in enumerate(stations_df["id_unmapped"].values.tolist())}
    mongo_id_mapping = {station['id']: station['_id'] for station in stations}
    status_df["station_id"] = status_df["station_id"].replace(ids_mapping)
    status_df['station_mongoid'] = status_df['station_id'].map(mongo_id_mapping)

    return status_df

