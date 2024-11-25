import json
from datetime import datetime


def normalize_time(dt: datetime) -> float:
    """
    Normalize the hour and minute of a datetime object to a value between 0 and 1.

    Parameters:
    dt (datetime): The datetime object to normalize.

    Returns:
    float: The normalized hour and minute.
    """
    return dt.hour / 24.0 + dt.minute / 1440.0


def read_json_to_dict(path):
    """
    Read a JSON file and convert it to a dictionary.

    path (str): The file path to the JSON file.

    dict: The dictionary representation of the JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)

def save_dict_to_json(data, path):
    """
    Save a dictionary to a JSON file.

    Parameters:
    data (dict): The dictionary to save.
    path (str): The file path to save the JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f)