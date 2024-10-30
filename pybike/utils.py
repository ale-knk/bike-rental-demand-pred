def normalize_time(dt):
    """
    Normalize the hour and minute of a datetime object to a value between 0 and 1.

    Parameters:
    dt (datetime): The datetime object to normalize.

    Returns:
    float: The normalized hour and minute.
    """
    return dt.hour / 24.0 + dt.minute / 1440.0
