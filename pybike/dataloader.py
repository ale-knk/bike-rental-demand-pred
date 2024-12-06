from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from pybike.preprocessing import set_trips_df
from pybike.trip import CoordinateNormalizer, Trips


class TripsGen:
    """
    A generator class for creating sequences of trips.

    This class takes a collection of trips and generates overlapping sequences
    of a specified length. It is useful for preparing data for sequence models
    where each input sample consists of a sequence of trips.

    Attributes:
        trips (Trips): The complete collection of trips.
        seq_len (int): The length of each trip sequence. Defaults to 15.
    """

    def __init__(self, trips: Trips, seq_len: int = 15) -> None:
        """
        Initializes the TripsGen with a collection of trips and a sequence length.

        Args:
            trips (Trips): The complete collection of trips.
            seq_len (int, optional): The length of each trip sequence. Defaults to 15.
        """
        self.seq_len: int = seq_len
        self.trips: Trips = trips

    def __len__(self) -> int:
        """
        Returns the number of possible trip sequences.

        The number is calculated as the total number of trips minus the sequence length plus one.

        Returns:
            int: The number of trip sequences.
        """
        return len(self.trips) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Trips:
        """
        Retrieves a trip sequence starting at the specified index.

        Args:
            idx (int): The starting index for the trip sequence.

        Returns:
            Trips: A Trips object containing the sequence of trips.
        """
        trip_seq = self.trips[idx : idx + self.seq_len]
        return Trips(trips=trip_seq)


class TripsDataset(Dataset):
    """
    A PyTorch Dataset for handling trip sequences.

    This dataset wraps around a TripsGen instance to provide input-target pairs
    suitable for training sequence models. Each input consists of features derived
    from a sequence of trips, and each target consists of the attributes of the
    subsequent trip.

    Attributes:
        trips_gen (TripsGen): An instance of TripsGen to generate trip sequences.
    """

    def __init__(self, trips_gen: TripsGen) -> None:
        """
        Initializes the TripsDataset with a TripsGen instance.

        Args:
            trips_gen (TripsGen): An instance of TripsGen to generate trip sequences.
        """
        self.trips_gen: TripsGen = trips_gen

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of trip sequences available.
        """
        return len(self.trips_gen)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Retrieves the input-target pair for the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: A tuple containing:
                - inputs (Dict[str, torch.Tensor]): A dictionary of input tensors.
                - targets (Dict[str, Any]): A dictionary of target values.
        """
        trip_sequence: Trips = self.trips_gen[idx]
        input_trips: Trips = Trips(trips=trip_sequence.trips[:-1])
        target_trip = trip_sequence.trips[-1]

        inputs: Dict[str, torch.Tensor] = {
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

        targets: Dict[str, Any] = {
            "start_station_id": target_trip.start_station_id,
            "start_time": torch.tensor(target_trip.start_time_norm).unsqueeze(-1),
            "end_station_id": target_trip.end_station_id,
            "end_time": torch.tensor(target_trip.end_time_norm).unsqueeze(-1),
        }

        return inputs, targets

    def _get_tensor(self, trips: Trips, method_name: str) -> torch.Tensor:
        """
        Retrieves a tensor by calling a specified method on the Trips object.

        This is a helper method assumed to exist for converting trip attributes
        into tensors. The actual implementation should handle the method invocation.

        Args:
            trips (Trips): The Trips object containing trip data.
            method_name (str): The name of the method to call on the Trips object.

        Returns:
            torch.Tensor: The resulting tensor from the method call.
        """
        method = getattr(trips, method_name)
        return method()


def create_dataloader(
    trips: Trips, seq_len: int = 15, batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for a given set of trips.

    This function initializes a TripsGen instance with the provided trips and sequence length,
    wraps it in a TripsDataset, and then creates a DataLoader with the specified batch size
    and shuffle option.

    Args:
        trips (Trips): The complete collection of trips.
        seq_len (int, optional): The length of each trip sequence. Defaults to 15.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.

    Returns:
        DataLoader: A PyTorch DataLoader providing batches of trip sequences.
    """
    trips_gen = TripsGen(trips, seq_len)
    dataset = TripsDataset(trips_gen)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def set_dataloaders(
    splits: List[str],
    scaler: CoordinateNormalizer,
    seq_len: int = 15,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader | List[DataLoader]:
    """
    Sets up DataLoaders for multiple data splits.

    This function processes each split by loading the corresponding trips data,
    normalizing coordinates if necessary, and creating a DataLoader using the
    `create_dataloader` function. It ensures that a scaler is fitted on the training
    set and applied to validation and test sets.

    Args:
        splits (List[str]): A list of dataset splits (e.g., ['train', 'validation', 'test']).
        seq_len (int, optional): The length of each trip sequence. Defaults to 15.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        scaler (CoordinateNormalizer | None, optional): An instance of CoordinateNormalizer.
            If `None`, a new scaler will be created and fitted on the training set.
            For validation and test sets, a scaler must be provided. Defaults to None.

    Returns:
        Union[DataLoader, List[DataLoader]]: A single DataLoader if only one split is provided;
            otherwise, a list of DataLoaders corresponding to each split.

    Raises:
        Exception: If a scaler is not provided for validation and test sets.
    """
    dataloaders: List[DataLoader] = []
    for split in splits:
        trips_df = set_trips_df(split=split).iloc[:1000]
        trips = Trips.from_df(trips_df)
        if split == "train":
            trips.normalize_coords(scaler, fit=True)
        else:
            trips.normalize_coords(scaler, fit=False)
        dataloader = create_dataloader(trips, seq_len, batch_size, shuffle)
        dataloaders.append(dataloader)

    if len(dataloaders) == 1:
        return dataloaders[0]
    return dataloaders
