import os
from datetime import datetime

import torch
from bike_rental_core.demand_pred.model import BikeRoutePredictor
from bike_rental_core.models.stations import Stations
from bike_rental_core.models.trips import TripSeq

from pybike import utils
from pybike.preprocessing import set_stations_df, set_trips_df, CoordinateNormalizer
from pybike.utils import read_json_to_dict
from pybike.trip import Trips
from pybike.dataloader import set_dataloaders

class BikeDemandAgent:
    def __init__(
        self,
        base_dir: str,
        start_date: datetime,
        end_date: datetime,
        delta: int = 15,
        alpha: int = 90,
    ):
        self.base_dir = base_dir
        self.start_date = start_date
        self.end_date = end_date
        self.delta = delta  
        self.alpha = alpha

    def _setup(self):
        self._setup_model()
        self._setup_data()
    
    def _setup_model(self):
        self.model_path = os.path.join(self.base_dir, "model.pth")
        self.config = utils.read_json_to_dict(
            os.path.join(self.base_dir, "config.json")
        )
        self.model = BikeRoutePredictor(**self.config["model"])
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.seq_len = self.config["dataloader"]["seq_len"]

    def _setup_data(self):
        self.scaler = CoordinateNormalizer()
        self.scaler.load(os.path.join(self.base_dir, "scaler.pkl"))

        self.trips_df = set_trips_df(split="test")
        self.trips = Trips.from_df(self.trips_df)
        self.trips.normalize_coords(self.scaler)
        
    def predict(self, trips_seq: TripSeq, future_steps: int = 1):
        start_stations_preds = []
        start_times_preds = []
        end_stations_preds = []
        end_times_preds = []

        current_data = {
            "start_stations": trips_seq.get_start_stations(add_batch_dim=True),
            "start_coords": trips_seq.get_start_coords(add_batch_dim=True),
            "start_times": trips_seq.get_start_times(add_batch_dim=True),
            "end_stations": trips_seq.get_end_stations(add_batch_dim=True),
            "end_coords": trips_seq.get_end_coords(add_batch_dim=True),
            "end_times": trips_seq.get_end_times(add_batch_dim=True),
        }

        for _ in range(future_steps):
            start_stations = current_data["start_stations"]
            start_coords = current_data["start_coords"]
            start_times = current_data["start_times"]
            end_stations = current_data["end_stations"]
            end_coords = current_data["end_coords"]
            end_times = current_data["end_times"]

            with torch.no_grad():
                start_station_pred, start_time_pred, end_station_pred, end_time_pred = (
                    self.model(
                        start_stations,
                        start_coords,
                        start_times,
                        end_stations,
                        end_coords,
                        end_times,
                    )
                )

            # Append predictions to the lists
            start_stations_preds.append(start_station_pred.argmax().item())
            start_times_preds.append(start_time_pred.item())
            end_stations_preds.append(end_station_pred.argmax().item())
            end_times_preds.append(end_time_pred.item())

            current_data = {
                "start_stations": torch.cat(
                    (
                        start_stations[:, 1:],
                        torch.tensor(start_stations_preds).unsqueeze(0)[:, -1:],
                    ),
                    dim=1,
                ),
                "start_coords": torch.cat(
                    (
                        start_coords[:, 1:, :],
                        torch.tensor(
                            self.stations.get_coords(
                                station_id=start_stations_preds[-1]
                            )
                        ).unsqueeze(0),
                    ),
                    dim=1,
                ),
                "start_times": torch.cat(
                    (
                        start_times[:, 1:, :],
                        torch.tensor(start_times_preds)
                        .unsqueeze(0)
                        .unsqueeze(2)[:, -1:, :],
                    ),
                    dim=1,
                ),
                "end_stations": torch.cat(
                    (
                        end_stations[:, 1:],
                        torch.tensor(end_stations_preds).unsqueeze(0)[:, -1:],
                    ),
                    dim=1,
                ),
                "end_coords": torch.cat(
                    (
                        end_coords[:, 1:, :],
                        torch.tensor(
                            self.stations.get_coords(station_id=end_stations_preds[-1])
                        ).unsqueeze(0),
                    ),
                    dim=1,
                ),
                "end_times": torch.cat(
                    (
                        end_times[:, 1:, :],
                        torch.tensor(end_times_preds)
                        .unsqueeze(0)
                        .unsqueeze(2)[:, -1:, :],
                    ),
                    dim=1,
                ),
            }

        return (
            start_stations_preds,
            start_times_preds,
            end_stations_preds,
            end_times_preds,
        )

    def run(self):
        """
        Esta función debe simular el uso de un agente que utilice un modelo entrenado para predecir la demanda de bicicletas en una ciudad.
        Utilizando el self.dataloader, debe ir generando secuencias de seq_len trips y predecir los siguientes trips hasta 
        """
        self._setup()

        k = self.seq_len
        current_date = self.trips[k].start_date
        current_trips = self.trips[:k]

        while True:

            start_stations_preds, start_times_preds, end_stations_preds, end_times_preds = self.predict(trips_seq)

            trips_list = self.trips[]
            trips_seq = self.trips.get_seq(
                start_date=self.start_date, end_date=self.end_date, delta=self.delta
            )
            trips_seq.normalize_coords(self.scaler)

            start_stations_preds, start_times_preds, end_stations_preds, end_times_preds = self.predict(trips_seq)

            print("Predicted start stations:", start_stations_preds)
            print("Predicted start times:", start_times_preds)
            print("Predicted end stations:", end_stations_preds)
            print("Predicted end times:", end_times_preds)

            self.start_date = self.start_date + timedelta(minutes=self.delta)
            self.end_date = self.end_date + timedelta(minutes=self.delta)