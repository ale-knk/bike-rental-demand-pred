import os
import torch
import pandas as pd

from pybike import utils
from bike_rental_core.demand_pred.model import BikeRoutePredictor
from pybike.data.utils import load_and_preprocess_stations_df
from bike_rental_core.models.stations import Stations
from bike_rental_core.models.trips import TripSeq

class BikeDemandAgent:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, "model.pth")
        self.config = utils.load_json_to_dict(os.path.join(self.model_dir, "config.json"))
        self.stations_df = load_and_preprocess_stations_df()
        self.stations = Stations()
        self.model = BikeRoutePredictor(
            n_stations=self.config["n_stations"],
            d_model=self.config["d_model"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"]
        )
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
                
    def predict(self, 
                trips_seq: TripSeq, 
                future_steps: int = 1):

        start_stations_preds = []
        start_times_preds = []
        end_stations_preds = []
        end_times_preds = []

        current_data = {
            "start_stations":trips_seq.get_start_stations(add_batch_dim=True),
            "start_coords":trips_seq.get_start_coords(add_batch_dim=True),
            "start_times":trips_seq.get_start_times(add_batch_dim=True),
            "end_stations":trips_seq.get_end_stations(add_batch_dim=True),
            "end_coords":trips_seq.get_end_coords(add_batch_dim=True),
            "end_times":trips_seq.get_end_times(add_batch_dim=True),         
        }

        for _ in range(future_steps):

            start_stations = current_data['start_stations']
            start_coords = current_data['start_coords']
            start_times = current_data['start_times']
            end_stations = current_data['end_stations']
            end_coords = current_data['end_coords']
            end_times = current_data['end_times']

            with torch.no_grad():
                start_station_pred, start_time_pred, end_station_pred, end_time_pred = self.model(
                    start_stations, start_coords, start_times, end_stations, end_coords, end_times
                )

            # Append predictions to the lists
            start_stations_preds.append(start_station_pred.argmax().item())
            start_times_preds.append(start_time_pred.item())
            end_stations_preds.append(end_station_pred.argmax().item())
            end_times_preds.append(end_time_pred.item())

            current_data = {
                'start_stations': torch.cat((start_stations[:,1:], torch.tensor(start_stations_preds).unsqueeze(0)[:,-1:]), dim=1),
                'start_coords': torch.cat((start_coords[:,1:,:],torch.tensor(self.stations.get_coords(station_id=start_stations_preds[-1])).unsqueeze(0)), dim=1),
                'start_times': torch.cat((start_times[:,1:,:], torch.tensor(start_times_preds).unsqueeze(0).unsqueeze(2)[:,-1:,:]), dim=1),
                'end_stations': torch.cat((end_stations[:,1:], torch.tensor(end_stations_preds).unsqueeze(0)[:,-1:]), dim=1),
                'end_coords': torch.cat((end_coords[:,1:,:],torch.tensor(self.stations.get_coords(station_id=end_stations_preds[-1])).unsqueeze(0)), dim=1),
                'end_times': torch.cat((end_times[:,1:,:], torch.tensor(end_times_preds).unsqueeze(0).unsqueeze(2)[:,-1:,:]), dim=1),
            }

        return start_stations_preds, start_times_preds, end_stations_preds, end_times_preds

