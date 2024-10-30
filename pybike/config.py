import os
from pathlib import Path
from dotenv import load_dotenv



load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

class Config:
    STATIONS_CSV_PATH = os.getenv("STATIONS_CSV_PATH", BASE_DIR.parent / 'data/stations.csv')
    TRIPS_CSV_PATH = os.getenv("TRIPS_CSV_PATH", BASE_DIR.parent / 'data/trips.csv')
    STATUS_CSV_PATH = os.getenv("STATUS_CSV_PATH", BASE_DIR.parent / 'data/status.csv')
