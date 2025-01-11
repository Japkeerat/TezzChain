from pathlib import Path
from dataclasses import dataclass


@dataclass
class ChromaDB:
    host: str = "http://localhost:8000"
    db_path: str | Path = None
    collection_name: str = "default"
    allow_reset: bool = False
    n_results: int = 5  # Number of results to fetch from the database
