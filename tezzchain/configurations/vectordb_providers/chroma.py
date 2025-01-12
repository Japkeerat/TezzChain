from pathlib import Path
from dataclasses import dataclass


@dataclass
class ChromaDB:
    host: str = "http://localhost:8000"
    port: int = 8000
    db_path: str | Path = None
    tenant_id: str | None = None
    collection_name: str = "default"
    allow_reset: bool = False
    n_results: int = 5  # Number of results to fetch from the database
    max_threads: int = 40
