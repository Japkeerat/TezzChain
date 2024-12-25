"""
Dataclass to hold the configuration of Tezzchain. This helps in providing default values
for the configuration parameters and also helps in validating the configuration parameters.
"""

from typing import Literal
from dataclasses import dataclass


@dataclass
class GlobalConfiguration:
    allow_tezzchain_telemetry: Literal["all", "error", "none"] = "all"
    allow_client_telemetry: bool = False
    llm_provider: Literal["ollama"] = "ollama"
    vectordb_provider: Literal["chromadb"] = "chromadb"
    embedding_provider: Literal["ollama"] = "ollama"
    chunking_algorithm: Literal["basic"] = "basic"
