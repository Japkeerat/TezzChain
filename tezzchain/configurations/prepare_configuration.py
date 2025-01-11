"""
This module is responsible for preparing the configuration for the application.
It reads the user provided configuration file (either YAML or JSON), if provided, and fills in default
values for the configuration that are not provided from the various dataclasses defined in the application.
This helps in prevalidating the configuration parameters and ensuring a complete configuraiton is 
provided to the necessary components of the application without worrying about missing configurations.
"""

from pathlib import Path
from typing import Optional
from dataclasses import asdict

from tezzchain.utils import read_file_intelligently
from tezzchain.configurations.chunkers import chunk_config
from tezzchain.configurations.llm_providers import llm_configurators
from tezzchain.configurations.vectordb_providers import vectordb_config
from tezzchain.configurations.global_configuration import GlobalConfiguration
from tezzchain.configurations.embedding_providers import embedding_configurators


class TezzchainConfiguration:
    def __init__(self, config_file: Optional[Path] = None):
        if config_file:
            configuration = read_file_intelligently(config_file)
        else:
            configuration = {"APP": dict()}
        global_config = self.__prepare_global_configuration(configuration)
        client_telemetry_config = self.__prepare_client_telemetry_configuration(
            configuration.get("CLIENT-TELEMETRY", dict()),
            global_config["allow_client_telemetry"],
        )
        llm_config = self.__prepare_llm_configuration(
            configuration.get("LLM", dict()), global_config["llm_provider"]
        )
        embedding_config = self.__prepare_embedding_configuration(
            configuration.get("EMBEDDING", dict()), global_config["embedding_provider"]
        )
        chunk_config = self.__prepare_chunk_config(
            configuration.get("CHUNK", dict()), global_config["chunking_algorithm"]
        )
        vectordb_config = self.__prepare_vectordb_config(
            configuration.get("VECTORDB", dict()), global_config["vectordb_provider"]
        )
        self.config = self.__merge_configurations(
            global_config,
            client_telemetry_config,
            llm_config,
            embedding_config,
            chunk_config,
            vectordb_config,
        )

    def __prepare_global_configuration(self, configuration: dict) -> dict:
        """
        Configurations that are important for the overall functionality of Tezzchain.
        """
        valid_keys = {
            field.name for field in GlobalConfiguration.__dataclass_fields__.values()
        }
        config_params = {k: v for k, v in configuration.items() if k in valid_keys}
        return asdict(GlobalConfiguration(**config_params))

    def __prepare_client_telemetry_configuration(
        self, client_telemetry_config: dict, allow_client_telemetry: bool
    ) -> dict:
        """
        If the user of this library wants to collect telemetry data about the usage of Tezzchain, they'll
        need to configure it in the config file.
        """
        if allow_client_telemetry:
            if (
                "api" not in client_telemetry_config
                or "host" not in client_telemetry_config
            ):
                raise ValueError(
                    """Client telemetry is enabled but either 'api' or 'host' is not provided 
                    in the configuration."""
                )
            return client_telemetry_config
        else:
            return {"api": None, "host": None}

    def __prepare_llm_configuration(
        self, configuration: dict, llm_provider: str
    ) -> dict:
        """
        Configurations that are specific to the LLM provider that the user has chosen.
        """
        valid_keys = {
            field.name
            for field in llm_configurators[llm_provider].__dataclass_fields__.values()
        }
        config_params = {k: v for k, v in configuration.items() if k in valid_keys}
        return asdict(llm_configurators[llm_provider](**config_params))

    def __prepare_embedding_configuration(
        self, configuration: dict, embedding_provider: str
    ) -> dict:
        """
        Configurations that are specific to the Embedding provider that has been chosen
        """
        valid_keys = {
            field.name
            for field in embedding_configurators[
                embedding_provider
            ].__dataclass_fields__.values()
        }
        config_params = {k: v for k, v in configuration.items() if k in valid_keys}
        return asdict(embedding_configurators[embedding_provider](**config_params))

    def __prepare_chunk_config(
        self, configuration: dict, chunking_algorithm: str
    ) -> dict:
        """
        Configurations that are specific to the Chunking algorithm that has been chosen
        """
        valid_keys = {
            field.name
            for field in chunk_config[chunking_algorithm].__dataclass_fields__.values()
        }
        config_params = {k: v for k, v in configuration.items() if k in valid_keys}
        return asdict(chunk_config[chunking_algorithm](**config_params))

    def __prepare_vectordb_config(
        self, configuration: dict, vectordb_provider: str
    ) -> dict:
        """
        Configurations that are specific to the VectorDB provider that has been chosen
        """
        valid_keys = {
            field.name
            for field in vectordb_config[
                vectordb_provider
            ].__dataclass_fields__.values()
        }
        config_params = {k: v for k, v in configuration.items() if k in valid_keys}
        return asdict(vectordb_config[vectordb_provider](**config_params))

    def __merge_configurations(
        self,
        global_config: dict,
        client_telemetry_config: dict,
        llm_config: dict,
        embedding_config: dict,
        chunk_config: dict,
        vectordb_config: dict,
    ) -> dict:
        """
        Prepares the final configuration dictionary to maintain the configuration of the application.
        """
        config = {
            "APP": global_config,
            "CLIENT-TELEMETRY": client_telemetry_config,
            "LLM": llm_config,
            "EMBEDDING": embedding_config,
            "CHUNK": chunk_config,
            "VECTORDB": vectordb_config,
        }
        return config

    def get_config(self):
        return self.config
