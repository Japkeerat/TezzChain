"""
Implements the OllamaEmbedding class, which is a wrapper around the Ollama API.

It inherits from BaseEmbedding which is a base class for all embedding providers in TezzChain.
"""

import logging
from typing import Optional

from ollama import Client, Options

from tezzchain.embedding.base import BaseEmbedding


logger = logging.getLogger("tezzchain")


__tezzchain_ollama_embedding_version__ = "0.0.1"
__ollama_server_supported_version__ = "0.5.4"


class OllamaEmbedding(BaseEmbedding):

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OllamaEmbedding, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model: Optional[str] = "qwen2:0.5b-instruct",
        host: Optional[str] = "http://localhost:11434",
        **kwargs
    ):
        if not hasattr(self, "initialized"):
            self.model = model
            self.host = host
            self.client = Client(host=self.host)
            self.kwargs = kwargs
            self.initialized = True

    def __get_options(self):
        options = Options(**self.kwargs)
        return options

    def embed(self, text: str) -> list[float]:
        options = self.__get_options()
        return self.client.embed(input=text, model=self.model, options=options)

    def get_model(self) -> str:
        """
        Returns the name of the current model.

        :return: The name of the current model.
        """
        return self.model

    def get_client(self) -> Client:
        """
        Returns the Ollama API client instance.

        :return: The Ollama API client instance.
        """
        return self.client
