"""
Implements the OllamaLLM class, which is a wrapper around the Ollama API.

It inherits from BaseLLM which is a base class for all LLMs in TezzChain. It is designed as a singleton 
class.
"""

import logging
from pathlib import Path
from typing import Optional, Generator

from ollama import Client, Options

from tezzchain.llm.base import BaseLLM


logger = logging.getLogger("tezzchain")


__tezzchain_ollama_llm_version__ = "0.0.1"
__ollama_server_supported_version__ = "0.5.4" 


class OllamaLLM(BaseLLM):
    """
    Singleton class that wraps the Ollama API to provide language model functionalities.
    
    Inherits from BaseLLM, the base class for all LLMs in TezzChain.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of OllamaLLM is created (Singleton pattern).
        
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: The singleton instance of OllamaLLM.
        """
        if cls._instance is None:
            cls._instance = super(OllamaLLM, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model: Optional[str] = "qwen2:0.5b-instruct",
        host: Optional[str] = "http://localhost:11434",
        streaming: Optional[bool] = True,
        **kwargs
    ):
        """
        Initializes the OllamaLLM with the specified parameters.
        
        :param model: The name of the model to use.
        :param host: The host URL of the Ollama API.
        :param streaming: Whether to enable streaming of responses.
        :param kwargs: Additional keyword arguments.
        """
        if not hasattr(self, "initialized"):
            self.model = model
            self.host = host
            self.response_streaming = streaming
            self.client = Client(host=self.host)
            self.kwargs = kwargs
            self.initialized = True

    @classmethod
    def create_instance(
        cls,
        model: Optional[str] = "qwen2:0.5b-instruct",
        host: Optional[str] = None,
        streaming: Optional[bool] = True,
        **kwargs
    ) -> 'OllamaLLM':
        """
        Creates and returns an instance of OllamaLLM.
        
        :param model: The name of the model to use.
        :param host: The host URL of the Ollama API.
        :param streaming: Whether to enable streaming of responses.
        :param kwargs: Additional keyword arguments.
        :return: An instance of OllamaLLM.
        """
        instance = cls(model=model, host=host, streaming=streaming, **kwargs)
        if "modelfile" in kwargs:
            instance.model = instance.__create_custom_model(
                model, kwargs["modelfile"], stream=streaming
            )
        return instance

    def __create_custom_model(
        self, model: str, modelfile: Path | str, stream: bool
    ) -> str:
        """
        Creates a custom model using the provided modelfile content.
        
        :param model: The name of the model to use.
        :param modelfile: The path to the modelfile or the content of the modelfile.
        :param stream: Whether to enable streaming of responses.
        :return: The name of the model.
        """
        modelfile_content = modelfile.read_text()
        chunks = self.client.create(
            model=model, modelfile=modelfile_content, stream=stream
        )
        for chunk in chunks:
            logger.info(chunk)
        return model

    def __get_options(self, num_predict: int) -> Options:
        """
        Constructs and returns the Options object for the API request.
        
        :param num_predict: The number of predictions to generate.
        :return: An Options object.
        """
        options = Options(
            num_ctx=self.kwargs.get("num_ctx", 2048),
            low_vram=self.kwargs.get("low_vram", False),
            num_predict=(
                self.kwargs.get("num_predict", -1)
                if num_predict == -10
                else num_predict
            ),
            **self.kwargs["hyperparameters"]
        )
        return options

    def generate(self, query: str, num_predict: Optional[int] = -10) -> Generator:
        """
        Generates a response to the given query.
        
        :param query: The input query string.
        :param num_predict: The number of predictions to generate.
        :return: A generator that yields response chunks.
        """
        options = self.__get_options(num_predict)
        for chunk in self.client.generate(
            model=self.model,
            prompt=query,
            stream=self.response_streaming,
            options=options,
        ):
            yield chunk

    def chat(
        self, messages: list[dict[str, str]], num_predict: Optional[int] = -10
    ) -> Generator:
        """
        Handles a chat interaction with the model.
        
        :param messages: A list of messages in the format [{"role": "user", "content": "message"}, ...].
        :param num_predict: The number of predictions to generate.
        :return: A generator that yields response chunks.
        """
        options = self.__get_options(num_predict)
        for chunk in self.client.chat(
            model=self.model,
            stream=self.response_streaming,
            options=options,
            messages=messages,
        ):
            response = {
                "response": chunk["message"]["content"],
                "response_completed": chunk["done"],
            }
            if response["response_completed"]:
                response["response_completion_reason"] = chunk["done_reason"]
            yield response

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