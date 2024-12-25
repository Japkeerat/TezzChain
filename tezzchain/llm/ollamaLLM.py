import logging
from pathlib import Path
from typing import Optional, Generator

from ollama import Client, Options

from tezzchain.llm.base import BaseLLM


logger = logging.getLogger("tezzchain")


class OllamaLLM(BaseLLM):
    _instance = None

    def __new__(cls, *args, **kwargs):
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
    ):
        instance = cls(model=model, host=host, streaming=streaming, **kwargs)
        if "modelfile" in kwargs:
            instance.model = instance.__create_custom_model(
                model, kwargs["modelfile"], stream=streaming
            )
        return instance

    def __create_custom_model(
        self, model: str, modelfile: Path | str, stream: bool
    ) -> str:
        modelfile_content = modelfile.read_text()
        chunks = self.client.create(
            model=model, modelfile=modelfile_content, stream=stream
        )
        for chunk in chunks:
            logger.info(chunk)
        return model

    def __get_options(self, num_predict: int) -> Options:
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
        return self.model

    def get_client(self) -> Client:
        return self.client
