from typing import Optional
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs):
        pass

    @abstractmethod
    def generate(self, query: str, **kwargs):
        pass

    @abstractmethod
    def get_model(self) -> str:
        pass
