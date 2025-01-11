"""
Abstract Base Class for Language Models defining the functions that each LLM must implement
"""

from typing import Optional
from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs):
        pass

    @abstractmethod
    def get_model(self) -> str:
        pass
