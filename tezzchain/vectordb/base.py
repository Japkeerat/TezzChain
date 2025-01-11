"""
Abstract Base Class for Vector Databases defining the functions that each Vector DB must implement
"""

from typing import Optional
from abc import ABC, abstractmethod


class BaseVectorDB(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_content(self, content: str, embedding: list, metadata: dict, id: str):
        pass
