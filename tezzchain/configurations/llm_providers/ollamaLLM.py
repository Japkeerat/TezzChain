"""
Configurations for running the LLM through Ollama
"""

from typing import Sequence, Optional
from dataclasses import dataclass, asdict


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434/"
    model: str = "qwen2:0.5b-instruct"
    streaming: bool = True
    low_vram: bool = False
    num_ctx: int = 2048
    num_predict: int = -1
    seed: int = 42
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[Sequence[str]] = None

    def to_structured_dict(self):
        full_dict = asdict(self)
        hyperparameters = {
            k: v
            for k, v in full_dict.items()
            if v is not None
            and k
            not in {
                "host",
                "model",
                "streaming",
                "low_vram",
                "num_ctx",
                "num_predict",
                "seed",
            }
        }
        return {
            "host": self.host,
            "model": self.model,
            "streaming": self.streaming,
            "low_vram": self.low_vram,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "seed": self.seed,
            "hyperparameters": hyperparameters,
        }