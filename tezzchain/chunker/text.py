from pathlib import Path
from typing import Optional

from unstructured.partition.text import partition_text
from unstructured.chunking.basic import chunk_elements


class TextChunker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def chunk(self, content):
        elements = partition_text(text=content)
        chunks = chunk_elements(
            elements=elements,
            max_characters=self.kwargs.get("max_characters", None),
            new_after_n_chars=self.kwargs.get("new_after_n_chars", None),
            overlap=self.kwargs.get("overlap", None),
            overlap_all=self.kwargs.get("overlap_all", None),
            include_orig_elements=self.kwargs.get("include_orig_elements", None),
        )
        return chunks
