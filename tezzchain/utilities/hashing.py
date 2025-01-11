from hashlib import md5
from pathlib import Path

from tezzchain.utilities.read_file import read_file_as_text


def get_file_hash(path: Path) -> str:
    content = read_file_as_text(path).encode("utf-8")
    hash = md5(content)
    return str(hash)
