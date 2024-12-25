import json
import shutil
from pathlib import Path

import yaml

from tezzchain.constants import TEZZCHAIN_TEMP_DIR


def cleanup_temp_dir():
    for item in TEZZCHAIN_TEMP_DIR.glob("*"):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def __check_file_existence(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path.__str__} doesn't exist")


def read_yaml_file(file_path: Path) -> dict:
    __check_file_existence(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    return content


def read_json_file(file_path: Path) -> dict:
    __check_file_existence(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content


def read_file_intelligently(file_path: Path) -> dict:
    __check_file_existence(file_path)
    extension = file_path.suffix.lower()
    if extension == ".json":
        return read_json_file(file_path)
    elif extension in [".yaml", ".yml"]:
        return read_yaml_file(file_path)
    else:
        raise ValueError(
            f"""Extension {extension} for the configuration file is not supported. 
            Only supported extensions are '.json', '.yaml', '.yml'."""
        )
