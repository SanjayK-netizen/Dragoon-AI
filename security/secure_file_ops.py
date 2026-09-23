import json, os
from pathlib import Path
from typing import Any, Iterable, Mapping
from dotenv import dotenv_values

def mask_secret(value: object, keep_last: int=4)->str:
   ...
def redact_values(value: Any, sensitive_keys=("password","secret","token","api_key","jwt")) -> Any:
   ...
def get_secret(name: str, default: str|None=None, *, required: bool = False, env_file: str|Path = ".env") -> str|None:
   # get os.environ, dotenv_values
   ...
def read_text_file(path, *, encoding="utf-8", max_size=1048576) -> str: check exists and size
def write_text_file(path,...)
def write_json_file(path,payload,...)
def log_safe(message,payload,...)