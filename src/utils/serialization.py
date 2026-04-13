import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable_default(obj: Any):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _dump_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent, default=_jsonable_default),
        encoding="utf-8",
    )
