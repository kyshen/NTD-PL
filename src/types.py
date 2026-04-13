from typing import Tuple, Dict, Any, Union, Optional, List
from dataclasses import dataclass
from pathlib import Path
from src.utils.serialization import _dump_json
import json

import numpy as np

Float = np.float32
Array = np.ndarray

@dataclass
class Tensor:
    shape: Tuple[int, ...]
    dense: Array
    mask: Optional[Array] = None
    

@dataclass
class Configs:
    data_cfg: Dict[str, Any]
    method_cfg: Dict[str, Any]


class LogCallback:
    def __init__(self, log_level: int) -> None:
        self.log_level = log_level
        self.logs = None

    def addlog(self, log: Dict) -> None:
        if self.logs is None:
            self.logs = []
        self.logs.append(log)


state_name: str = "state.mat"
eval_name: str = "eval.json"
logs_name: str = "logs.json"


def _require_scipy_io():
    try:
        from scipy.io import savemat, loadmat
    except ImportError as exc:
        raise ImportError("Saving/loading .mat requires scipy. Please install scipy first.") from exc
    return savemat, loadmat


def _to_mat_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return np.array(value)
    if isinstance(value, (int, float, bool)):
        return np.array(value)
    if isinstance(value, str):
        return np.array(value, dtype=object)
    if isinstance(value, list):
        if all(isinstance(item, np.ndarray) for item in value):
            arr = np.empty((len(value),), dtype=object)
            arr[:] = value
            return arr
        return np.array(value, dtype=object)
    if isinstance(value, tuple):
        return np.array(list(value), dtype=object)
    return np.array(value, dtype=object)


def _from_mat_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object:
        flat = value.reshape(-1)
        out = []
        for item in flat:
            if isinstance(item, np.ndarray) and item.size == 1:
                out.append(item.item())
            else:
                out.append(item)
        return out
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value

@dataclass
class Result:
    state_dict: Dict[str, Any]
    eval: Dict[str, Any]
    logs: Optional[List[Dict]]

    def save(self, outdir: Union[str, Path], exist_ok: bool = True) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=exist_ok)

        state_path = outdir / state_name
        savemat, _ = _require_scipy_io()
        state_dict = {k: _to_mat_compatible(v) for k, v in self.state_dict.items()}
        savemat(state_path, state_dict, do_compression=True)
        _dump_json(outdir / eval_name, self.eval)
        _dump_json(outdir / logs_name, self.logs)
        
        return outdir

    @staticmethod
    def load(outdir: Union[str, Path]) -> "Result":
        outdir = Path(outdir)

        state_path = outdir / state_name
        _, loadmat = _require_scipy_io()
        mat_dict = loadmat(state_path, squeeze_me=True)
        state_dict = {
            k: _from_mat_value(v)
            for k, v in mat_dict.items()
            if not k.startswith("__")
        }

        eval_dict = json.loads((outdir / eval_name).read_text(encoding="utf-8"))
        logs_dict = json.loads((outdir / logs_name).read_text(encoding="utf-8"))

        return Result(state_dict=state_dict, eval=eval_dict, logs=logs_dict)

    
