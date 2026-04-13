from src.tasks.base import BaseTask
from src.types import Result, LogCallback
from src.methods.base import BaseDecomposeMethod
from typing import Any, Dict
import numpy as np
from src.data import BaseData
from src.metrics import CR, NMSE, NMSE_dB, RMSE, SAM


class DecomposeTask(BaseTask):
    def __init__(self, **task_cfg: Any) -> None:
        super().__init__()
        self.cfg = task_cfg
        self.logcallback = LogCallback(log_level=self.cfg["log_level"])

    def setup(self, data: BaseData, method: BaseDecomposeMethod) -> None:
        self.data = data
        self.method = method

    def run(self) -> Result:
        if not isinstance(self.method, BaseDecomposeMethod) or not isinstance(self.data, BaseData):
            raise ValueError("Method and data must be set before running the task.")
        tensor = self.data.get(split="fit")
        self._measure_fit(self.method.fit, tensor, tensor.mask, self.logcallback)
        state_dict = self.method.get_state_dict()
        eval = self.evaluate()
        logs = self.logcallback.logs
        return Result(state_dict, eval, logs)
    
    def evaluate(self) -> dict:
        if self.method is None or self.data is None:
            raise ValueError("Method and data must be set before evaluation.")
        rec = self.method.reconstruct()
        eval = {}
        tensor_eval = self.data.get(split='eval')
        eval.update(CR(self.data.get_size(), self.method.get_num_params()))
        eval.update(RMSE(tensor_eval, rec))
        eval.update(NMSE(tensor_eval, rec))
        eval.update(NMSE_dB(tensor_eval, rec))
        eval.update(SAM(tensor_eval, rec))
        if self.fit_time_sec is not None:
            eval["fit_time_sec"] = self.fit_time_sec
        return eval
