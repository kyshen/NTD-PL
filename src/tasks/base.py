from abc import ABC, abstractmethod
from src.types import Result
from typing import Any, Union, Optional
from time import perf_counter
from src.data import BaseData
from src.methods.base import BaseDecomposeMethod


BaseMethod = BaseDecomposeMethod

class BaseTask(ABC):

    def __init__(self) -> None:
        self.method: Optional[BaseMethod] = None
        self.data: Optional[BaseData] = None
        self.fit_time_sec: Optional[float] = None

    def _measure_fit(self, fit_fn, *args, **kwargs) -> None:
        start = perf_counter()
        fit_fn(*args, **kwargs)
        self.fit_time_sec = perf_counter() - start

    @abstractmethod
    def run(self) -> Result:
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self) -> dict:
        raise NotImplementedError
