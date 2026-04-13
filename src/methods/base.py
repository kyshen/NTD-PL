from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional
from src.types import LogCallback
from src.types import Tensor, Array

class BaseMethod(ABC):
    def __init__(self) -> None:
        self.task_type: str

    @abstractmethod
    def reconstruct(self) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def get_num_params(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError


class BaseDecomposeMethod(BaseMethod):

    @abstractmethod
    def fit(self, data: Tensor, mask: Optional[Array], logcallback: LogCallback) -> None:
        raise NotImplementedError

