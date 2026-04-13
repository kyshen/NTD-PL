from abc import ABC, abstractmethod
from src.types import Tensor, Array
from typing import Optional, Any


class BaseData(ABC):

    def __init__(self, **data_cfg: Any):
        self.cfg = data_cfg
        self._mask: Optional[Array] = None
        self._dense = self._make_dense()
        self._dense_eval = self._dense.copy()
        pass
    
    @abstractmethod
    def _make_dense(self) -> Array:
        raise NotImplementedError
    
    def get_size(self) -> int:
        return self._dense.size

    def get(self, split: str) -> Tensor:
        if split == 'fit':
            return Tensor(
                shape=self._dense.shape,
                dense=self._dense,
                mask=self._mask,
            )
        elif split == 'eval':
            return Tensor(
                shape=self._dense_eval.shape,
                dense=self._dense_eval,
                mask=self._mask,
            )
        else:
            raise ValueError(f"Unsupported split: {split}")
        
    def get_mask(self) -> Array:
        if self._mask is None:
            raise ValueError("Mask has not been set for this data.")
        return self._mask
