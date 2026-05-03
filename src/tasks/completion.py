from __future__ import annotations

from typing import Any

import numpy as np

from src.data import BaseData
from src.metrics import val_ERGAS, val_NMSE_dB, val_PSNR, val_RMSE, val_SAM, val_SSIM
from src.methods.base import BaseDecomposeMethod
from src.tasks.base import BaseTask
from src.types import LogCallback, Result, Tensor
from src.utils.completion_ops import random_observed_mask, structured_observed_mask


def _metric_bundle(
    original: Tensor,
    reconstructed: Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    return {
        f"RMSE_{prefix}": val_RMSE(original, reconstructed),
        f"SAM_{prefix}": val_SAM(original, reconstructed),
        f"NMSE_dB_{prefix}": val_NMSE_dB(original, reconstructed),
        f"PSNR_{prefix}": val_PSNR(original, reconstructed),
        f"SSIM_{prefix}": val_SSIM(original, reconstructed),
        f"ERGAS_{prefix}": val_ERGAS(original, reconstructed),
    }


class RandomMissingCompletionTask(BaseTask):
    def __init__(self, **task_cfg: Any) -> None:
        super().__init__()
        self.cfg = task_cfg
        self.logcallback = LogCallback(log_level=int(self.cfg["log_level"]))
        self.observed_mask: np.ndarray | None = None

    def setup(self, data: BaseData, method: BaseDecomposeMethod) -> None:
        self.data = data
        self.method = method
        dense = np.asarray(self.data.get(split="eval").dense)
        self.observed_mask = random_observed_mask(
            dense.shape,
            missing_rate=float(self.cfg["missing_rate"]),
            seed=int(self.cfg["seed"]),
        )
        self.data._mask = np.array(self.observed_mask, dtype=bool, copy=True)

    def run(self) -> Result:
        if not isinstance(self.method, BaseDecomposeMethod) or not isinstance(self.data, BaseData):
            raise ValueError("Method and data must be set before running the task.")
        if self.observed_mask is None:
            raise ValueError("Task mask has not been initialized. Call setup() first.")

        tensor = self.data.get(split="fit")
        self._measure_fit(self.method.fit, tensor, tensor.mask, self.logcallback)
        reconstruction = self.method.reconstruct()
        state_dict = self.method.get_state_dict()
        state_dict["reconstruction"] = np.array(reconstruction.dense, copy=True)
        state_dict["observed_mask"] = np.array(self.observed_mask, copy=True)
        state_dict["missing_mask"] = np.array(~self.observed_mask, copy=True)
        if hasattr(self.data, "scene_name"):
            state_dict["scene_name"] = getattr(self.data, "scene_name")
        eval_dict = self.evaluate(reconstruction)
        logs = self.logcallback.logs
        return Result(state_dict, eval_dict, logs)

    def evaluate(self, reconstruction: Tensor | None = None) -> dict[str, float]:
        if self.method is None or self.data is None:
            raise ValueError("Method and data must be set before evaluation.")
        if self.observed_mask is None:
            raise ValueError("Task mask has not been initialized. Call setup() first.")

        rec = reconstruction if reconstruction is not None else self.method.reconstruct()
        original_dense = np.asarray(self.data.get(split="eval").dense, dtype=np.float32)
        observed_mask = np.asarray(self.observed_mask, dtype=bool)
        missing_mask = ~observed_mask

        original_all = Tensor(shape=original_dense.shape, dense=original_dense)
        eval_dict = {
            "missing_rate": float(self.cfg["missing_rate"]),
            "observed_rate": float(np.mean(observed_mask)),
            "mask_seed": int(self.cfg["seed"]),
        }
        eval_dict.update(_metric_bundle(original_all, rec, prefix="all"))

        if np.any(missing_mask):
            original_missing = Tensor(shape=original_dense.shape, dense=original_dense, mask=missing_mask)
            eval_dict.update(_metric_bundle(original_missing, rec, prefix="missing"))
        else:
            eval_dict["RMSE_missing"] = float("nan")
            eval_dict["SAM_missing"] = float("nan")
            eval_dict["NMSE_dB_missing"] = float("nan")

        if self.fit_time_sec is not None:
            eval_dict["fit_time_sec"] = self.fit_time_sec
        return eval_dict


class StructuredMissingCompletionTask(RandomMissingCompletionTask):
    def setup(self, data: BaseData, method: BaseDecomposeMethod) -> None:
        self.data = data
        self.method = method
        dense = np.asarray(self.data.get(split="eval").dense)
        block_shape = self.cfg.get("block_shape", None)
        if block_shape is not None:
            block_shape = tuple(int(v) for v in block_shape)
        self.observed_mask = structured_observed_mask(
            dense.shape,
            pattern=str(self.cfg["pattern"]),
            missing_rate=float(self.cfg["missing_rate"]),
            seed=int(self.cfg["seed"]),
            block_shape=block_shape,
            stripe_axis=int(self.cfg.get("stripe_axis", 1)),
            stripe_width=self.cfg.get("stripe_width", None),
            band_axis=int(self.cfg.get("band_axis", -1)),
        )
        self.data._mask = np.array(self.observed_mask, dtype=bool, copy=True)

    def evaluate(self, reconstruction: Tensor | None = None) -> dict[str, float]:
        eval_dict = super().evaluate(reconstruction)
        eval_dict["mask_pattern"] = str(self.cfg["pattern"])
        return eval_dict
