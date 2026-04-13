from typing import Tuple

import numpy as np


def validate_target_shape(target_shape: Tuple[int, int]) -> tuple[int, int]:
    target_shape = tuple(int(v) for v in target_shape)
    if len(target_shape) != 2:
        raise ValueError(f"target_shape must have length 2, got {target_shape}")

    target_h, target_w = target_shape
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_shape must be positive, got {target_shape}")
    return target_h, target_w


def downsample_image_to_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = validate_target_shape(target_shape)

    height, width = image.shape[:2]
    if target_h > height or target_w > width:
        raise ValueError(f"target_shape {(target_h, target_w)} exceeds image size {(height, width)}")

    row_idx = np.linspace(0, height - 1, num=target_h).round().astype(int)
    col_idx = np.linspace(0, width - 1, num=target_w).round().astype(int)
    sampled = image[row_idx][:, col_idx]
    return sampled
