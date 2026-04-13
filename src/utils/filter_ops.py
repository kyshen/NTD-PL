from typing import Any

import numpy as np


_TEXT_BITMAPS = {
    "A": np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
    ], dtype=bool),
    "K": np.array([
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ], dtype=bool),
    "M": np.array([
        [1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
    ], dtype=bool),
    "S": np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
    ], dtype=bool),
    "?": np.array([
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=bool),
    " ": np.zeros((7, 3), dtype=bool),
}


def random_mask(shape: tuple[int, ...], rho: float) -> np.ndarray:
    return np.random.rand(*shape) < rho


def structured_mask(shape: tuple[int, ...], rho: float, cfg: dict[str, Any]) -> np.ndarray:
    if len(shape) < 2:
        return random_mask(shape, rho)

    height, width = shape[0], shape[1]
    pattern = cfg.get("structured_pattern", "block")
    if pattern == "block":
        missing2d = _block_mask2d(height, width, rho, cfg)
    elif pattern == "brush":
        missing2d = _brush_mask2d(height, width, rho, cfg)
    elif pattern == "text":
        missing2d = _text_mask2d(height, width, rho, cfg)
    else:
        raise ValueError(f"Unsupported structured_pattern: {pattern}")
    return _broadcast_mask(~missing2d, shape)


def orthogonal_nonlinear_part(x: np.ndarray, gx: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    coef = np.sum(gx * x) / (np.sum(x * x) + eps)
    return gx - coef * x


def mix_with_exact_energy_ratio(x: np.ndarray, r: np.ndarray, alpha: float, eps: float = 1e-8) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    target_norm = np.sqrt(np.prod(x.shape))
    r_norm = np.linalg.norm(r)
    if r_norm < eps:
        return x.copy()

    r = r / r_norm * target_norm
    return np.sqrt(1 - alpha) * x + np.sqrt(alpha) * r


def typical_scale(x: np.ndarray, q: float = 0.9, eps: float = 1e-8) -> np.ndarray:
    return np.quantile(np.abs(x), q) + eps


def _broadcast_mask(mask2d: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    mask = mask2d.astype(bool, copy=False)
    while mask.ndim < len(shape):
        mask = mask[..., None]
    return np.broadcast_to(mask, shape).copy()


def _paint_disk(mask: np.ndarray, cy: int, cx: int, radius: int):
    y0 = max(0, cy - radius)
    y1 = min(mask.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(mask.shape[1], cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    mask[y0:y1, x0:x1] |= disk


def _complete_missing_target(missing: np.ndarray, target_missing: int):
    remaining = target_missing - int(missing.sum())
    if remaining <= 0:
        return
    flat = np.flatnonzero(~missing)
    if flat.size == 0:
        return
    extra = np.random.choice(flat, size=min(remaining, flat.size), replace=False)
    missing.reshape(-1)[extra] = True


def _block_mask2d(height: int, width: int, rho: float, cfg: dict[str, Any]) -> np.ndarray:
    target_missing = int(round((1.0 - rho) * height * width))
    missing = np.zeros((height, width), dtype=bool)
    if target_missing == 0:
        return missing

    min_block_ratio = float(cfg.get("block_min_ratio", 0.15))
    max_block_ratio = float(cfg.get("block_max_ratio", 0.45))
    max_attempts = int(cfg.get("block_max_attempts", 128))

    for _ in range(max_attempts):
        block_h = np.random.randint(
            max(1, int(np.ceil(height * min_block_ratio))),
            max(2, int(np.ceil(height * max_block_ratio))) + 1,
        )
        block_w = np.random.randint(
            max(1, int(np.ceil(width * min_block_ratio))),
            max(2, int(np.ceil(width * max_block_ratio))) + 1,
        )
        block_h = min(block_h, height)
        block_w = min(block_w, width)
        top = np.random.randint(0, height - block_h + 1)
        left = np.random.randint(0, width - block_w + 1)
        missing[top:top + block_h, left:left + block_w] = True
        if missing.sum() >= target_missing:
            break

    _complete_missing_target(missing, target_missing)
    return missing


def _brush_mask2d(height: int, width: int, rho: float, cfg: dict[str, Any]) -> np.ndarray:
    target_missing = int(round((1.0 - rho) * height * width))
    missing = np.zeros((height, width), dtype=bool)
    if target_missing == 0:
        return missing

    min_width = int(cfg.get("brush_min_width", max(1, round(min(height, width) * 0.03))))
    max_width = int(cfg.get("brush_max_width", max(min_width, round(min(height, width) * 0.12))))
    min_length = float(cfg.get("brush_min_length", 0.12))
    max_length = float(cfg.get("brush_max_length", 0.4))
    max_strokes = int(cfg.get("brush_max_strokes", 64))

    stroke_count = 0
    while missing.sum() < target_missing and stroke_count < max_strokes:
        stroke_count += 1
        width_px = np.random.randint(min_width, max_width + 1)
        segments = np.random.randint(1, 4)
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        for _ in range(segments):
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            length = np.random.uniform(min_length, max_length) * min(height, width)
            y2 = int(np.clip(np.round(y + length * np.sin(angle)), 0, height - 1))
            x2 = int(np.clip(np.round(x + length * np.cos(angle)), 0, width - 1))
            steps = max(abs(y2 - y), abs(x2 - x), 1)
            ys = np.linspace(y, y2, steps + 1).round().astype(int)
            xs = np.linspace(x, x2, steps + 1).round().astype(int)
            for py, px in zip(ys, xs):
                _paint_disk(missing, py, px, max(1, width_px // 2))
            y, x = y2, x2

    _complete_missing_target(missing, target_missing)
    return missing


def _text_bitmap(text: str) -> np.ndarray:
    glyphs = [_TEXT_BITMAPS.get(char, _TEXT_BITMAPS["?"]) for char in text.upper()]
    rows = glyphs[0].shape[0]
    spacer = np.zeros((rows, 1), dtype=bool)
    bitmap = glyphs[0]
    for glyph in glyphs[1:]:
        bitmap = np.hstack((bitmap, spacer, glyph))
    return bitmap


def _text_mask2d(height: int, width: int, rho: float, cfg: dict[str, Any]) -> np.ndarray:
    target_missing = int(round((1.0 - rho) * height * width))
    missing = np.zeros((height, width), dtype=bool)
    if target_missing == 0:
        return missing

    text = str(cfg.get("text_mask_string", "MASK"))
    bitmap = _text_bitmap(text)
    natural_max_scale = max(1, min(height // bitmap.shape[0], width // bitmap.shape[1]))
    scale_lo = max(1, min(natural_max_scale, int(cfg.get("text_min_scale", 1))))
    scale_hi = max(scale_lo, min(natural_max_scale, int(cfg.get("text_max_scale", natural_max_scale))))
    placements = int(cfg.get("text_max_placements", 8))

    for _ in range(placements):
        scale = np.random.randint(scale_lo, scale_hi + 1)
        stamp = np.kron(bitmap, np.ones((scale, scale), dtype=bool))
        if stamp.shape[0] > height or stamp.shape[1] > width:
            continue
        top = np.random.randint(0, height - stamp.shape[0] + 1)
        left = np.random.randint(0, width - stamp.shape[1] + 1)
        missing[top:top + stamp.shape[0], left:left + stamp.shape[1]] |= stamp
        if missing.sum() >= target_missing:
            break

    _complete_missing_target(missing, target_missing)
    return missing
