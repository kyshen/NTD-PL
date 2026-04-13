from __future__ import annotations

from pathlib import Path

from .io import FIGURE_ROOT
from .specs import BaseSpec


def output_path(spec: BaseSpec) -> Path:
    directory = FIGURE_ROOT / ("main" if spec.export.section == "main" else "appendix")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / spec.export.stem


def save_figure(fig, spec: BaseSpec) -> list[Path]:
    target = output_path(spec)
    written: list[Path] = []
    for fmt in spec.export.formats:
        path = target.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if fmt.lower() != "pdf":
            save_kwargs["dpi"] = 300
        fig.savefig(path, **save_kwargs)
        written.append(path)
    return written


def render_and_save(fig, spec: BaseSpec, *, sync_legacy: bool = True) -> list[Path]:
    del sync_legacy
    return save_figure(fig, spec)
