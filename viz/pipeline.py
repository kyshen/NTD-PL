from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .aggregate import build
from .catalog import FIGURE_REGISTRY, get_figure_job
from .export import render_and_save
from .renderers import render_figure


def select_jobs(*, scope: str = "all", figure_ids: Iterable[str] | None = None):
    selected = FIGURE_REGISTRY
    if scope != "all":
        selected = [job for job in selected if job.spec.export.section == scope]
    if figure_ids is not None:
        wanted = set(figure_ids)
        selected = [job for job in selected if job.figure_id in wanted]
    return selected


def build_figure(figure_id: str, *, sync_legacy: bool = True) -> list[Path]:
    job = get_figure_job(figure_id)
    data = build(job.aggregate_key)
    fig = render_figure(data, job.spec)
    try:
        return render_and_save(fig, job.spec, sync_legacy=sync_legacy)
    finally:
        fig.clf()


def build_figures(
    *,
    scope: str = "all",
    figure_ids: Iterable[str] | None = None,
    sync_legacy: bool = True,
) -> list[tuple[str, list[Path]]]:
    outputs: list[tuple[str, list[Path]]] = []
    for job in select_jobs(scope=scope, figure_ids=figure_ids):
        data = build(job.aggregate_key)
        fig = render_figure(data, job.spec)
        try:
            paths = render_and_save(fig, job.spec, sync_legacy=sync_legacy)
        finally:
            fig.clf()
        outputs.append((job.figure_id, paths))
    return outputs
