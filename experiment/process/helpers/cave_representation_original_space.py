from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from tensorly.tucker_tensor import tucker_to_tensor

from ...config import get_env
from ...hsi_defaults import CAVE_RECON_MAIN_RANK, CAVE_RECON_RANKS
from ...utils.io import load_run_parquets, load_state_mat, maybe_numeric
from ...utils.paper import sync_artifact_to_latex, write_csv_artifact, write_text_artifact
from ...utils.plotting import PALETTE, apply_theme, legend_style, method_style, save_figure, style_axes
from .cave_random_completion import _cave_dataset_kwargs_from_row
from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter


RANK_ORDER = CAVE_RECON_RANKS
MAIN_RANK = CAVE_RECON_MAIN_RANK
MAIN_SCENES = (14, 8, 2)
FOCUS_SCENES = (2, 3, 8)  # beads, cd, feathers
NTDPL_PMAX = 6


@dataclass(frozen=True)
class SceneRunPayload:
    scene_id: int
    scene_name: str
    original: np.ndarray
    recon_tucker: np.ndarray
    recon_ntdpl: np.ndarray


def _jsonish(value: Any) -> Any:
    out = value
    while isinstance(out, str):
        text = out.strip()
        if not text:
            return text
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return out
        if loaded == out:
            return loaded
        out = loaded
    return out


def _parse_rank(value: Any) -> tuple[int, int, int]:
    parsed = _jsonish(value)
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            raise ValueError("Empty rank value.")
        items = [part.strip() for part in text.strip("[]()").split(",") if part.strip()]
        return tuple(int(item) for item in items)  # type: ignore[return-value]
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)  # type: ignore[return-value]
    raise ValueError(f"Cannot parse rank from value: {value}")


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]}, {rank[1]}, {rank[2]})"


def _pm_text(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def _pm_latex(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _latex_pvalue(value: float) -> str:
    if value < 1e-3:
        base, exp = f"{value:.2e}".split("e")
        exponent = int(exp)
        return f"{float(base):.2f}$\\times 10^{{{exponent}}}$"
    return f"{value:.3f}"


def _rank_biserial(diffs: np.ndarray) -> float:
    nonzero = diffs[~np.isclose(diffs, 0.0)]
    if nonzero.size == 0:
        return 0.0
    order = np.argsort(np.abs(nonzero))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(nonzero) + 1, dtype=np.float64)
    positive_rank_sum = float(ranks[nonzero > 0].sum())
    negative_rank_sum = float(ranks[nonzero < 0].sum())
    denom = len(nonzero) * (len(nonzero) + 1) / 2.0
    if denom <= 0:
        return 0.0
    return (positive_rank_sum - negative_rank_sum) / denom


def _save_main_figure(env: object, fig: plt.Figure, stem: str) -> None:
    output_base = env.project_root / "experiment" / "outputs" / "figures" / "main" / stem
    save_figure(fig, output_base, formats=("pdf", "png"), dpi=600)
    for fmt in ("pdf", "png"):
        sync_artifact_to_latex(env, output_base.with_suffix(f".{fmt}"), target_name=f"figures/main/{stem}.{fmt}")


def _state_path_from_row(row: pd.Series) -> str:
    candidate = row.get("state_path", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    raise RuntimeError(
        "Missing `state_path` for cave-representation visual payloads. "
        "Re-run `python -m experiment cave-representation run` to regenerate original-space raw states."
    )


def _state_reconstruction(state: dict[str, Any]) -> np.ndarray:
    for key in ("reconstruction", "fitted"):
        if key in state:
            value = np.asarray(state[key])
            if value.ndim == 3:
                return np.asarray(value, dtype=np.float32)

    core = np.asarray(state["core"], dtype=np.float32)
    factors_obj = state["factors"]
    if isinstance(factors_obj, np.ndarray) and factors_obj.dtype == object:
        factors = [np.asarray(item, dtype=np.float32) for item in factors_obj.reshape(-1)]
    elif isinstance(factors_obj, list):
        factors = [np.asarray(item, dtype=np.float32) for item in factors_obj]
    else:
        raise TypeError(f"Unsupported factors payload: {type(factors_obj)}")

    latent = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
    beta_obj = state.get("beta", None)
    if beta_obj is None:
        return latent

    beta = np.asarray(beta_obj, dtype=np.float32).reshape(-1)
    if beta.size <= 1:
        return latent

    terms = [float(coeff) * (latent ** degree) for degree, coeff in enumerate(beta)]
    return np.sum(np.stack(terms, axis=0), axis=0, dtype=np.float32)


def _load_runs() -> tuple[pd.DataFrame, object]:
    env = get_env("cave-representation")
    runs = load_run_parquets(env.results_dir)["runs"].copy()
    if runs.empty:
        raise RuntimeError(
            "No runs found for cave-representation. Expected migrated original-space results under "
            "`multirun/cave-representation/runs.parquet`."
        )

    frame = runs.copy()
    if "scene_id" not in frame.columns:
        frame["scene_id"] = maybe_numeric(frame.get("data.id", frame.get("ovr.data.id"))).astype(int)
    if "method_name" not in frame.columns:
        frame["method_name"] = frame.get("method._name", frame.get("ovr.method")).astype(str)
    if "rank" not in frame.columns:
        frame["rank"] = frame.get("method.rank", frame.get("ovr.method.rank")).map(_parse_rank)
    else:
        frame["rank"] = frame["rank"].map(_parse_rank)
    if "rank_text" not in frame.columns:
        frame["rank_text"] = frame["rank"].map(_rank_text)
    if "p_max" not in frame.columns:
        raw_pmax = frame.get("ovr.method.p_max", frame.get("method.p_max"))
        frame["p_max"] = maybe_numeric(raw_pmax) if raw_pmax is not None else np.nan
    for metric in ("CR", "RMSE", "NMSE_dB", "SAM", "fit_time_sec"):
        if metric in frame.columns:
            frame[metric] = maybe_numeric(frame[metric]).astype(float)

    frame = frame.loc[frame["method_name"].isin(["tucker", "ntdpl"])].copy()
    frame = frame.loc[frame["rank"].isin(RANK_ORDER)].copy()
    frame = frame.sort_values(["rank_text", "scene_id", "method_name"]).drop_duplicates(
        subset=["rank_text", "scene_id", "method_name"],
        keep="last",
    )
    return frame.reset_index(drop=True), env


def _summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank in RANK_ORDER:
        rank_text = _rank_text(rank)
        panel = frame.loc[frame["rank"] == rank].copy()
        for method_name, label in (("tucker", "Tucker"), ("ntdpl", "NTD-PL")):
            subset = panel.loc[panel["method_name"] == method_name].copy()
            if subset.empty:
                continue
            rows.append(
                {
                    "Rank": rank_text,
                    "Method": label,
                    "Params": int(round(float(np.prod(rank) + 512 * rank[0] + 512 * rank[1] + 31 * rank[2]) + (NTDPL_PMAX + 1 if method_name == "ntdpl" else 0))),
                    "Scenes": int(subset["scene_id"].nunique()),
                    "CR": _pm_text(float(subset["CR"].mean()), float(subset["CR"].std(ddof=0)), 2),
                    "RMSE": _pm_text(float(subset["RMSE"].mean()), float(subset["RMSE"].std(ddof=0)), 5),
                    "NMSE(dB)": _pm_text(float(subset["NMSE_dB"].mean()), float(subset["NMSE_dB"].std(ddof=0)), 3),
                    "SAM(deg)": _pm_text(float(subset["SAM"].mean()), float(subset["SAM"].std(ddof=0)), 3),
                    "RMSE_mean": float(subset["RMSE"].mean()),
                    "RMSE_std": float(subset["RMSE"].std(ddof=0)),
                    "NMSE_mean": float(subset["NMSE_dB"].mean()),
                    "NMSE_std": float(subset["NMSE_dB"].std(ddof=0)),
                    "SAM_mean": float(subset["SAM"].mean()),
                    "SAM_std": float(subset["SAM"].std(ddof=0)),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No rows available for original-space CAVE reconstruction summary.")
    return summary


def reconstruction_table() -> None:
    frame, env = _load_runs()
    summary = _summary_table(frame)
    csv_summary = summary.loc[:, ["Rank", "Method", "Params", "Scenes", "CR", "RMSE", "NMSE(dB)", "SAM(deg)"]]
    write_csv_artifact(env, csv_summary, "recon_summary.csv")

    tex_lines = [
        r"\begin{tabular}{c|c|c|c|c|c|c|c}",
        r"    \hline",
        r"    Rank & Method & Params & Scenes & CR & RMSE & NMSE(dB) & SAM(deg) \\",
        r"    \hline",
    ]
    for rank in map(_rank_text, RANK_ORDER):
        panel = summary.loc[summary["Rank"] == rank].copy()
        for idx, (_, row) in enumerate(panel.iterrows()):
            rank_cell = rank if idx == 0 else ""
            tex_lines.append(
                "    "
                + " & ".join(
                    [
                        rank_cell,
                        str(row["Method"]),
                        str(int(row["Params"])),
                        str(int(row["Scenes"])),
                        str(row["CR"]).replace("+-", r"$\pm$"),
                        str(row["RMSE"]).replace("+-", r"$\pm$"),
                        str(row["NMSE(dB)"]).replace("+-", r"$\pm$"),
                        str(row["SAM(deg)"]).replace("+-", r"$\pm$"),
                    ]
                )
                + r" \\"
            )
        tex_lines.append(r"    \hline")
    tex_lines[-1] = r"    \bottomrule"
    tex_lines.append(r"\end{tabular}")
    write_text_artifact(env, "\n".join(tex_lines) + "\n", "recon_summary.tex")

    pub_lines = [
        r"\begin{tabular}{@{}c l r c c c@{}}",
        r"\toprule",
        r"Rank & Method & Param. & RMSE$\downarrow$ & NMSE(dB)$\downarrow$ & SAM($^\circ$)$\downarrow$\\",
        r"\midrule",
    ]
    rank_texts = [_rank_text(rank) for rank in RANK_ORDER]
    for rank_idx, rank in enumerate(rank_texts):
        panel = summary.loc[summary["Rank"] == rank].copy().reset_index(drop=True)
        best_rmse = float(panel["RMSE_mean"].min())
        best_nmse = float(panel["NMSE_mean"].min())
        best_sam = float(panel["SAM_mean"].min())
        row_span = max(int(len(panel)), 1)
        for idx, (_, row) in enumerate(panel.iterrows()):
            rank_cell = ""
            if idx == 0:
                rank_math = f"${rank}$"
                if rank == _rank_text(MAIN_RANK):
                    rank_math = f"${rank}^{{\\star}}$"
                rank_cell = rf"\multirow{{{row_span}}}{{*}}{{{rank_math}}}"
            rmse_text = _pm_latex(float(row["RMSE_mean"]), float(row["RMSE_std"]), 4)
            nmse_text = _pm_latex(float(row["NMSE_mean"]), float(row["NMSE_std"]), 3)
            sam_text = _pm_latex(float(row["SAM_mean"]), float(row["SAM_std"]), 2)
            if np.isclose(float(row["RMSE_mean"]), best_rmse):
                rmse_text = rf"\textbf{{{rmse_text}}}"
            if np.isclose(float(row["NMSE_mean"]), best_nmse):
                nmse_text = rf"\textbf{{{nmse_text}}}"
            if np.isclose(float(row["SAM_mean"]), best_sam):
                sam_text = rf"\textbf{{{sam_text}}}"
            pub_lines.append(
                " & ".join(
                    [
                        rank_cell,
                        str(row["Method"]),
                        f"{int(row['Params']):,}",
                        rmse_text,
                        nmse_text,
                        sam_text,
                    ]
                )
                + r" \\"
            )
        if rank_idx < len(rank_texts) - 1:
            pub_lines.append(r"\midrule")
    pub_lines.append(r"\midrule")
    pub_lines.append(r"\multicolumn{6}{l}{\footnotesize $\star$ Main report rank in Sec.~5.3.2.}\\")
    pub_lines.append(r"\bottomrule")
    pub_lines.append(r"\end{tabular}")
    write_text_artifact(env, "\n".join(pub_lines) + "\n", "recon_summary_pub.tex")


def _significance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank in RANK_ORDER:
        panel = frame.loc[frame["rank"] == rank].copy()
        pivot = panel.pivot_table(index="scene_id", columns="method_name", values=["RMSE", "NMSE_dB", "SAM"], aggfunc="mean")
        for metric in ("RMSE", "NMSE_dB", "SAM"):
            diffs = (pivot[(metric, "tucker")] - pivot[(metric, "ntdpl")]).to_numpy(dtype=float)
            wins = int(np.sum(diffs > 0.0))
            ties = int(np.sum(np.isclose(diffs, 0.0)))
            losses = int(np.sum(diffs < 0.0))
            nonzero = diffs[~np.isclose(diffs, 0.0)]
            sign_p = 1.0 if nonzero.size == 0 else float(binomtest(int(np.sum(nonzero > 0.0)), nonzero.size, 0.5).pvalue)
            try:
                wilcoxon_p = 1.0 if nonzero.size == 0 else float(wilcoxon(nonzero, zero_method="wilcox").pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            rows.append(
                {
                    "Rank": _rank_text(rank),
                    "Metric": metric,
                    "Wins / Ties / Losses": f"{wins} / {ties} / {losses}",
                    "Win/Loss/Tie": f"{wins}/{losses}/{ties}",
                    "Mean Delta": float(diffs.mean()),
                    "Median Delta": float(np.median(diffs)),
                    "Sign Test P": sign_p,
                    "Wilcoxon P": wilcoxon_p,
                    "Rank-biserial": float(_rank_biserial(diffs)),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No rows available for original-space CAVE significance summary.")
    return summary


def significance_table() -> None:
    frame, env = _load_runs()
    summary = _significance_frame(frame)
    write_csv_artifact(env, summary, "significance.csv")

    tex_lines = [
        r"\begin{tabular}{c|c|c|c|c|c|c}",
        r"\toprule",
        r"Rank & Metric & Wins / Ties / Losses & Mean $\Delta$ & Median $\Delta$ & Sign test $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        tex_lines.append(
            " & ".join(
                [
                    f"${row['Rank']}$",
                    str(row["Metric"]),
                    str(row["Wins / Ties / Losses"]),
                    f"{float(row['Mean Delta']):.6f}",
                    f"{float(row['Median Delta']):.6f}",
                    _latex_pvalue(float(row["Sign Test P"])),
                    _latex_pvalue(float(row["Wilcoxon P"])),
                ]
            )
            + r" \\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text_artifact(env, "\n".join(tex_lines) + "\n", "significance_pub.tex")


def main_rank_significance_summary() -> None:
    frame, env = _load_runs()
    summary = _significance_frame(frame)
    main = summary.loc[summary["Rank"] == _rank_text(MAIN_RANK)].copy()
    write_csv_artifact(env, main, "recon_significance_summary.csv")

    tex_lines = [
        r"\begin{tabular}{c c c c c}",
        r"\toprule",
        r"Metric & Win/Loss/Tie & Median gain & Sign test $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for _, row in main.iterrows():
        tex_lines.append(
            " & ".join(
                [
                    str(row["Metric"]),
                    str(row["Win/Loss/Tie"]),
                    f"{float(row['Median Delta']):.4f}",
                    _latex_pvalue(float(row["Sign Test P"])),
                    _latex_pvalue(float(row["Wilcoxon P"])),
                ]
            )
            + r" \\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text_artifact(env, "\n".join(tex_lines) + "\n", "recon_significance_summary.tex")


def _make_dataset(scene_id: int, row: pd.Series) -> CAVEHSIData:
    dataset_kwargs = _cave_dataset_kwargs_from_row(row)
    dataset = CAVEHSIData(
        path=dataset_kwargs["path"],
        id=int(scene_id),
        target_shape=dataset_kwargs["target_shape"],
        crop_shape=dataset_kwargs["crop_shape"],
    )
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    return dataset


def _select_representation_row(frame: pd.DataFrame, *, scene_id: int, rank: tuple[int, int, int], method_name: str) -> pd.Series:
    panel = frame.loc[
        frame["scene_id"].eq(int(scene_id))
        & frame["rank"].map(lambda item: tuple(item) == tuple(rank))
        & frame["method_name"].eq(method_name)
    ].copy()
    if method_name == "ntdpl":
        panel = panel.loc[np.isclose(panel["p_max"], float(NTDPL_PMAX), atol=1e-12)].copy()
    panel = panel.sort_values("RMSE").reset_index(drop=True)
    if panel.empty:
        raise RuntimeError(f"Missing cave-representation row for scene={scene_id}, rank={rank}, method={method_name}.")
    return panel.iloc[0]


def _run_payload(frame: pd.DataFrame, scene_id: int, rank: tuple[int, int, int]) -> SceneRunPayload:
    row_t = _select_representation_row(frame, scene_id=scene_id, rank=rank, method_name="tucker")
    row_n = _select_representation_row(frame, scene_id=scene_id, rank=rank, method_name="ntdpl")
    dataset = _make_dataset(scene_id, row_t)
    original = np.asarray(dataset.get("eval").dense, dtype=np.float32)
    recon_tucker = _state_reconstruction(load_state_mat(_state_path_from_row(row_t)))
    recon_ntdpl = _state_reconstruction(load_state_mat(_state_path_from_row(row_n)))
    return SceneRunPayload(
        scene_id=int(scene_id),
        scene_name=str(getattr(dataset, "scene_name", f"scene-{scene_id}")),
        original=original,
        recon_tucker=recon_tucker,
        recon_ntdpl=recon_ntdpl,
    )


def _pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    band_count = cube.shape[-1]
    indices = [int(round((band_count - 1) * frac)) for frac in (0.75, 0.5, 0.2)]
    rgb = np.stack([cube[..., idx] for idx in indices], axis=-1)
    rgb = np.clip(rgb, 0.0, None)
    scale = float(np.max(rgb))
    if scale > 1e-12:
        rgb = rgb / scale
    return rgb


def _rmse_map(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    diff = np.asarray(original, dtype=np.float32) - np.asarray(reconstruction, dtype=np.float32)
    return np.sqrt(np.mean(diff * diff, axis=-1))


def _sam_map(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    ref = np.asarray(original, dtype=np.float32)
    rec = np.asarray(reconstruction, dtype=np.float32)
    numerator = np.sum(ref * rec, axis=-1)
    denominator = np.linalg.norm(ref, axis=-1) * np.linalg.norm(rec, axis=-1)
    cosine = np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _selected_pixel(payload: SceneRunPayload) -> tuple[int, int]:
    rmse_gain = _rmse_map(payload.original, payload.recon_tucker) - _rmse_map(payload.original, payload.recon_ntdpl)
    sam_gain = _sam_map(payload.original, payload.recon_tucker) - _sam_map(payload.original, payload.recon_ntdpl)
    rmse_scale = max(float(np.quantile(np.abs(rmse_gain), 0.95)), 1e-12)
    sam_scale = max(float(np.quantile(np.abs(sam_gain), 0.95)), 1e-12)
    score = rmse_gain / rmse_scale + 0.35 * (sam_gain / sam_scale)
    intensity = np.mean(payload.original, axis=-1)
    mask = (intensity >= float(np.quantile(intensity, 0.25))) & (intensity <= float(np.quantile(intensity, 0.95)))
    masked = np.where(mask, score, -np.inf)
    if not np.isfinite(masked).any():
        masked = score
    row, col = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(row), int(col)


def _display_downsample(image: np.ndarray, max_size: int = 256) -> tuple[np.ndarray, int]:
    if image.ndim < 2:
        return image, 1
    max_dim = max(int(image.shape[0]), int(image.shape[1]))
    if max_dim <= max_size:
        return image, 1
    stride = int(np.ceil(max_dim / max_size))
    return image[::stride, ::stride, ...], stride


def _main_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.loc[frame["rank"] == MAIN_RANK].copy()
    pivot = panel.pivot_table(index="scene_id", columns="method_name", values=["RMSE", "NMSE_dB", "SAM"], aggfunc="mean")
    rows: list[dict[str, float | int | str]] = []
    for scene_id in pivot.index:
        rows.append(
            {
                "Rank": _rank_text(MAIN_RANK),
                "scene_id": int(scene_id),
                "Scene": f"Scene {int(scene_id)}",
                "RMSE_tucker": float(pivot.loc[scene_id, ("RMSE", "tucker")]),
                "RMSE_ntdpl": float(pivot.loc[scene_id, ("RMSE", "ntdpl")]),
                "NMSE_dB_tucker": float(pivot.loc[scene_id, ("NMSE_dB", "tucker")]),
                "NMSE_dB_ntdpl": float(pivot.loc[scene_id, ("NMSE_dB", "ntdpl")]),
                "SAM_tucker": float(pivot.loc[scene_id, ("SAM", "tucker")]),
                "SAM_ntdpl": float(pivot.loc[scene_id, ("SAM", "ntdpl")]),
            }
        )
    paired = pd.DataFrame(rows).sort_values("scene_id").reset_index(drop=True)
    paired["RMSE_delta_tucker_minus_ntdpl"] = paired["RMSE_tucker"] - paired["RMSE_ntdpl"]
    paired["NMSE_dB_delta_tucker_minus_ntdpl"] = paired["NMSE_dB_tucker"] - paired["NMSE_dB_ntdpl"]
    paired["SAM_delta_tucker_minus_ntdpl"] = paired["SAM_tucker"] - paired["SAM_ntdpl"]
    paired["RMSE_gain"] = paired["RMSE_delta_tucker_minus_ntdpl"]
    paired["SAM_gain"] = paired["SAM_delta_tucker_minus_ntdpl"]
    return paired


def _select_main_scenes(pair_frame: pd.DataFrame, count: int = 3) -> tuple[int, ...]:
    if pair_frame.empty:
        raise RuntimeError("No scene-level rows available for CAVE reconstruction main figure.")
    ordered = pair_frame.sort_values("RMSE_gain", ascending=False).reset_index(drop=True)
    positive = ordered.loc[ordered["RMSE_gain"] > 0.0].copy()
    pool = positive if len(positive) >= count else ordered
    targets = np.linspace(0.0, 0.8, num=count)
    chosen: list[int] = []
    for target in targets:
        idx = int(round(target * (len(pool) - 1)))
        scene_id = int(pool.iloc[idx]["scene_id"])
        if scene_id not in chosen:
            chosen.append(scene_id)
    for scene_id in pool["scene_id"].tolist():
        if len(chosen) >= count:
            break
        sid = int(scene_id)
        if sid not in chosen:
            chosen.append(sid)
    for scene_id in ordered["scene_id"].tolist():
        if len(chosen) >= count:
            break
        sid = int(scene_id)
        if sid not in chosen:
            chosen.append(sid)
    return tuple(chosen[:count])


def _paired_scene_panel(ax: plt.Axes, *, pair_frame: pd.DataFrame, metric: str, xlabel: str) -> None:
    ordered = pair_frame.sort_values("scene_id", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered), dtype=float)
    left = ordered[f"{metric}_tucker"].to_numpy(dtype=float)
    right = ordered[f"{metric}_ntdpl"].to_numpy(dtype=float)
    gain = left - right

    for idx in range(len(ordered)):
        line_color = PALETTE.ntdpl if gain[idx] > 0.0 else PALETTE.rose
        ax.plot(
            [left[idx], right[idx]],
            [y[idx], y[idx]],
            color=line_color,
            linewidth=1.6,
            alpha=0.88,
            solid_capstyle="round",
            zorder=2,
        )

    ax.scatter(left, y, color=PALETTE.tucker, marker="s", s=26, alpha=0.95, zorder=3)
    ax.scatter(right, y, color=PALETTE.ntdpl, marker="o", s=28, alpha=0.98, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"S{int(scene_id)}" for scene_id in ordered["scene_id"]], fontsize=8.0)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.8, len(ordered) - 0.2)
    ax.invert_yaxis()
    style_axes(ax, grid=True)
    ax.xaxis.grid(True, color=PALETTE.grid, alpha=0.65, linewidth=0.6)
    ax.yaxis.grid(False)

    wins = int(np.sum(gain > 0.0))
    total = int(gain.size)
    mean_gain = float(gain.mean())
    text_fmt = "{:.4f}" if metric == "RMSE" else "{:.2f}"
    ax.text(
        0.98,
        0.98,
        f"wins: {wins}/{total}\nmean Δ: {text_fmt.format(mean_gain)}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.95},
    )


def _scene_improvement_sorted_bar_plot(pair_frame: pd.DataFrame, *, output_base: Any) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, metric, ylabel, title in (
        (axes[0], "RMSE_gain", r"$\Delta$RMSE", "RMSE improvement (sorted bar)"),
        (axes[1], "SAM_gain", r"$\Delta$SAM (deg)", "SAM improvement (sorted bar)"),
    ):
        ordered = pair_frame.sort_values(metric, ascending=False).reset_index(drop=True)
        x = np.arange(len(ordered))
        values = ordered[metric].to_numpy(dtype=float)
        colors = [PALETTE.ntdpl if value > 0.0 else PALETTE.rose for value in values]
        ax.bar(x, values, width=0.72, color=colors, edgecolor="none")
        ax.axhline(0.0, color=PALETTE.border, linewidth=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{int(scene_id)}" for scene_id in ordered["scene_id"]], fontsize=7.7)
        ax.set_xlabel("Sorted scenes")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=5)
        style_axes(ax, grid=True)
    fig.tight_layout()
    save_figure(fig, output_base, formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def scene_improvement_overview_plot() -> None:
    frame, env = _load_runs()
    pair_frame = _main_pair_frame(frame)
    write_csv_artifact(env, pair_frame, "scene_improvement_overview.csv")

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), sharey=True)
    _paired_scene_panel(axes[0], pair_frame=pair_frame, metric="RMSE", xlabel="RMSE")
    _paired_scene_panel(axes[1], pair_frame=pair_frame, metric="SAM", xlabel="SAM (deg)")
    axes[0].set_title("Scene-wise paired RMSE (main rank)", pad=5)
    axes[1].set_title("Scene-wise paired SAM (main rank)", pad=5)
    axes[0].set_ylabel("Scene")
    axes[1].set_ylabel("")
    handles = [
        Line2D([0], [0], marker="s", linestyle="None", color=PALETTE.tucker, markerfacecolor=PALETTE.tucker, label="Tucker"),
        Line2D([0], [0], marker="o", linestyle="None", color=PALETTE.ntdpl, markerfacecolor=PALETTE.ntdpl, label="NTD-PL"),
        Line2D([0, 1], [0, 0], color=PALETTE.ntdpl, linewidth=1.7, label="NTD-PL better"),
        Line2D([0, 1], [0, 0], color=PALETTE.rose, linewidth=1.7, label="Tucker better"),
    ]
    fig.legend(**legend_style(handles, [item.get_label() for item in handles], loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.03)))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_main_figure(env, fig, "cave_reconstruction_scene_gain")
    plt.close(fig)

    _scene_improvement_sorted_bar_plot(
        pair_frame,
        output_base=(env.artifacts_dir / "scene_improvement_overview_sorted_bar"),
    )


def _roi_from_payload(payload: SceneRunPayload, roi_size: int = 20) -> tuple[int, int, int, float]:
    reduction = _rmse_map(payload.original, payload.recon_tucker) - _rmse_map(payload.original, payload.recon_ntdpl)
    intensity = np.mean(payload.original, axis=-1)
    mask = intensity >= float(np.quantile(intensity, 0.25))
    masked = np.where(mask, reduction, -np.inf)
    if not np.isfinite(masked).any():
        masked = reduction
    row, col = np.unravel_index(int(np.argmax(masked)), masked.shape)
    h, w = reduction.shape
    side = max(8, min(int(roi_size), h, w))
    half = side // 2
    r0 = int(np.clip(row - half, 0, h - side))
    c0 = int(np.clip(col - half, 0, w - side))
    return r0, c0, side, float(reduction[row, col])


def visual_compare_plot() -> None:
    frame, env = _load_runs()
    pair_frame = _main_pair_frame(frame)
    available_scene_ids = set(pair_frame["scene_id"].astype(int).tolist())
    scene_ids = tuple(scene_id for scene_id in FOCUS_SCENES if int(scene_id) in available_scene_ids)
    if len(scene_ids) < len(FOCUS_SCENES):
        fallback = _select_main_scenes(pair_frame, count=3)
        scene_ids = tuple(dict.fromkeys([*scene_ids, *fallback]))[:3]
    payloads = [_run_payload(frame, scene_id, MAIN_RANK) for scene_id in scene_ids]

    error_tucker = [_rmse_map(item.original, item.recon_tucker) for item in payloads]
    error_ntdpl = [_rmse_map(item.original, item.recon_ntdpl) for item in payloads]
    reduction_maps = [err_t - err_n for err_t, err_n in zip(error_tucker, error_ntdpl, strict=False)]
    err_all = np.concatenate([np.ravel(item) for item in [*error_tucker, *error_ntdpl]])
    red_all = np.concatenate([np.ravel(item) for item in reduction_maps])
    error_vmax = max(float(np.quantile(err_all, 0.995)), 1e-6)
    reduction_vmax = max(float(np.quantile(np.abs(red_all), 0.995)), 1e-6)

    apply_theme()
    fig, axes = plt.subplots(len(payloads), 6, figsize=(11.2, 2.0 * len(payloads) + 0.15), constrained_layout=False)
    if len(payloads) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, payload in enumerate(payloads):
        _, _, _, roi_peak = _roi_from_payload(payload, roi_size=20)
        rmse_gain = float(pair_frame.loc[pair_frame["scene_id"] == payload.scene_id, "RMSE_gain"].iloc[0])
        columns = [
            ("Original", _pseudo_rgb(payload.original), "rgb"),
            ("Tucker", _pseudo_rgb(payload.recon_tucker), "rgb"),
            ("NTD-PL", _pseudo_rgb(payload.recon_ntdpl), "rgb"),
            ("Tucker error", error_tucker[row_idx], "error"),
            ("NTD-PL error", error_ntdpl[row_idx], "error"),
            ("Error reduction", reduction_maps[row_idx], "reduction"),
        ]
        for col_idx, (title, image, kind) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            display_image, stride = _display_downsample(np.asarray(image))
            if kind == "error":
                ax.imshow(display_image, cmap="magma", vmin=0.0, vmax=error_vmax)
            elif kind == "reduction":
                ax.imshow(display_image, cmap="RdBu_r", vmin=-reduction_vmax, vmax=reduction_vmax)
                ax.text(
                    0.03,
                    0.97,
                    f"max local Δ={roi_peak:.4f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7.4,
                    color=PALETTE.black,
                    bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.93},
                )
            else:
                ax.imshow(np.clip(display_image, 0.0, 1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{payload.scene_name}\nΔRMSE={rmse_gain:.4f}", rotation=90, labelpad=11)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.975, bottom=0.03, wspace=0.02, hspace=0.01)

    _save_main_figure(env, fig, "cave_reconstruction_visual_grid")
    plt.close(fig)


def spectral_curve_plot() -> None:
    frame, env = _load_runs()
    pair_frame = _main_pair_frame(frame)
    scene_ids = _select_main_scenes(pair_frame, count=3)
    payloads = [_run_payload(frame, scene_id, MAIN_RANK) for scene_id in scene_ids]

    apply_theme()
    fig, axes = plt.subplots(1, len(payloads), figsize=(8.9, 3.0), squeeze=False)
    handles = [
        Line2D([0], [0], **method_style("Ground truth"), label="Ground truth"),
        Line2D([0], [0], **method_style("Tucker"), label="Tucker"),
        Line2D([0], [0], **method_style("NTD-PL"), label="NTD-PL"),
    ]
    for idx, payload in enumerate(payloads):
        row, col = _selected_pixel(payload)
        bot_ax = axes[0, idx]

        band_axis = np.arange(1, payload.original.shape[-1] + 1)
        gt = payload.original[row, col, :]
        t_curve = payload.recon_tucker[row, col, :]
        n_curve = payload.recon_ntdpl[row, col, :]
        rmse_t = float(np.sqrt(np.mean((gt - t_curve) ** 2)))
        rmse_n = float(np.sqrt(np.mean((gt - n_curve) ** 2)))
        sam_t = float(_sam_map(gt.reshape(1, 1, -1), t_curve.reshape(1, 1, -1))[0, 0])
        sam_n = float(_sam_map(gt.reshape(1, 1, -1), n_curve.reshape(1, 1, -1))[0, 0])
        bot_ax.plot(band_axis, gt, **method_style("Ground truth"))
        bot_ax.plot(band_axis, t_curve, **method_style("Tucker"))
        bot_ax.plot(band_axis, n_curve, **method_style("NTD-PL"))
        bot_ax.set_xlabel("Band")
        bot_ax.set_title(payload.scene_name, pad=3)
        if idx == 0:
            bot_ax.set_ylabel("Reflectance")
        style_axes(bot_ax, grid=True)
        bot_ax.text(
            0.03,
            0.96,
            f"pixel=({row},{col})\nΔRMSE={rmse_t - rmse_n:.4f}\nΔSAM={sam_t - sam_n:.2f}°",
            transform=bot_ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.92},
        )

    fig.legend(**legend_style(handles, [h.get_label() for h in handles], loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.02)))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_main_figure(env, fig, "cave_reconstruction_spectra")
    plt.close(fig)
