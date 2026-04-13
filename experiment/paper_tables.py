from __future__ import annotations

from .config import get_env
from .utils.paper import write_text_artifact


def feature_placeholder_tables() -> None:
    """Write the placeholder feature tables used in the paper LaTeX.

    These tables intentionally keep the same English/math formatting and the
    same placeholder values (\texttt{x}) as the LaTeX draft, but they are
    generated and synced from experiment so the LaTeX source can use
    \\input{inputs/*.tex} everywhere.
    """

    env = get_env("paper-tables")

    feature_reconstruction = "\n".join(
        [
            r"\begin{tabular}{c|ccc|ccc|ccc}",
            r"    \hline",
            r"    压缩比 & \multicolumn{3}{c|}{Tucker} & \multicolumn{3}{c|}{Feature-AE} & \multicolumn{3}{c}{NTD-PL} \\",
            r"    \hline",
            r"     & NMSE$\downarrow$ & CosSim$\uparrow$ & 时间$\downarrow$ & NMSE$\downarrow$ & CosSim$\uparrow$ & 时间$\downarrow$ & NMSE$\downarrow$ & CosSim$\uparrow$ & 时间$\downarrow$ \\",
            r"    \hline",
            r"    $10\times$ & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} \\",
            r"    $20\times$ & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} \\",
            r"    $40\times$ & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} \\",
            r"    \hline",
            r"\end{tabular}",
            "",
        ]
    )

    feature_semantic = "\n".join(
        [
            r"\begin{tabular}{c|cc|cc|cc}",
            r"    \hline",
            r"    方法 & \multicolumn{2}{c|}{原始特征} & \multicolumn{2}{c|}{Tucker} & \multicolumn{2}{c}{NTD-PL} \\",
            r"    \hline",
            r"     & CosSim$\uparrow$ & Top-1$\uparrow$ & CosSim$\uparrow$ & Top-1$\uparrow$ & CosSim$\uparrow$ & Top-1$\uparrow$ \\",
            r"    \hline",
            r"    $10\times$ & 1.0000 & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} \\",
            r"    $20\times$ & 1.0000 & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} \\",
            r"    $40\times$ & 1.0000 & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} & \texttt{x} \\",
            r"    \hline",
            r"\end{tabular}",
            "",
        ]
    )

    rec_path, rec_latex = write_text_artifact(
        env, feature_reconstruction, artifact_name="feature_reconstruction_table.tex"
    )
    sem_path, sem_latex = write_text_artifact(env, feature_semantic, artifact_name="feature_semantic_table.tex")

    print(f"Saved: {rec_path}")
    print(f"Synced: {rec_latex}")
    print(f"Saved: {sem_path}")
    print(f"Synced: {sem_latex}")
