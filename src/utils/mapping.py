from typing import List


def beta2dict(beta: List, p_max: int) -> dict:
    p = len(beta) - 1
    for _ in range(p_max - p):
        beta.append(0.0)
    return {f"beta_{i}": v for i, v in enumerate(beta)}
