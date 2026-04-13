import math
from typing import Sequence, Tuple


IntSeq = Sequence[int]


def _check_inputs(shape: IntSeq, tucker_rank: IntSeq) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    shape = tuple(int(x) for x in shape)
    tucker_rank = tuple(int(r) for r in tucker_rank)

    if len(shape) != len(tucker_rank):
        raise ValueError("shape and tucker_rank must have the same length")
    if len(shape) < 2:
        raise ValueError("This helper assumes tensor order >= 2")
    if any(x <= 0 for x in shape):
        raise ValueError("All entries in shape must be positive")
    if any(r <= 0 for r in tucker_rank):
        raise ValueError("All entries in tucker_rank must be positive")

    return shape, tucker_rank


def tucker_param_count(shape: IntSeq, tucker_rank: IntSeq) -> int:
    shape, tucker_rank = _check_inputs(shape, tucker_rank)
    return sum(i * r for i, r in zip(shape, tucker_rank)) + math.prod(tucker_rank)


def cp_param_count(shape: IntSeq, cp_rank: int, include_weights: bool = False) -> int:
    if cp_rank < 1:
        raise ValueError("cp_rank must be >= 1")
    p = cp_rank * sum(shape)
    if include_weights:
        p += cp_rank
    return p


def tt_param_count(shape: IntSeq, tt_rank: IntSeq) -> int:
    shape = tuple(int(x) for x in shape)
    tt_rank = tuple(int(r) for r in tt_rank)
    if len(tt_rank) != len(shape) + 1:
        raise ValueError("TT rank must have length len(shape)+1")
    return sum(tt_rank[k] * shape[k] * tt_rank[k + 1] for k in range(len(shape)))


def tr_param_count(shape: IntSeq, tr_rank: IntSeq) -> int:
    shape = tuple(int(x) for x in shape)
    tr_rank = tuple(int(r) for r in tr_rank)
    if len(tr_rank) != len(shape) + 1:
        raise ValueError("TR rank must have length len(shape)+1")
    if tr_rank[0] != tr_rank[-1]:
        raise ValueError("TR rank must satisfy rank[0] == rank[-1]")
    return sum(tr_rank[k] * shape[k] * tr_rank[k + 1] for k in range(len(shape)))


def _best_by_param_count(candidates, target, param_fn):
    best = None
    best_err = float("inf")
    best_params = None
    for cand in candidates:
        p = param_fn(cand)
        err = abs(p - target)
        if err < best_err:
            best = cand
            best_err = err
            best_params = p
    return best, best_params


def cp_rank_from_tucker(shape: IntSeq, tucker_rank: IntSeq, include_weights: bool = False) -> int:
    shape, tucker_rank = _check_inputs(shape, tucker_rank)
    target = tucker_param_count(shape, tucker_rank)

    denom = sum(shape) + (1 if include_weights else 0)
    r0 = max(1, round(target / denom))

    candidates = sorted(set(max(1, r0 + d) for d in range(-4, 5)))
    best_rank, _ = _best_by_param_count(
        candidates,
        target,
        lambda r: cp_param_count(shape, r, include_weights=include_weights),
    )
    if best_rank is None:
        raise ValueError("Failed to find a valid CP rank proposal")
    return int(best_rank)


def _tt_validate_rank(shape: Tuple[int, ...], proposed_rank: IntSeq) -> list[int]:
    n = len(shape)
    rank = list(int(r) for r in proposed_rank)

    if len(rank) != n + 1:
        raise ValueError("TT proposed_rank must have length len(shape)+1")
    rank[0] = 1
    rank[-1] = 1

    out = rank[:]
    out[0] = 1
    out[-1] = 1

    for k in range(n - 1):
        n_row = out[k] * shape[k]
        n_col = math.prod(shape[k + 1 :])
        out[k + 1] = min(out[k + 1], n_row, n_col)

    out[0] = 1
    out[-1] = 1
    return [int(x) for x in out]


def tt_rank_from_tucker(shape: IntSeq, tucker_rank: IntSeq) -> list[int]:
    shape, tucker_rank = _check_inputs(shape, tucker_rank)
    target = tucker_param_count(shape, tucker_rank)
    n = len(shape)

    if n == 2:
        s0 = max(1, round(target / (shape[0] + shape[1])))
    else:
        a = sum(shape[1:-1])
        b = shape[0] + shape[-1]
        if a == 0:
            s0 = max(1, round(target / max(1, b)))
        else:
            s0 = max(1, round((-b + math.sqrt(b * b + 4 * a * target)) / (2 * a)))

    s_candidates = set(range(1, max(8, 4 * s0 + 9)))
    s_candidates.update({1, s0, max(1, s0 // 2), 2 * s0 + 1})

    candidate_ranks = []
    for s in sorted(s_candidates):
        proposed = [1] + [s] * (n - 1) + [1]
        validated = _tt_validate_rank(shape, proposed)
        candidate_ranks.append(tuple(validated))

    candidate_ranks = sorted(set(candidate_ranks))

    best_rank, _ = _best_by_param_count(
        candidate_ranks,
        target,
        lambda r: tt_param_count(shape, r),
    )
    if best_rank is None:
        raise ValueError("Failed to find a valid TT rank proposal")
    return list(best_rank)


def _tr_validate_rank_mode0(shape: Tuple[int, ...], proposed_rank: IntSeq) -> list[int]:
    n = len(shape)
    rank = list(int(r) for r in proposed_rank)

    if len(rank) != n + 1:
        raise ValueError("TR proposed_rank must have length len(shape)+1")
    if rank[0] != rank[-1]:
        raise ValueError("TR proposed_rank must satisfy rank[0] == rank[-1]")

    cap0 = min(shape[0], math.prod(shape[1:]))
    if rank[0] * rank[1] > cap0:
        raise ValueError(
            f"TR infeasible for default mode=0: rank[0]*rank[1]={rank[0] * rank[1]} > {cap0}"
        )

    out = rank[:]
    for k in range(1, n - 1):
        n_row = out[k] * shape[k]
        n_col = math.prod(shape[k + 1 :]) * out[0]
        out[k + 1] = min(out[k + 1], n_row, n_col)

    out[-1] = out[0]
    return [int(x) for x in out]


def tr_rank_from_tucker(shape: IntSeq, tucker_rank: IntSeq) -> list[int]:
    shape, tucker_rank = _check_inputs(shape, tucker_rank)
    target = tucker_param_count(shape, tucker_rank)
    n = len(shape)

    cap0 = min(shape[0], math.prod(shape[1:]))
    best_rank = None
    best_err = float("inf")

    for s in range(1, cap0 + 1):
        c_max = cap0 // s
        if c_max < 1:
            break

        a = sum(shape[1:-1])
        b = shape[0] + shape[-1]

        if b > 0:
            c_guess = round((target - a * s * s) / max(1, b * s))
        else:
            c_guess = 1

        cands = {1, c_max}
        for dc in range(-3, 4):
            c = c_guess + dc
            if 1 <= c <= c_max:
                cands.add(c)

        for c in cands:
            proposed = [c] + [s] * (n - 1) + [c]
            validated = _tr_validate_rank_mode0(shape, proposed)
            p = tr_param_count(shape, validated)
            err = abs(p - target)
            if err < best_err:
                best_err = err
                best_rank = validated

    if best_rank is None:
        raise ValueError("Failed to find a valid TR rank proposal")
    return list(best_rank)
