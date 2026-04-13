"""
Benchmark script to compare Tucker vs. original NTDPL vs. optimized NTDPL.

Usage:
    python benchmark_ntdpl.py --tensor_sizes 100 200 300 --ranks 10 15 20
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path
import sys
from tensorly.decomposition import tucker as tucker_decompose

# Add workspace to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

from src.methods.ntdpl import ntdpl as ntdpl_original
try:
    from src.ntdpl import ntdpl_optimized
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    print("Warning: Optimized version not available")


def benchmark_tucker(X, rank, n_iter_max=50, random_state=0):
    """Run Tucker decomposition as linear baseline."""
    start_time = time.time()
    try:
        core, factors = tucker_decompose(
            X,
            rank=rank,
            n_iter_max=n_iter_max,
            init="svd",
            random_state=random_state,
        )
        elapsed = time.time() - start_time
        return elapsed, (core, factors), None
    except Exception as e:
        elapsed = time.time() - start_time
        return elapsed, None, str(e)


def create_synthetic_tensor(shape, rank, random_state=42):
    """
    Create a synthetic low-rank tensor for benchmarking.
    
    Tensor = Tucker decomposition with random factors
    """
    rng = np.random.default_rng(random_state)
    
    # Create random factors
    factors = [rng.normal(size=(n, r), scale=0.1) for n, r in zip(shape, rank)]
    
    # Create random core
    core = rng.normal(size=rank, scale=0.1)
    
    # Reconstruct tensor
    from tensorly.tucker_tensor import tucker_to_tensor
    tensor = tucker_to_tensor((core, factors))
    
    return np.asarray(tensor, dtype=np.float32)


def benchmark_method(
    method_func,
    X,
    rank,
    n_iter_max=50,
    p_max=3,
    **kwargs
):
    """Run one benchmark iteration."""
    start_time = time.time()
    
    try:
        result = method_func(
            X=X,
            rank=rank,
            init_n_iter_max=10,
            p_max=p_max,
            n_iter_max=n_iter_max,
            use_continuation=False,
            factor_normalize=True,
            lr_core=0.001,
            lr_factors=0.001,
            lambda_core=0.0,
            lambda_factors=0.0,
            lambda_beta=0.0,
            beta_update_method="moments_normal_eq",
            mask=None,
            init="tucker",
            random_state=0,
            beta_update_interval=5,
            return_history=False,
            **kwargs
        )
        elapsed = time.time() - start_time
        return elapsed, result, None
    except Exception as e:
        elapsed = time.time() - start_time
        return elapsed, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Benchmark NTDPL methods")
    parser.add_argument("--tensor_sizes", type=int, nargs="+", default=[100, 200], 
                       help="Tensor dimensions (cube)")
    parser.add_argument("--ranks", type=int, nargs="+", default=[10, 20],
                       help="Decomposition ranks")
    parser.add_argument("--n_iter", type=int, default=50, help="Iterations")
    parser.add_argument("--p_max", type=int, default=3, help="Max polynomial degree")
    parser.add_argument("--n_repeat", type=int, default=3, help="Repetitions")
    parser.add_argument("--output", type=str, default="benchmark_results.txt",
                       help="Output file")
    parser.add_argument("--csv_output", type=str, default="benchmark_results.csv",
                       help="CSV output file")
    
    args = parser.parse_args()
    
    # Prepare output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("NTDPL Optimization Benchmark")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Test parameters
    test_configs = []
    for size in args.tensor_sizes:
        shape = (size, size, size)
        for rank in args.ranks:
            rank_tuple = (rank, rank, rank)
            test_configs.append((shape, rank_tuple))
    
    results_table = []
    results_rows = []
    results_table.append([
        "Tensor Shape",
        "Rank",
        "Tucker (s)",
        "Original (s)",
        "Optimized (s)",
        "NTDPL/Tucker",
        "Speedup",
        "Valid"
    ])
    results_table.append(["-" * 15, "-" * 10, "-" * 12, "-" * 14, "-" * 14, "-" * 12, "-" * 10, "-" * 8])
    
    print("\nRunning benchmarks...")
    print("(This may take several minutes)\n")
    
    for shape, rank in test_configs:
        print(f"Testing {shape} with rank {rank}...", end="", flush=True)
        
        # Create synthetic tensor
        X = create_synthetic_tensor(shape, rank)
        
        times_tucker = []
        times_original = []
        times_optimized = []
        
        for rep in range(args.n_repeat):
            elapsed, _, err = benchmark_tucker(
                X,
                rank,
                n_iter_max=args.n_iter,
                random_state=rep,
            )
            if err is None:
                times_tucker.append(elapsed)

            # Original method
            elapsed, _, err = benchmark_method(
                ntdpl_original, X, rank,
                n_iter_max=args.n_iter,
                p_max=args.p_max
            )
            if err is None:
                times_original.append(elapsed)
            
            # Optimized method
            if OPTIMIZED_AVAILABLE:
                elapsed, _, err = benchmark_method(
                    ntdpl_optimized, X, rank,
                    n_iter_max=args.n_iter,
                    p_max=args.p_max,
                )
                if err is None:
                    times_optimized.append(elapsed)
        
        # Compute statistics
        if times_tucker:
            avg_tucker = np.mean(times_tucker)
        else:
            avg_tucker = float('nan')

        if times_original:
            avg_orig = np.mean(times_original)
        else:
            avg_orig = float('nan')
        
        if times_optimized:
            avg_opt = np.mean(times_optimized)
        else:
            avg_opt = float('nan')
        
        if not np.isnan(avg_orig) and not np.isnan(avg_tucker) and avg_tucker > 0:
            ntdpl_vs_tucker = avg_orig / avg_tucker
        else:
            ntdpl_vs_tucker = float('nan')

        if not np.isnan(avg_orig) and not np.isnan(avg_opt) and avg_opt > 0:
            speedup = avg_orig / avg_opt
            valid = "OK"
        else:
            speedup = float('nan')
            valid = "ERR"
        
        results_table.append([
            f"{shape[0]}^3",
            f"{rank[0]}",
            f"{avg_tucker:.2f}",
            f"{avg_orig:.2f}",
            f"{avg_opt:.2f}",
            f"{ntdpl_vs_tucker:.2f}x" if not np.isnan(ntdpl_vs_tucker) else "N/A",
            f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A",
            valid
        ])

        results_rows.append({
            "tensor_shape": f"{shape[0]}^3",
            "rank": int(rank[0]),
            "tucker_sec": float(avg_tucker),
            "ntdpl_sec": float(avg_orig),
            "ntdpl_optimized_sec": float(avg_opt),
            "ntdpl_vs_tucker": float(ntdpl_vs_tucker),
            "speedup": float(speedup),
            "valid": valid,
        })
        
        print(f" Tucker: {avg_tucker:.2f}s, Original: {avg_orig:.2f}s, Optimized: {avg_opt:.2f}s, Speedup: {speedup:.2f}x"
              if not np.isnan(speedup) else f" Error")
    
    # Print table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    col_widths = [max(len(row[i]) for row in results_table) for i in range(len(results_table[0]))]
    for row in results_table:
        print("  ".join(f"{cell:{width}}" for cell, width in zip(row, col_widths)))
    
    print("\nNote: Speedup = Original Time / Optimized Time")
    print("      (Higher is better)")
    
    if OPTIMIZED_AVAILABLE:
        avg_speedup = np.nanmean([float(row[6].replace('x', '')) for row in results_table[2:] if row[6] != 'N/A'])
        print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write('\n'.join(output_lines))
        f.write('\n\n')
        col_widths = [max(len(row[i]) for row in results_table) for i in range(len(results_table[0]))]
        for row in results_table:
            f.write("  ".join(f"{cell:{width}}" for cell, width in zip(row, col_widths)))
            f.write('\n')

    csv_fields = [
        "tensor_shape",
        "rank",
        "tucker_sec",
        "ntdpl_sec",
        "ntdpl_optimized_sec",
        "ntdpl_vs_tucker",
        "speedup",
        "valid",
    ]
    with open(args.csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(results_rows)
    
    print(f"\nResults saved to {args.output}")
    print(f"CSV saved to {args.csv_output}")


if __name__ == "__main__":
    main()
