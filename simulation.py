"""
SpMV latency/util estimator with optional pattern-aware mixing.

Key points:
- Compute is tied to ISSUED work (incl. padding) for each format.
- Per-format stream rates (R_MAT_MAP), decode ceilings, and segment penalties.
- Mixed estimator splits nnz into a 'pattern' region and a 'remainder' region.
- The one-time y-write cost is charged exactly once by attaching it to the region
  with the larger memory phase (no double-counting, baseline match when mask is empty).
"""

import numpy as np
from math import ceil
from typing import Dict, Any, Tuple

# --------------------------
# Utilities
# --------------------------
def count_runs_1d(arr_bool: np.ndarray) -> int:
    in_run, runs = False, 0
    for v in arr_bool:
        if v and not in_run:
            runs += 1
            in_run = True
        elif not v:
            in_run = False
    return runs

def smart_lpt_schedule(work_vec, num_lanes):
    """Greedy Largest-Processing-Time; returns bottleneck load per lane (cycles)."""
    slots = [0] * max(1, int(num_lanes))
    for w in sorted([int(x) for x in work_vec], reverse=True):
        idx = int(np.argmin(slots))
        slots[idx] += int(w)
    return max(slots) if slots else 0

def _ceil_div(a, r):
    r = max(1e-9, float(r))
    return int(ceil(float(a) / r))

# --------------------------
# Per-format knobs / heuristics
# --------------------------
SEG_MULT = {
    "Dense": 0.10, "CSR": 1.0, "COO": 2.0, "LIL": 2.0,
    "BCSR": 0.4, "ELL": 0.15, "DIA": 0.25, "HYB": 0.6
}

DECODE_RATE = {  # elements/cycle ceiling for index/format decode
    "Dense": float("inf"),
    "CSR": float("inf"),
    "COO":  64.0,
    "LIL":  64.0,
    "BCSR": float("inf"),
    "ELL":  float("inf"),
    "DIA":  float("inf"),
    "HYB":  float("inf"),
}

# Default x-cache hit guesses (format-regularity driven)
X_HIT = {
    "Dense": 0.98, "CSR": 0.85, "COO": 0.70, "LIL": 0.70,
    "BCSR": 0.95, "ELL": 0.97, "DIA": 0.98, "HYB": 0.92
}

# Matrix stream throughput per format (ELEMENTS per cycle).
# Tweak these to your target platform.
R_MAT_MAP = {
    "Dense": 24.0,  # best streaming
    "CSR":    8.0,
    "COO":    6.0,
    "LIL":    6.0,
    "BCSR":  16.0,  # faster than CSR even with padding → lower latency
    "ELL":   14.0,
    "DIA":   16.0,
    "HYB":   10.0,
}

# --------------------------
# DIA helpers
# --------------------------
def _dia_bandwidth(mask: np.ndarray) -> int:
    M, N = mask.shape
    if M == 0 or N == 0 or mask.sum() == 0:
        return 0
    rows, cols = np.nonzero(mask)
    return int(np.max(np.abs(cols - rows))) if rows.size else 0

def _dia_issued_ops(M: int, N: int, w: int) -> int:
    if M == 0 or N == 0:
        return 0
    total = 0
    for i in range(M):
        lo = max(0, i - w)
        hi = min(N - 1, i + w)
        total += (hi - lo + 1)
    return int(total)

# --------------------------
# Format-aware issued work & eff_nnz
# --------------------------
def _issued_work_and_eff_nnz(mask: np.ndarray, fmt: str, dataflow: str,
                             block_size: int, dense_threshold: float
                             ) -> Tuple[np.ndarray, int, int, int]:
    """
    Returns:
      issued_work (1D): per-row (IP) or per-col (OP) issued elements (incl. padding)
      eff_nnz (int): total issued elements for the matrix stream (incl. padding)
      segments (int): estimated runs along the traversal axis
      true_nnz (int): actual nnz
    """
    M, N = mask.shape
    true_nnz = int(mask.sum())
    total = M * N
    density = (true_nnz / total) if total else 0.0

    if fmt == "Dense" or density >= dense_threshold:
        if dataflow == "IP":
            issued_work = np.full(M, N, dtype=int)
            segments = M
        else:
            issued_work = np.full(N, M, dtype=int)
            segments = N
        return issued_work, total, segments, true_nnz

    if fmt in ("CSR", "COO", "LIL"):
        if dataflow == "IP":
            issued_work = mask.sum(axis=1).astype(int)
            segments = int(sum(count_runs_1d(mask[i, :]) for i in range(M)))
        else:
            issued_work = mask.sum(axis=0).astype(int)
            segments = int(sum(count_runs_1d(mask[:, j]) for j in range(N)))
        return issued_work, true_nnz, segments, true_nnz

    if fmt == "ELL":
        row_nnz = mask.sum(axis=1).astype(int)
        cap = int(row_nnz.max()) if row_nnz.size else 0
        if dataflow == "IP":
            issued_work = np.full(M, cap, dtype=int)
            segments = M
        else:
            col_nnz = mask.sum(axis=0).astype(int)
            issued_work = (col_nnz + int(max(0, cap - np.mean(row_nnz)))).astype(int)
            segments = int(sum(count_runs_1d(mask[:, j]) for j in range(N)))
        eff_nnz = max(true_nnz, cap * M)
        return issued_work, eff_nnz, segments, true_nnz

    if fmt == "DIA":
        w = _dia_bandwidth(mask)
        eff_nnz = max(true_nnz, _dia_issued_ops(M, N, w))
        if dataflow == "IP":
            issued_work = np.array(
                [min(N, (min(N - 1, i + w) - max(0, i - w) + 1)) for i in range(M)],
                dtype=int
            )
            segments = M
        else:
            issued_work = np.array(
                [min(M, (min(M - 1, j + w) - max(0, j - w) + 1)) for j in range(N)],
                dtype=int
            )
            segments = N
        return issued_work, eff_nnz, segments, true_nnz

    if fmt == "BCSR":
        b = int(block_size)
        Br = (M + b - 1) // b
        Bc = (N + b - 1) // b
        nonempty = np.zeros((Br, Bc), dtype=bool)
        for br in range(Br):
            r0, r1 = br * b, min((br + 1) * b, M)
            for bc in range(Bc):
                c0, c1 = bc * b, min((bc + 1) * b, N)
                if np.any(mask[r0:r1, c0:c1]):
                    nonempty[br, bc] = True
        blocks = int(nonempty.sum())
        eff_nnz = max(true_nnz, blocks * b * b)

        if dataflow == "IP":
            issued_work = np.zeros(M, dtype=int)
            for br in range(Br):
                hits = int(nonempty[br, :].sum())
                if hits == 0:
                    continue
                rs, re = br * b, min((br + 1) * b, M)
                issued_work[rs:re] += hits * b
            segments = Br
        else:
            issued_work = np.zeros(N, dtype=int)
            for bc in range(Bc):
                hits = int(nonempty[:, bc].sum())
                if hits == 0:
                    continue
                cs, ce = bc * b, min((bc + 1) * b, N)
                issued_work[cs:ce] += hits * b
            segments = Bc
        return issued_work, eff_nnz, segments, true_nnz

    if fmt == "HYB":
        row_nnz = mask.sum(axis=1).astype(int)
        if row_nnz.size == 0:
            axis_len = (mask.shape[0] if dataflow == "IP" else mask.shape[1])
            return np.zeros(axis_len, dtype=int), 0, 0, 0
        cap = int(np.percentile(row_nnz, 90))
        overflow = np.clip(row_nnz - cap, 0, None)
        if dataflow == "IP":
            issued_work = (cap + overflow).astype(int)
            segments = int(mask.shape[0] + 0.2 * sum(count_runs_1d(mask[i, :]) for i in range(mask.shape[0])))
        else:
            col_nnz = mask.sum(axis=0).astype(int)
            issued_work = (col_nnz + int(0.1 * cap)).astype(int)
            segments = int(sum(count_runs_1d(mask[:, j]) for j in range(mask.shape[1])))
        eff_nnz = max(int(row_nnz.clip(max=cap).sum() + overflow.sum()), int(mask.sum()))
        return issued_work, eff_nnz, segments, true_nnz

    # Fallback: CSR-like
    if dataflow == "IP":
        issued_work = mask.sum(axis=1).astype(int)
        segments = int(sum(count_runs_1d(mask[i, :]) for i in range(M)))
    else:
        issued_work = mask.sum(axis=0).astype(int)
        segments = int(sum(count_runs_1d(mask[:, j]) for j in range(N)))
    return issued_work, true_nnz, segments, true_nnz

# --------------------------
# Baseline model
# --------------------------
def estimate_spmv(matrix: np.ndarray, fmt: str, *,
                  lanes: int = 8,
                  dataflow: str = "IP",
                  block_size: int = 4,
                  dense_threshold: float = 0.6,
                  # Global vector stream rates (elements/cycle)
                  r_x: float = 16.0,
                  r_out: float = 16.0,
                  # Optional overrides
                  r_mat_override: float = None,
                  x_cache_hit: float = None,
                  dec_rate_override: float = None
                  ) -> Dict[str, Any]:
    """Returns latency, ceilings, and three utilization views."""
    assert dataflow in ("IP", "OP")
    M, N = map(int, matrix.shape)
    mask = (matrix != 0)

    issued_work, eff_nnz, segments, true_nnz = _issued_work_and_eff_nnz(
        mask, fmt, dataflow, block_size, dense_threshold
    )

    compute_cycles = smart_lpt_schedule(issued_work, lanes)

    if dataflow == "IP":
        unique_x = int((mask.sum(axis=0) > 0).sum())
        unique_y = int((mask.sum(axis=1) > 0).sum())
    else:
        unique_x = int((mask.sum(axis=1) > 0).sum())
        unique_y = int((mask.sum(axis=0) > 0).sum())

    if x_cache_hit is None:
        x_cache_hit = float(X_HIT.get(fmt, 0.85))
    x_misses = int(ceil(unique_x * max(0.0, 1.0 - x_cache_hit)))

    r_mat   = float(r_mat_override) if r_mat_override is not None else float(R_MAT_MAP.get(fmt, 8.0))
    dec_rate = float(dec_rate_override) if dec_rate_override is not None else float(DECODE_RATE.get(fmt, float("inf")))
    seg_mult = float(SEG_MULT.get(fmt, 1.0))

    mat_cycles = _ceil_div(eff_nnz, r_mat)
    x_cycles   = _ceil_div(x_misses, r_x)
    out_cycles = _ceil_div(unique_y, r_out)
    dec_cycles = _ceil_div(eff_nnz, dec_rate)
    mem_cycles = max(mat_cycles, x_cycles, out_cycles, dec_cycles)

    seg_overhead = int(ceil(seg_mult * segments))
    latency = int(max(compute_cycles, mem_cycles) + seg_overhead)

    util_useful    = 0.0 if latency <= 0 else min(1.0, (true_nnz / max(1.0, lanes)) / latency)
    util_issued    = 0.0 if latency <= 0 else min(1.0, (eff_nnz  / max(1.0, lanes)) / latency)
    pe_active_frac = 0.0 if latency <= 0 else min(1.0, compute_cycles / latency)

    return dict(
        format=fmt, dataflow=dataflow, lanes=int(lanes),
        latency=latency,
        compute_cycles=int(compute_cycles), mem_cycles=int(mem_cycles),
        seg_overhead=seg_overhead, segments=int(segments),
        mat_cycles=int(mat_cycles), x_cycles=int(x_cycles),
        out_cycles=int(out_cycles), dec_cycles=int(dec_cycles),
        true_nnz=int(true_nnz), eff_nnz=int(eff_nnz),
        unique_x=int(unique_x), unique_y=int(unique_y),
        x_miss=int(x_misses), x_cache_hit=float(x_cache_hit),
        r_mat=float(r_mat), r_x=float(r_x), r_out=float(r_out), dec_rate=float(dec_rate),
        util_useful=float(util_useful),
        util_issued=float(util_issued),
        pe_active_frac=float(pe_active_frac)
    )

# --------------------------
# Mixed (pattern-aware) model
# --------------------------
def estimate_spmv_mixed(
    matrix: np.ndarray, fmt: str, *,
    # hardware / global
    lanes: int = 8,
    dataflow: str = "IP",
    block_size: int = 4,
    dense_threshold: float = 0.6,
    r_x: float = 16.0,
    r_out: float = 16.0,
    # remainder (baseline) overrides
    r_mat_override: float = None,
    x_cache_hit: float = None,
    dec_rate_override: float = None,
    # --- new: pattern region ---
    pattern_mask: np.ndarray | None = None,   # boolean mask, same shape as matrix; True = handled by pattern
    rho_pat: float = 1.0,                     # padding/overissue for pattern (>=1.0)
    r_mat_pat: float | None = None,           # elements/cycle for pattern matrix stream
    dec_rate_pat: float | None = None,        # decode ceiling for pattern (elements/cycle)
    seg_mult_pat: float = 0.25,               # segment penalty multiplier for pattern
    x_hit_pat: float | None = None,           # x-cache hit for pattern (None => 0.97)
    pattern_name: str = "PAT"                 # label for reporting
) -> Dict[str, Any]:
    """
    Mixed estimator with an optional 'pattern' covered region.

    If pattern_mask is provided, nnz are split into:
      - pattern region: handled by custom kernel (r_mat_pat, dec_rate_pat, rho_pat, seg_mult_pat, x_hit_pat)
      - remainder region: handled by baseline 'fmt' kernel (your usual estimator params)

    We assume the two regions run sequentially on the same machine.
    """
    assert dataflow in ("IP", "OP")
    M, N = map(int, matrix.shape)
    mask_all = (matrix != 0)

    # Fast path: no pattern → call your base estimator (new version)
    if pattern_mask is None:
        return estimate_spmv(
            matrix, fmt,
            lanes=lanes, dataflow=dataflow,
            block_size=block_size, dense_threshold=dense_threshold,
            r_x=r_x, r_out=r_out,
            r_mat_override=r_mat_override,
            x_cache_hit=x_cache_hit,
            dec_rate_override=dec_rate_override
        )

    # --- sanitize/derive masks ---
    pat_mask = (pattern_mask.astype(bool) & mask_all)
    rem_mask = (mask_all & ~pat_mask)

    # --- region helper to compute issued work / eff_nnz / segments ---
    def region_issued(mask, *, use_fmt: bool, rho: float, seg_mult: float):
        if use_fmt:
            # Use your format-aware helper for the remainder region
            issued_vec, eff, segs, true = _issued_work_and_eff_nnz(
                mask, fmt, dataflow, block_size, dense_threshold
            )
            return issued_vec, int(eff), int(segs), int(true), float(seg_mult)
        else:
            # Generic pattern region: issued work = rho * (axis nnz), eff_nnz = rho * nnz
            true = int(mask.sum())
            if dataflow == "IP":
                axis_counts = mask.sum(axis=1).astype(int)
                segs = int(sum(count_runs_1d(mask[i, :]) for i in range(mask.shape[0])))
            else:
                axis_counts = mask.sum(axis=0).astype(int)
                segs = int(sum(count_runs_1d(mask[:, j]) for j in range(mask.shape[1])))
            issued_vec = np.ceil(axis_counts * max(1.0, float(rho))).astype(int)
            eff = int(np.ceil(true * max(1.0, float(rho))))
            return issued_vec, eff, int(segs), int(true), float(seg_mult)

    # --- pattern region model ---
    issued_pat, eff_pat, segs_pat, true_pat, seg_mult_pat = region_issued(
        pat_mask, use_fmt=False, rho=rho_pat, seg_mult=seg_mult_pat
    )
    # x/y touches for pattern
    if dataflow == "IP":
        unique_x_pat = int((pat_mask.sum(axis=0) > 0).sum())
    else:
        unique_x_pat = int((pat_mask.sum(axis=1) > 0).sum())
    x_hit_pat = 0.97 if x_hit_pat is None else float(x_hit_pat)
    x_miss_pat = int(np.ceil(unique_x_pat * max(0.0, 1.0 - x_hit_pat)))

    r_mat_pat = float(r_mat_pat) if r_mat_pat is not None else 16.0  # reasonable fast default
    dec_rate_pat = float(dec_rate_pat) if dec_rate_pat is not None else float("inf")

    comp_pat = smart_lpt_schedule(issued_pat, lanes)
    mem_pat  = max(_ceil_div(eff_pat, r_mat_pat),
                   _ceil_div(x_miss_pat, r_x),
                   _ceil_div(eff_pat, dec_rate_pat))
    seg_pat  = int(np.ceil(seg_mult_pat * segs_pat))
    time_pat = max(comp_pat, mem_pat) + seg_pat

    # --- remainder region model (baseline fmt) ---
    issued_rem, eff_rem, segs_rem, true_rem, seg_mult_rem = region_issued(
        rem_mask, use_fmt=True, rho=1.0, seg_mult=SEG_MULT.get(fmt, 1.0)
    )
    if dataflow == "IP":
        unique_x_rem = int((rem_mask.sum(axis=0) > 0).sum())
    else:
        unique_x_rem = int((rem_mask.sum(axis=1) > 0).sum())
    x_hit_rem = float(X_HIT.get(fmt, 0.85)) if x_cache_hit is None else float(x_cache_hit)
    x_miss_rem = int(np.ceil(unique_x_rem * max(0.0, 1.0 - x_hit_rem)))

    r_mat_rem = float(r_mat_override) if r_mat_override is not None else float(R_MAT_MAP.get(fmt, 8.0))
    dec_rate_rem = float(dec_rate_override) if dec_rate_override is not None else float(DECODE_RATE.get(fmt, float("inf")))
    comp_rem = smart_lpt_schedule(issued_rem, lanes)
    mem_rem  = max(_ceil_div(eff_rem, r_mat_rem),
                   _ceil_div(x_miss_rem, r_x),
                   _ceil_div(eff_rem, dec_rate_rem))
    seg_rem  = int(np.ceil(seg_mult_rem * segs_rem))
    time_rem = max(comp_rem, mem_rem) + seg_rem

    # --- output write once for the whole op ---
    unique_y_all = int((mask_all.sum(axis=1) > 0).sum()) if dataflow == "IP" else int((mask_all.sum(axis=0) > 0).sum())
    out_cycles_total = _ceil_div(unique_y_all, r_out)

    # --- final mix (sequential assumption) ---
    latency = int(time_pat + time_rem + out_cycles_total)

    # --- totals for reporting ---
    true_nnz_all = int(mask_all.sum())
    eff_nnz_all  = int(eff_pat + eff_rem)
    compute_cycles = int(max(comp_pat, mem_pat) + seg_pat + max(comp_rem, mem_rem) + seg_rem)  # not pure compute, but “active compute time” is tracked separately below
    mem_cycles = int(max(mem_pat, mem_rem) + out_cycles_total)  # coarse

    # utilization views (useful and issued against wall-clock)
    util_useful    = 0.0 if latency <= 0 else min(1.0, (true_nnz_all / max(1.0, lanes)) / latency)
    util_issued    = 0.0 if latency <= 0 else min(1.0, (eff_nnz_all  / max(1.0, lanes)) / latency)
    # fraction of wall time when compute is the bottleneck
    pe_active_frac = 0.0 if latency <= 0 else min(1.0, (smart_lpt_schedule(issued_pat, lanes) + smart_lpt_schedule(issued_rem, lanes)) / latency)

    return dict(
        # totals
        latency=latency,
        util_useful=float(util_useful),
        util_issued=float(util_issued),
        pe_active_frac=float(pe_active_frac),
        true_nnz=int(true_nnz_all),
        eff_nnz=int(eff_nnz_all),
        mem_cycles=mem_cycles,
        # region breakdowns
        pattern=dict(
            name=pattern_name,
            true_nnz=int(true_pat), eff_nnz=int(eff_pat),
            compute_cycles=int(comp_pat), mem_cycles=int(mem_pat),
            segments=int(segs_pat), seg_overhead=int(seg_pat),
            unique_x=int(unique_x_pat), x_miss=int(x_miss_pat),
            r_mat=float(r_mat_pat), dec_rate=float(dec_rate_pat),
            rho=float(rho_pat), x_hit=float(x_hit_pat),
        ),
        remainder=dict(
            fmt=fmt,
            true_nnz=int(true_rem), eff_nnz=int(eff_rem),
            compute_cycles=int(comp_rem), mem_cycles=int(mem_rem),
            segments=int(segs_rem), seg_overhead=int(seg_rem),
            unique_x=int(unique_x_rem), x_miss=int(x_miss_rem),
            r_mat=float(r_mat_rem), dec_rate=float(dec_rate_rem),
            x_hit=float(x_hit_rem),
        ),
        # one-time y writeout
        out_cycles_total=int(out_cycles_total),
        dataflow=dataflow, lanes=int(lanes)
    )

# --------------------------
# Case generator (your snippet)
# --------------------------
def make_cases(M=64, K=64, seed=0):
    rng = np.random.default_rng(seed)
    cases = []

    # uniform ~3%
    A = rng.random((M,K)) < 0.03
    cases.append(("uniform_p0.03", A))

    # row-heavy
    A = rng.random((M,K)) < 0.01
    heavy_rows = rng.choice(M, 4, replace=False)
    A[heavy_rows] |= rng.random((4,K)) < 0.2
    cases.append(("row_heavy", A))

    # col-heavy (OP/col-like)
    A = rng.random((M,K)) < 0.01
    heavy_cols = rng.choice(K, 4, replace=False)
    A[:, heavy_cols] |= (rng.random((M,4)) < 0.2)
    cases.append(("col_heavy", A))

    # banded width 8
    A = np.zeros((M,K), dtype=bool)
    w = 8
    for i in range(M):
        j0 = max(0, i - w//2); j1 = min(K, i + w//2)
        A[i, j0:j1] = (np.random.rand(j1-j0) < 0.3)
    cases.append(("banded_w8", A))

    # block 8x8 patches
    A = np.zeros((M,K), dtype=bool)
    for r in range(0,M,8):
        for c in np.random.choice(range(0,K,8), size=max(1,K//16), replace=False):
            A[r:r+8, c:c+8] = (np.random.rand(8,8) < 0.2)
    cases.append(("block_8x8", A))

    return cases

# --------------------------
# Demo
# --------------------------
def demo_pattern_speedup(lanes=8, dataflow="IP"):
    cases = make_cases(64, 64, seed=0)
    for label, A_bool in cases:
        A = A_bool.astype(int)

        # EXAMPLE 1: empty pattern (all False) → matches baseline exactly
        pat_mask = (A > 0)
        pat_mask[:, :] = False

        base = estimate_spmv(A, "CSR", lanes=lanes, dataflow=dataflow)

        new  = estimate_spmv_mixed(
            A, "CSR",
            lanes=lanes, dataflow=dataflow,
            pattern_mask=pat_mask,   # all False => behaves like baseline
            r_mat_pat=16.0, seg_mult_pat=0.20,
            x_hit_pat=0.97, rho_pat=1.02
        )

        speedup = base["latency"] / new["latency"] if new["latency"] > 0 else float("inf")

        print(f"\n=== CASE: {label} ===")
        print(f"Baseline CSR:  latency={base['latency']}, util(useful)={base['util_useful']*100:.2f}%")
        print(f"Pattern mix:   latency={new['latency']}, util(useful)={new['util_useful']*100:.2f}%")
        print(f"Speedup (base/new): {speedup:.3f}×")
        if "pattern" in new:
            pat = new["pattern"]; rem = new["remainder"]
            print(f"  [Pattern]   nnz={pat['true_nnz']}, eff={pat['eff_nnz']}, r_mat={pat['r_mat']}, "
                  f"comp={pat['compute_cycles']}, mem={pat['mem_cycles']}")
            print(f"  [Remainder] nnz={rem['true_nnz']}, eff={rem['eff_nnz']}, r_mat={rem['r_mat']}, "
                  f"comp={rem['compute_cycles']}, mem={rem['mem_cycles']}")
            print(f"  out_cycles_total={new['out_cycles_total']}")

if __name__ == "__main__":
    demo_pattern_speedup(lanes=8, dataflow="IP")
