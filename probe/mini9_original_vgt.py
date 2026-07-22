"""The ORIGINAL VGT (Curry et al. robust variant, as ported in stratified_knobs_probe.py:vgt_dim)
applied the ORIGINAL way: collect ONE big point cloud, then for each probe count-in-ball up to the
MEDIAN distance and take the median of sliding-window log-log slopes. No gap-cut, no nearest-K.

We run it two ways on the same rollout bank:
  - distances from the LEARNED attention distance (our stratifying metric)
  - distances from the Euclidean embedding (the literal original: L2 on an embedding)
and print our gap-cut/nearest-K reading (dim_from_bank) alongside for reference.

Point: the original assumes a LOCALLY-sampled cloud (or `sub` restricted to one stratum). On a GLOBAL
bank the median-distance cap reaches across strata, so this shows whether the original method survives
a global cloud + a stratifying metric, or whether the gap-cut/locality step is actually necessary.
"""
import argparse, numpy as np, torch
from mini7_local_vgt import NAG, G, Kk, AttnDist, Euclid, train, rand_states
from mini8_bank import collect_bank, dist_all, dim_from_bank, pick_probes

def _orig_estimator(d, num_radii=40, window=5):
    """the ORIGINAL vgt_dim estimator (unchanged): count-in-ball, log-spaced radii up to the median
    distance of the (given) cloud, dim = median of sliding-window log-log slopes."""
    N = d.size
    if N < 20: return np.nan
    rmin, rmax = d[0], d[int(N * 0.5)]                       # median cap of THIS cloud
    if not (rmax > rmin): return np.nan
    radii = np.logspace(np.log10(rmin), np.log10(rmax), num_radii)
    counts = np.searchsorted(d, radii, side="left").astype(float)
    valid = counts > 10
    if valid.sum() < window + 1: return np.nan
    lr, lc = np.log(radii[valid]), np.log(counts[valid])
    slopes = [np.polyfit(lr[i:i + window], lc[i:i + window], 1)[0] for i in range(len(lr) - window)]
    return float(np.median(slopes)) if slopes else np.nan

def vgt_original(d):                                          # original: median cap of the GLOBAL cloud
    return _orig_estimator(np.sort(d[d > 1e-9]))

def _localcut(d, kmax=4000, gapthr=0.8):
    """replace the global median cap: restrict to the LOCAL neighbourhood (nearest-K, then the
    free/frozen gap) so the original estimator's median cap is taken over the local cloud."""
    d = np.sort(d[d > 1e-9])
    if d.size < 40: return d[:0]
    d = d[:min(kmax, d.size)]
    ld = np.log(d); gap = ld[1:] - ld[:-1]; lo, hi = 20, int(0.9 * d.size)
    cand = np.where(gap[lo:hi] > gapthr)[0]
    return d[:lo + int(cand[0]) + 1] if cand.size else d

def vgt_original_local(d):                                    # original estimator, median cap of the LOCAL cut
    return _orig_estimator(_localcut(d))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed); attn = train(AttnDist(), args.steps, rng)
    torch.manual_seed(args.seed); np.random.seed(args.seed); eucl = train(Euclid(), args.steps, rng)

    roll = collect_bank(2500, 50, rng)                       # 125k rollout states (cloud + probe source)
    print(f"cloud = {len(roll)} rollout states, reused for all probes; 8 probes per key\n")
    print(f"{'':<6}{'orig[median]':>14}{'orig[localcut]':>16}{'rank[localcut]':>16}   (attn distance)")
    for m in (1, 2, 3):
        probes = pick_probes(roll, m, 8, rng); og, ol, rk = [], [], []
        for p in probes:
            d = dist_all(attn, p, roll)
            og.append(vgt_original(d))                        # original estimator + GLOBAL median cap
            ol.append(vgt_original_local(d))                  # original estimator + LOCAL-cut cap (the fix)
            rk.append(dim_from_bank(attn, p, roll)[0])        # our rank-based slope + local cut
        f = lambda a: np.nanmean(a) if np.isfinite(a).any() else np.nan
        print(f"key={m} {f(og):>14.2f}{f(ol):>16.2f}{f(rk):>16.2f}   (true dim {m})")

if __name__ == "__main__":
    main()
