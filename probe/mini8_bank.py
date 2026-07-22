"""VGT dimension from a GLOBAL reusable bank instead of per-probe jitter.

mini7 built a fresh local pool around each probe. Here we instead collect ONE big bank of states
once, by running rollouts (legal random walks) from many random starts, optionally adding jitter,
and REUSE it for every probe: score the bank with the learned distance, cut at the free/frozen gap,
read the correlation dimension. The question is DENSITY -- does a probe get enough near-neighbours
(pool states sharing its frozen coordinates) from a global bank?

Why rollouts help: a frozen agent cannot move, so within a fixed-key segment a trajectory stays on
ONE stratum and naturally traces out its free directions -- a far better sampler of a low-dim slice
than uniform sampling (which earlier gave ~0 near-neighbours).

Reuses the trained model + env from mini7_local_vgt.
"""
import argparse, numpy as np, torch
from mini7_local_vgt import NAG, G, Kk, AttnDist, Euclid, train, vgt, legal_neighbour, rand_states, free_mask

def collect_bank(n_traj, length, rng, jitter=0, W=2, free_only=False):
    """n_traj parallel legal rollouts of `length` steps -> states; optionally add jittered copies."""
    s = rand_states(n_traj, rng); frames = []
    for _ in range(length):
        frames.append(s.copy()); s = legal_neighbour(s, rng)
    bank = np.concatenate(frames, 0)
    if jitter > 0:
        reps = [bank]
        for _ in range(jitter):
            b = bank.copy(); fr = free_mask(b[:, NAG]) if free_only else np.ones((b.shape[0], NAG), bool)
            for i in range(NAG):
                dj = rng.integers(-W, W + 1, b.shape[0]) * fr[:, i]
                b[:, i] = (b[:, i] + dj) % G
            reps.append(b)
        bank = np.concatenate(reps, 0)
    return bank

@torch.no_grad()
def dist_all(head, probe, bank, bs=20000):
    pt = torch.tensor(probe[None]); out = []
    for k in range(0, len(bank), bs):
        b = torch.tensor(bank[k:k + bs]); out.append(head(b, pt.repeat(len(b), 1)).numpy())
    return np.concatenate(out)

def dim_from_bank(head, probe, bank, kmax=4000, gapthr=0.8):
    d = np.sort(dist_all(head, probe, bank)); d = d[d > 1e-6]
    if d.size < 40: return np.nan, 0
    d = d[:min(kmax, d.size)]                                            # LOCALITY: nearest kmax (needed
    ld = np.log(d); gap = ld[1:] - ld[:-1]; lo, hi = 20, int(0.9 * d.size)  # when NO frozen gap exists, e.g. full-DOF)
    cand = np.where(gap[lo:hi] > gapthr)[0]                              # then tighten at the free/frozen gap if any
    near = lo + int(cand[0]) + 1 if cand.size else d.size
    return vgt(d[:near]), near

def pick_probes(rollout_bank, m, k, rng):
    idx = np.where(rollout_bank[:, NAG] == m)[0]
    return rollout_bank[rng.choice(idx, size=min(k, idx.size), replace=False)] if idx.size else np.empty((0, NAG + 1), int)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    head = train(AttnDist(), args.steps, rng)

    roll = collect_bank(3000, 60, rng)                                   # 180k rollout states (probe source)
    banks = {
        "rollout(180k)":         roll,
        "rollout+jitterAll(x3)": collect_bank(3000, 60, rng, jitter=2, W=2, free_only=False),
        "rollout+jitterFree(x3)":collect_bank(3000, 60, rng, jitter=2, W=3, free_only=True),
    }
    print(f"probes drawn FROM the rollout bank (states we actually visited), 8 per key")
    for bn, bank in banks.items():
        print(f"\n=== bank: {bn}  ({len(bank)} states) ===")
        print(f"{'':<6}{'dim (mean)':>12}{'near-count (mean)':>20}{'valid/8':>9}")
        for m in (1, 2, 3):
            probes = pick_probes(roll, m, 8, rng)
            dims, nears = [], []
            for p in probes:
                dd, nn = dim_from_bank(head, p, bank); dims.append(dd); nears.append(nn)
            dims = np.array(dims); valid = np.isfinite(dims).sum()
            md = np.nanmean(dims) if valid else np.nan
            print(f"key={m} {md:>12.2f}{np.mean(nears):>20.0f}{valid:>7}/8")

if __name__ == "__main__":
    main()
