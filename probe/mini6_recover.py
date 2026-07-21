"""Can we RECOVER the true guard graph from the learned pieces? (no dependency prior wired in)

Uses SumHeadLD from mini5 (a detour token per ordered component pair (f,g); gates learn which fire).
The plain contribution readout smears once dependencies stack (transitive A<-C carries real cost,
plus some genuinely-spurious pieces). Two ideas to sharpen it:

  (1) GROUP-SPARSITY (L1): a group-lasso on per-piece contributions, so pieces that can be zeroed
      are. Sweep lambda. Should prune truly-spurious pieces while keeping the necessary ones
      (including the transitive A<-C, which a single-component piece cannot avoid).
  (2) ABLATION IMPORTANCE (causal): zero each detour piece on the trained model and measure the
      held-out RMSE increase. A necessary piece hurts when removed; a spurious one does not.

Reported on the nested env (true direct guards A<-B, B<-C) plus grid_key as a single-guard sanity.
"""
import numpy as np, torch
from mini5_general import ENVS, SumHeadLD, MDS, train, split, rmse_on

def contrib_matrix(head, idx):
    V, gate, moved = head._parts(torch.tensor(idx[0]), torch.tensor(idx[1]))
    c = (gate * V).detach()
    M = np.full((head.F, head.F), np.nan)
    for t, (f, g) in enumerate(head.pairs):
        mv = moved[:, f]
        if mv.any(): M[f, g] = float(c[:, head.F + t][mv].mean())
    return M

@torch.no_grad()
def ablation_matrix(head, geo, env, idx):
    base = rmse_on(head, geo, env, idx)[0]
    M = np.full((head.F, head.F), np.nan)
    for t, (f, g) in enumerate(head.pairs):
        old = float(head.dmask[t]); head.dmask[t] = 0.0
        M[f, g] = rmse_on(head, geo, env, idx)[0] - base
        head.dmask[t] = old
    return base, M

def show(title, M, env, fmt="{:<7.2f}"):
    F = env.F; true = {f: (env.guards[f][0] if env.guards[f] else None) for f in range(F)}
    print(f"  {title}")
    print("        " + "".join(f"g={g:<6}" for g in range(F)))
    for f in range(F):
        row = "".join((fmt.format(M[f, g]) if not np.isnan(M[f, g]) else f"{'--':<7}") for g in range(F))
        star = "".join((" *" if true[f] == g else "  ") for g in range(F))
        print(f"  f={f}  {row}  true:{star}")

def run(envname, lams, steps=5000, seed=0):
    env = ENVS[envname](); geo, _ = env.geodesic()
    rng = np.random.default_rng(seed); tr, te = split(env, rng)
    print(f"\n===== {envname}  (F={env.F}, states={env.N}) =====")
    for lam in lams:
        torch.manual_seed(seed); np.random.seed(seed)
        h = train(SumHeadLD(env), geo, env, tr, steps, l1=lam)
        rte = rmse_on(h, geo, env, te)[0]
        print(f"\n  -- lambda={lam}  held-out RMSE={rte:.3f} --")
        show("contribution (gate*value)", contrib_matrix(h, te), env)
        if lam == 0.0:
            base, A = ablation_matrix(h, geo, env, te)
            show(f"ablation: held-out RMSE increase when piece removed (base {base:.3f})", A, env)

def loo_retrain(envname, steps=3000, seed=0, rigid=False, tied=False):
    """Causal recovery: retrain from scratch with one detour piece disabled; the RMSE it cannot
    recover is that dependency's true necessity (defeats the 'cost got relocated' confound).
    rigid=True: ungated component-local move-terms. tied=True: detour gate depends only on its own
    component's motion (piece (f,g) means exactly 'moving f costs extra via g')."""
    env = ENVS[envname](); geo, _ = env.geodesic()
    rng = np.random.default_rng(seed); tr, te = split(env, rng)
    torch.manual_seed(seed); np.random.seed(seed)
    full = train(SumHeadLD(env, rigid=rigid, tied=tied), geo, env, tr, steps); rfull = rmse_on(full, geo, env, te)[0]
    print(f"\n===== {envname}: LOO RETRAIN (rigid={rigid}, tied={tied}, full held-out RMSE={rfull:.3f}) =====")
    show("full-model contribution (gate*value)", contrib_matrix(full, te), env)
    M = np.full((env.F, env.F), np.nan)
    for t, (f, g) in enumerate(SumHeadLD(env).pairs):
        torch.manual_seed(seed); np.random.seed(seed)
        h = SumHeadLD(env, rigid=rigid, tied=tied); h.dmask[t] = 0.0  # this dependency forbidden
        h = train(h, geo, env, tr, steps); M[f, g] = rmse_on(h, geo, env, te)[0] - rfull
    show("held-out RMSE increase after retraining WITHOUT the piece (necessity)", M, env, fmt="{:<7.3f}")

if __name__ == "__main__":
    loo_retrain("grid_key", steps=3000)
    loo_retrain("nested", steps=3000)
