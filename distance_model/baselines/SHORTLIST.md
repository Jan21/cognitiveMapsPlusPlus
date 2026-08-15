# Baseline shortlist (merged from 3 deep-research reports, 2026-08-13)

Sources: `raw/claude.md` (294 sources, repos verified individually), `raw/gemini.md` (repo metadata + file paths), `raw/grok.md` (fast pass). ChatGPT run never completed (query saved, no response). All three reports converge on the same plan.

## The one dependency that covers everything

**`torchqmet`** ([quasimetric-learning/torch-quasimetric](https://github.com/quasimetric-learning/torch-quasimetric)), BSD-3-Clause, ~51 stars, last commit 2024-10-31, zero open issues. Ships IQE, MRN, MRNFixed, PQE, DeepNorm, WideNorm as standalone `nn.Module`s with `forward(x, y) -> distance` on two latents. Not on PyPI: `python setup.py install` or vendor the `torchqmet/` folder. File paths (per Gemini): `torchqmet/iqe.py`, `mrn.py`, `deep_norm.py`, `wide_norm.py`. Maintainer guidance: "If you are not sure which one to use, we recommend first trying IQEs."

Note: a stale `torchqmet-0.1.0` egg sits in the local conda env but does not import; reinstall before use.

## Mandatory set (all three reports agree)

| # | Baseline | Source | Effort | Role |
|---|---|---|---|---|
| 1 | **IQE** (`torchqmet.IQE`, dim_per_component=16) | torchqmet | 2-4 h | the reviewer-expected quasimetric SOTA |
| 2 | **MRNFixed** (`torchqmet.MRNFixed`, sym_p=1) | torchqmet | 2-4 h | residual metric family; use Fixed, original MRN (sym_p=2) violates the triangle inequality |
| 3 | **Scalar-head regressor** on concat(s,g) (MLP + transformer variant) | native | 1-4 h | no-inductive-bias control; expected to fail extrapolation, which makes our claim legible |

Near-free 4th: **symmetric embedding** `d = ||f(s) - f(g)||_1` (native, 1 h). Isolates the value of the readout vs plain metric embedding.

## Optional

- **PQE** (`torchqmet.PQE`; official standalone [ssnl/poisson_quasimetric_embedding](https://github.com/ssnl/poisson_quasimetric_embedding)): frame as ablation showing IQE's optimization advantage; PQE has documented diminishing-gradient issues. PQE-GG variant compiles a C++/CUDA extension on first use.
- **DeepNorm / WideNorm** (`torchqmet.DeepNorm`, `final_activation="relu"` REQUIRED for the quasimetric guarantee; do not use [spitis/deepnorms](https://github.com/spitis/deepnorms), TF1 + self-described untested torch port).

## Do not adopt (unanimous)

- **QRL** ([quasimetric-learning/quasimetric-rl](https://github.com/quasimetric-learning/quasimetric-rl)): its value head literally is torchqmet; the contribution is the RL objective (squared-ReLU local constraint, not our smooth L1), so citing it as a same-supervision baseline would misrepresent the protocol. Cite in related work.
- **Contrastive RL** (google-research monorepo): InfoNCE critic in Acme/JAX; porting changes the method's identity.
- **TMD** (vivekmyers/tmd-release), **BVN** (Improbable-AI/bvn): heads reduce to torchqmet / symmetric embedding respectively.
- **CLRS-30** ([google-deepmind/clrs](https://github.com/google-deepmind/clrs)): JAX/Haiku, hint-based supervision over graph traces, different task. Optional separate extrapolation stressor, not a same-supervision baseline.

## Setting-specific caveat the reports missed

All three reports score quasimetrics highly partly for handling ASYMMETRY. Our coupling gridworld has reversible moves, so BFS distance is symmetric. Consequences: (a) the symmetric-embedding baseline is not handicapped here and becomes a genuinely strong contender; (b) IQE's asymmetry advantage does not bite; its remaining edge is the triangle-inequality/compositional bias. Worth one sentence in the paper so reviewers do not read the baseline table as an asymmetry test.

## License notes

torchqmet BSD-3; quasimetric-rl MIT; clrs Apache-2.0; **Cranial-XIX/metric-residual-network and spitis/deepnorms have no LICENSE file** (fine to learn from, clarify before redistributing code).

## Predicted result pattern (consensus)

Scalar head: best in-distribution, collapses on config generalization and length extrapolation. Quasimetric heads: competitive in-distribution, extrapolate via sub-additivity. Symmetric norm: strong extrapolation, symmetric by construction (no penalty in our world). Our method's comparison story: match quasimetrics in-distribution, and the interesting axes are config generalization and (once the global-base fix lands) length extrapolation.
