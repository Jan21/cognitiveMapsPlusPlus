---
url: https://chatgpt.com/c/6a7cdd08-2554-83eb-8a28-ba7aebecdbaa
source: chatgpt
kind: deep_research
saved: 2026-08-13T05:49:14.310556
---

# Goal-conditioned Baseline Search

_Kind: deep_research_

## User

Deep research
 Baseline hunt: find up to 5 goal-conditioned distance/value baselines WITH public, replicable code repositories, suitable for head-to-head comparison against our method.

OUR SETTING (what a baseline must plug into):
We train a network to predict the shortest-path (BFS geodesic) distance between two states of a discrete gridworld with configurable movement constraints. Training is plain supervised regression: input is a pair (state s, goal state g), each state a short symbolic vector (agent positions plus constraint settings) or a rendered image; the target is the exact integer distance; the loss is smooth L1. There is NO policy, NO reward loop, NO environment interaction at training time. Our model reads the distance out as the accumulated per-step latent displacement of a recurrent transformer. For the paper we need baselines that parameterize the distance function differently, trained under the IDENTICAL supervision, and evaluated on: (a) in-distribution accuracy, (b) generalization to held-out constraint configurations, (c) length extrapolation (train on distances <= 4, test up to 12).

TASK:
Find up to 5 baseline methods, prioritized by (1) relevance to goal-conditioned distance parameterization and (2) ease of replication. A baseline qualifies ONLY if a public code repository exists that we can realistically adapt within a few days.

HARD REQUIREMENTS per baseline:
- A public repository (GitHub or equivalent), preferably the authors' official implementation. Provide the exact URL.
- The distance/value parameterization must be extractable as a standalone architecture (an nn.Module or equivalent) trainable with our supervised loss on (s, g) pairs. Methods locked into a full RL training stack only qualify if the distance head can be cleanly separated; say explicitly whether it can.
- The repo must show signs of life or reproducibility: state stars, last commit date, open/closed issue activity, whether third parties have reproduced or forked it meaningfully, and the license.

CANDIDATE FAMILIES TO SEARCH (from our earlier prior-work survey; verify repos, do not assume they exist):
1. Interval Quasimetric Embeddings (IQE, Wang and Isola, NeurReps 2022) and the torchqmet / quasimetric-learning libraries around it. Highest priority: the expected reviewer comparison.
2. Metric Residual Networks (MRN, Liu, Feng, Liu, Stone, AAAI 2023).
3. Quasimetric RL (QRL, Wang, Torralba, Isola, Zhang, ICML 2023), if the quasimetric value model is separable from the RL loop.
4. DeepNorm / WideNorm (Pitis et al., ICLR 2020): learned norms respecting the triangle inequality.
5. Poisson Quasimetric Embeddings (Wang and Isola, ICLR 2022).
6. Plain baselines worth confirming implementations for completeness: a scalar-head MLP/transformer regressor on concatenated (s, g), and a symmetric embedding-distance model V = -||f(s) - f(g)||.
7. Anything else you find that fits better, e.g. contrastive/temporal-distance representations (Contrastive RL, Dynamical Distance Learning, temporal-distance JEPA lines) with maintained code, or neural algorithmic reasoning shortest-path models (CLRS baselines repo) IF the distance readout can be trained under our supervision.

FOR EACH SELECTED BASELINE REPORT:
- Citation (authors, title, venue, year) and one sentence on the distance parameterization.
- Repo URL, official or third-party, stars, last commit, license, install notes (pip package? bare research code?).
- Exactly which file/class implements the distance function, if determinable from the repo structure.
- Adaptation plan in 2-3 sentences: what we would keep (architecture), what we would replace (data loading, training loop), and an effort estimate (hours/days).
- Known reproduction reports or issues that would block us.
- A relevance score 1-5 for our three evaluation axes (accuracy, config generalization, length extrapolation), with one sentence of justification.

RANKING AND VERDICT:
End with a ranked shortlist (best 5) and a one-paragraph recommendation of the minimal baseline set for the paper: which 3 are mandatory for reviewers (we expect IQE, MRN, scalar head unless your findings say otherwise), which are optional, and any baseline we should NOT bother with because the repo is dead or the adaptation cost is out of proportion.

Do not pad: if fewer than 5 methods have usable repositories, say so and explain what is missing. Verify repository existence and activity directly; do not cite a repo you have not confirmed exists.
