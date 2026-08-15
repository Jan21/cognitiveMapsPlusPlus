Click to collapseCtrl+B

Drag to resize



HomeCode

New`Ctrl+⇧+O`

Chats and tasksProjectsArtifactsScheduledCustomize

Projects

Pin projects to keep them here

Pinned

Drag to pin

Recents



[Goal-conditioned distance baseline survey and replication plan](/chat/42ec9f5b-343f-4a73-ab25-6fc2713c4eba)



[Algebraic structures in neural network representations](/chat/247c14e1-6dc6-497f-b415-f164fb50cbe2)



[Accumulated latent increments for goal-conditioned distance learning](/chat/80e69775-c5b4-49c1-b8d3-14f5c25ed3ba)



[Learning path for Anderson acceleration in optimization](/chat/db86e394-9052-4a89-a3ba-c9547fef26d9)



[Optimization-based alternatives to neural networks in reinforcement learning](/chat/842f7c4d-cba0-4a5b-8542-460d8010169d)



[Best €300k on-premise machine for frontier model inference](/chat/22f8d711-e742-448c-81b2-f2e1bf47c31f)



[Prior-art analysis of neural flow models and RL architectures](/chat/114bd994-23d2-40bc-86a4-456ca664af2d)



[Prior-art research inquiry](/chat/a6e12a24-9720-411f-bc67-994c404e38ad)



[Mary Magdalene's historical existence and biblical accounts](/chat/e14eac34-e4c5-4576-be3a-22d2ec2c7482)



[Grid cell emergence in trained recurrent neural networks](/chat/8393c47a-2a3b-4185-90b2-2de0731b7928)



[Kolmogorov complexity in neural networks research catalog](/chat/06176502-cb42-4b9b-9355-ee47114b4517)



[Semigroup theory in games and puzzles: literature review](/chat/58b58a71-4800-4f2b-b86e-fcf76bc93b47)



[Green relations in semigroup theory](/chat/59b05c86-cd50-4434-9f53-46f321634ea3)



[Model complexity evolution during neural network training](/chat/7cfadb71-2217-4926-a614-e5636ca1c031)



[Comparing paper similarity](/chat/a04c4e2b-ce62-46f5-827f-68df557c4cbb)



[Lean 4 theorem-proving language models (2025-2026)](/chat/8c10afea-89c2-4008-bec2-fb4e31ea3133)



[Czech investing creators directory research](/chat/69bb4966-a5a3-4ca8-9f24-08d3cefa1762)



[Ověření možnosti vyplnění formuláře](/chat/dcc38430-c6f9-4fdd-af92-1388fb58c70e)



[Form filling assistance](/chat/0c954406-31ff-4ca6-a968-18df8685ea1c)



[Hypernetworks for symbolic reasoning: paper analysis and synthesis](/chat/088b2157-721f-4ab3-811a-7071417c0c3f)



View all

Pinned

Drag to pin

Recents



[Investigate multirung ladder scaling behavior](/code/session_0132zHmmrmVG2D1rg7hpXcio)



[Two-tier classification with deterministic prefilter and LLM tie-break](/code/session_013FCzHC5XgyndLF45qZSVAm)



[Monitor llama-server benchmark throughput](/code/session_01BAcVfDapNRMLb2b4iD92sq)



[Analyze agentic loop implementations across IMO repos](/code/session_01XvY2wT4XZWXku6iYXNazKs)



[Document repository with web documentation](/code/session_01SJRuJVx5GDJ15MPfbgWQx7)



[Dispatch background conversation](/code/session_01XnXuBBRaeb9rbZMSemGFp6)



[Add LaTeX source to theorem proving prompt](/code/session_01HbgZTT4N5nhskscUk5BLh6)



Design

JHJan·Max



Goal-conditioned distance baseline survey and replication plan

Share

Claude finished the response

Use the up and down arrow keys to move between messages.

## You said: Baseline hunt: find up to 5 goal-conditioned distance/value baselines WITH public, replicable code repositories, suitable for head-to-head comparison against o…

Baseline hunt: find up to 5 goal-conditioned distance/value baselines WITH public, replicable code repositories, suitable for head-to-head comparison against our method.

OUR SETTING (what a baseline must plug into): We train a network to predict the shortest-path (BFS geodesic) distance between two states of a discrete gridworld with configurable movement constraints. Training is plain supervised regression: input is a pair (state s, goal state g), each state a short symbolic vector (agent positions plus constraint settings) or a rendered image; the target is the exact integer distance; the loss is smooth L1. There is NO policy, NO reward loop, NO environment interaction at training time. Our model reads the distance out as the accumulated per-step latent displacement of a recurrent transformer. For the paper we need baselines that parameterize the distance function differently, trained under the IDENTICAL supervision, and evaluated on: (a) in-distribution accuracy, (b) generalization to held-out constraint configurations, (c) length extrapolation (train on distances <= 4, test up to 12).

TASK: Find up to 5 baseline methods, prioritized by (1) relevance to goal-conditioned distance parameterization and (2) ease of replication. A baseline qualifies ONLY if a public code repository exists that we can realistically adapt within a few days.

HARD REQUIREMENTS per baseline: \- A public repository (GitHub or equivalent), preferably the authors' official implementation. Provide the exact URL. \- The distance/value parameterization must be extractable as a standalone architecture (an nn.Module or equivalent) trainable with our supervised loss on (s, g) pairs. Methods locked into a full RL training stack only qualify if the distance head can be cleanly separated; say explicitly whether it can. \- The repo must show signs of life or reproducibility: state stars, last commit date, open/closed issue activity, whether third parties have reproduced or forked it meaningfully, and the license.

CANDIDATE FAMILIES TO SEARCH (from our earlier prior-work survey; verify repos, do not assume they exist): 1\. Interval Quasimetric Embeddings (IQE, Wang and Isola, NeurReps 2022) and the torchqmet / quasimetric-learning libraries around it. Highest priority: the expected reviewer comparison. 2\. Metric Residual Networks (MRN, Liu, Feng, Liu, Stone, AAAI 2023). 3\. Quasimetric RL (QRL, Wang, Torralba, Isola, Zhang, ICML 2023), if the quasimetric value model is separable from the RL loop. 4\. DeepNorm / WideNorm (Pitis et al., ICLR 2020): learned norms respecting the triangle inequality. 5\. Poisson Quasimetric Embeddings (Wang and Isola, ICLR 2022). 6\. Plain baselines worth confirming implementations for completeness: a scalar-head MLP/transformer regressor on concatenated (s, g), and a symmetric embedding-distance model V = -||f(s) - f(g)||. 7\. Anything else you find that fits better, e.g. contrastive/temporal-distance representations (Contrastive RL, Dynamical Distance Learning, temporal-distance JEPA lines) with maintained code, or neural algorithmic reasoning shortest-path models (CLRS baselines repo) IF the distance readout can be trained under our supervision.

FOR EACH SELECTED BASELINE REPORT: \- Citation (authors, title, venue, year) and one sentence on the distance parameterization. \- Repo URL, official or third-party, stars, last commit, license, install notes (pip package? bare research code?). \- Exactly which file/class implements the distance function, if determinable from the repo structure. \- Adaptation plan in 2-3 sentences: what we would keep (architecture), what we would replace (data loading, training loop), and an effort estimate (hours/days). \- Known reproduction reports or issues that would block us. \- A relevance score 1-5 for our three evaluation axes (accuracy, config generalization, length extrapolation), with one sentence of justification.

RANKING AND VERDICT: End with a ranked shortlist (best 5) and a one-paragraph recommendation of the minimal baseline set for the paper: which 3 are mandatory for reviewers (we expect IQE, MRN, scalar head unless your findings say otherwise), which are optional, and any baseline we should NOT bother with because the repo is dead or the adaptation cost is out of proportion.

Do not pad: if fewer than 5 methods have usable repositories, say so and explain what is missing. Verify repository existence and activity directly; do not cite a repo you have not confirmed exists.

Message collapsedShow more

2 hours ago

## Claude responded: This is a well-specified request — I'll go straight into the research and verify each repository directly (stars, last commit, license, where the distance modu…

This is a well-specified request — I'll go straight into the research and verify each repository directly (stars, last commit, license, where the distance module lives).

Lead Researcher

Goal-conditioned distance baselines search

Research complete•

294 sources and counting...

294 sources

•

10m 16s

Your goal-conditioned distance/value baselines report is ready. It identifies torchqmet (BSD-3-Clause) as a single verified package covering IQE, MRN/MRNFixed, PQE, DeepNorm, and WideNorm heads, plus recommendations for two hand-written plain baselines (scalar-head regressor and symmetric embedding). It covers repo verification (stars, licenses, commit recency), adaptation effort estimates for smooth-L1 supervision, per-method relevance ratings across in-distribution accuracy, held-out generalization, and length extrapolation, and explicit guidance on which live repos (QRL, Contrastive RL, TMD, BVN, CLRS-30) to skip and why.

Goal-Conditioned Distance Baselines: Verified Repos for Supervised Geodesic-Regression Comparison

Document 

# Goal-Conditioned Distance/Value Baselines with Verified Repos for a Supervised Geodesic-Regression Comparison ## TL;DR \- **One package solves most of your problem: `torchqmet` (quasimetric-learning/torch-quasimetric, BSD-3-Clause, 50 stars, last c

2 hours ago

Want to be notified when Claude responds?Notify

  




Opus 5 High







[Claude is AI and can make mistakes. Please double-check responses.](https://support.anthropic.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on)

Copy



Publish

# Goal-Conditioned Distance/Value Baselines with Verified Repos for a Supervised Geodesic-Regression Comparison

## TL;DR

  * **One package solves most of your problem:`torchqmet` (quasimetric-learning/torch-quasimetric, BSD-3-Clause, 50 stars, last commit Oct 31 2024) ships IQE, MRN/MRNFixed, PQE, DeepNorm and WideNorm as standalone `nn.Module`s whose `forward(x, y)` maps two latents to a distance** — pair any of them with a shared encoder and train under your exact smooth-L1 supervision on (s,g) pairs. This gives you 4 of the 6 candidate families from a single verified, permissively-licensed dependency. The maintainers themselves advise starting with IQE: _"If you are not sure which one to use, we recommend first trying IQEs."_
  * **The mandatory reviewer set is achievable and cheap: IQE (via torchqmet), MRN (via torchqmet`MRNFixed`, official code at Cranial-XIX/metric-residual-network), and a plain scalar-head regressor on concat(s,g)** — plus the symmetric embedding baseline V = −‖f(s)−f(g)‖, which is a two-line special case.
  * **Skip the full RL stacks as baselines.** QRL, Contrastive RL and TMD all have live repos, but their distance heads are either already re-exported by torchqmet (QRL literally vendors `torchqmet`) or buried in an Acme/JAX or OGBench training loop; adapting them to pure supervised regression costs far more than it returns.


## Key Findings

  * Every repository below was opened and verified to exist. Star counts, licenses and commit/activity signals are reported as observed; where a datapoint could not be directly confirmed it is flagged explicitly.
  * The single highest-value asset is `torchqmet`, because it decouples the _latent quasimetric head_ d_z(f(x), f(y)) from any encoder or RL loop exactly the way your comparison requires. It is installed via `python setup.py install` (or by copying the `torchqmet` folder into your project); it is **not** on PyPI.
  * Your two "plain" baselines (scalar head, symmetric embedding) do not need a third-party repo; reference implementations nonetheless exist inside the MRN and BVN codebases.


## Details

### 1\. Interval Quasimetric Embeddings (IQE) — HIGHEST PRIORITY

  * **Citation:** Tongzhou Wang, Phillip Isola. "Improved Representation of Asymmetrical Distances with Interval Quasimetric Embeddings." NeurReps Workshop @ NeurIPS 2022 (Proceedings Track). arXiv:2211.15120. [GitHub](https://github.com/quasimetric-learning/torch-quasimetric)
  * **Parameterization:** reshapes each latent into k×l interval matrices and sums the Lebesgue measure (union length) of per-row intervals [u_ij, max(u_ij, v_ij)], giving an asymmetric, triangle-inequality-respecting distance. [arxiv](https://arxiv.org/pdf/2211.15120)
  * **Repo (official):** <https://github.com/quasimetric-learning/torch-quasimetric> — BSD-3-Clause; 50 stars; 3 forks; 8 commits; last commit Oct 31 2024; no open issues visible. Installable via `python setup.py install` then `import torchqmet`; the README explicitly offers the alternative of just adding "the `torchqmet` folder to your project." It is a research package, not on PyPI. [github](https://github.com/quasimetric-learning/torch-quasimetric)
  * **Distance class:** `torchqmet.IQE(input_size, dim_per_component=16, ...)`, re-exported from `torchqmet/__init__.py`; `forward(x, y)` takes two `(..., input_size)` latents and returns `(...)`, supporting pdist/cdist via broadcasting. Default is IQE-maxmean; `reduction="sum"` gives IQE-sum. (The exact internal filename for the IQE class inside the `torchqmet/` folder could not be confirmed — GitHub's directory tree is robots-blocked to automated fetching — but the public API and forward signature are verified.) The torchqmet README endorses this head as the default choice: _"the IQE paper analyzes many desired properties for d_z and propose IQE, which greatly improves over previous methods with a simple d_z form. If you are not sure which one to use, we recommend first trying IQEs."_ [github](https://github.com/quasimetric-learning/torch-quasimetric)[GitHub](https://github.com/quasimetric-learning/torch-quasimetric)
  * **Adaptation plan:** Keep the `IQE` module; add your own encoder f (symbolic-vector MLP or image CNN) so distance = IQE(f(s), f(g)); replace nothing in torchqmet's internals; write your own dataloader over (s,g,dist) and a smooth-L1 training loop. Effort: about half a day.
  * **Blocking issues:** None found; the repo has no open issues and the module has no external dependencies beyond PyTorch ≥1.11. [github](https://github.com/quasimetric-learning/torch-quasimetric)
  * **Relevance:** in-distribution accuracy **5** (purpose-built expressive quasimetric that the authors recommend as the default); held-out constraint generalization **4** (strong geometric inductive bias, but generalization to unseen constraint settings is untested for this exact task); length extrapolation **5** (triangle inequality + positive homogeneity are exactly the biases that help distances compose beyond training range).


### 2\. Metric Residual Networks (MRN)

  * **Citation:** Bo Liu, Yihao Feng, Qiang Liu, Peter Stone. "Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning." AAAI 2023 (oral). arXiv:2208.08133. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26058)
  * **Parameterization:** decomposes distance into a symmetric metric term ‖φ(x)−φ(y)‖₂ plus an asymmetric residual max-over-components term. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26058/25830)
  * **Repos (two options):**
    * **Cleanest:** `torchqmet.MRN(input_size, sym_p=2, ...)` and `torchqmet.MRNFixed(input_size, sym_p=1, ...)` in the same BSD-3-Clause package above. Per the `MRNFixed` docstring, it is MRN _"with fix proposed by the IQE paper (Sec. C.2)… Defaults to sym_p=1. This guarantees a quasimetric, unlike the original official MRN (where sym_p=2)."_ [GitHub + 2](https://github.com/quasimetric-learning/torch-quasimetric)
    * **Official standalone:** <https://github.com/Cranial-XIX/metric-residual-network> — 20 stars; 1 fork; 8 commits; 1 open issue; 1 open PR; no LICENSE file present in the repo tree. The distance/critic architectures live in `src/model.py`; the surrounding code is a DDPG+HER RL stack (`src/agent/ddpg.py`, `her.py`, etc.). A maintainer note records the fix that motivated `MRNFixed`: _"update 2022/12/6: Thank @SsnL for pointing out the bug that the metric part should be l2-norm instead of square loss. (See this paper [arXiv:2211.15120])."_ [GitHub](https://github.com/Cranial-XIX/metric-residual-network)
  * **Separability:** In torchqmet the head is fully standalone. In the official repo the MRN critic is a class inside `src/model.py` that can be lifted out, but it is embedded in an HER/DDPG stack you would otherwise ignore.
  * **Adaptation plan:** Use `torchqmet.MRNFixed` exactly like IQE; keep the module, add encoder, write smooth-L1 loop. Effort: a few hours (near-identical to IQE once IQE is wired). Using the official repo instead would cost 1–2 days to strip the RL loop.
  * **Blocking issues:** Prefer `MRNFixed`; the original MRN (sym_p=2) is not guaranteed to satisfy the triangle inequality, which could confound your length-extrapolation axis. [GitHub](https://github.com/quasimetric-learning/torch-quasimetric)
  * **Relevance:** in-distribution **5** ; held-out generalization **4** ; length extrapolation **4** (metric part composes well; the asymmetric residual is less constrained than IQE's construction).


### 3\. Scalar-head monolithic regressor on concat(s,g) — MANDATORY PLAIN BASELINE

  * **Citation:** No single paper; this is the standard monolithic UVFA-style baseline (Schaul et al. 2015 for universal value functions), used as the reference "monolithic network" in both the MRN and BVN papers.
  * **Parameterization:** an MLP or transformer that ingests the concatenated (or cross-attended) pair and emits a single scalar distance directly — no geometric constraint.
  * **Repo:** Reference implementation is the `Monolithic Network` critic in `src/model.py` of <https://github.com/Cranial-XIX/metric-residual-network> (details as above). This baseline is trivial enough that writing it natively is usually preferable to importing. [GitHub](https://github.com/Cranial-XIX/metric-residual-network)
  * **Adaptation plan:** Keep nothing external; implement a 3-layer MLP head (and, to mirror your own model, a small transformer variant) over the (s,g) encoding; reuse the same dataloader and smooth-L1 loop. Effort: 2–4 hours.
  * **Blocking issues:** None.
  * **Relevance:** in-distribution **5** (unconstrained fitters usually win in-distribution); held-out generalization **2** (no inductive bias for compositional structure); length extrapolation **1** (this is the classic baseline that fails to extrapolate — it is the control that makes your extrapolation claim meaningful).


### 4\. Symmetric embedding distance V = −‖f(s)−f(g)‖ — MANDATORY PLAIN BASELINE

  * **Citation:** The standard Siamese/temporal-distance embedding; the exact form d(s,g)=‖ψ(s)−ψ(g)‖₂ is used in temporal-distance representation learning (Park et al.) and is the symmetric special case underlying VIP (Ma et al., ICLR 2023, arXiv:2210.00030).
  * **Parameterization:** shared encoder into latent space; distance is the plain Euclidean norm of the difference — symmetric by construction.
  * **Repo:** No dedicated repo needed; it is a two-line readout on top of your encoder. If you want an off-the-shelf asymmetric generalization to sit beside it, `torchqmet.DeepNorm` / `torchqmet.WideNorm` (same package) implement norm-induced distances that reduce toward this baseline.
  * **Adaptation plan:** Keep your encoder; set distance = torch.norm(f(s)−f(g)); train with smooth L1. Effort: 1–2 hours.
  * **Blocking issues:** By construction it cannot represent asymmetric distances — expected to fail on any constraint config that induces one-way movement, which is itself an informative result.
  * **Relevance:** in-distribution **4** ; held-out generalization **3** ; length extrapolation **3** (a true metric composes better than the scalar head but cannot capture asymmetry from movement constraints).


### 5\. Poisson Quasimetric Embeddings (PQE) — or DeepNorm/WideNorm

  * **Citation:** Tongzhou Wang, Phillip Isola. "On the Learning and Learnability of Quasimetrics." ICLR 2022. arXiv:2206.15478. (DeepNorm/WideNorm: Pitis, Chan, Jamali, Ba. "An Inductive Bias for Distances: Neural Nets that Respect the Triangle Inequality." ICLR 2020. arXiv:2002.05825.) [GitHub](https://github.com/quasimetric-learning/torch-quasimetric)
  * **Parameterization:** PQE models distances as expected counts of a Poisson process over latent intervals (the first method with a universal-approximation guarantee for quasimetrics, trainable by gradient descent). DeepNorm/WideNorm parameterize asymmetric norm-induced metrics. [GitHub](https://github.com/SsnL/poisson_quasimetric_embedding)
  * **Repos:**
    * `torchqmet.PQE` / `PQELH` / `PQEGG` and `torchqmet.DeepNorm` / `torchqmet.WideNorm` in the BSD-3-Clause package above (PQE-GG needs a bundled C++/CUDA `cdf_ops` extension that compiles on first use). [GitHub](https://github.com/quasimetric-learning/torch-quasimetric)
    * PQE official standalone: <https://github.com/SsnL/poisson_quasimetric_embedding> — main class `pqe.PQE`. [GitHub](https://github.com/SsnL/poisson_quasimetric_embedding)
    * DeepNorm/WideNorm official: <https://github.com/spitis/deepnorms> — 11 stars; 4 forks; 1 commit; no LICENSE file; primarily TensorFlow v1 with a self-described "untested" PyTorch reimplementation in `metrics_pytorch.py`. [GitHub](https://github.com/spitis/deepnorms)
  * **Adaptation plan:** Use `torchqmet.PQE` (or DeepNorm) exactly as IQE/MRN. Prefer torchqmet over the standalone `spitis/deepnorms` repo, whose PyTorch port is unverified by its own authors. Effort: a few hours via torchqmet; 1–2+ days if you insist on the original `deepnorms` code.
  * **Blocking issues:** PQE has documented optimization difficulty (diminishing gradients; needs the discounted/reparametrized settings) — this is precisely the weakness IQE was designed to fix, so PQE is best framed as an ablation showing IQE's advantage rather than a strong contender. [Tongzhouwang](https://www.tongzhouwang.info/interval_quasimetric_embedding/)
  * **Relevance:** in-distribution **4** ; held-out generalization **4** ; length extrapolation **4** (quasimetric guarantees help, but optimization fragility can depress all three in practice).


### Methods with live repos that you should NOT adopt as baselines

  * **Quasimetric RL (QRL)** — Wang, Torralba, Isola, Zhang, ICML 2023. Repo <https://github.com/quasimetric-learning/quasimetric-rl> (MIT; 58 stars; 9 forks; 40 commits; last updated May 19 2025). The quasimetric value model _is_ separable — in fact the repo root file tree contains a vendored `torchqmet` entry and its critic is a torchqmet head — but QRL's contribution is the constrained RL objective, not a new distance parameterization. Its local constraint is a squared-ReLU deviation, **not** smooth-L1 (`local_constraint.py`: `sq_deviation = (dist - self.step_cost).relu().square().mean(); violation = (sq_deviation - self.epsilon ** 2); loss = violation * lagrange_mult`), so citing QRL's loss would misrepresent your identical-supervision protocol. Use torchqmet directly instead. [GitHub](https://github.com/quasimetric-learning/quasimetric-rl)[GitHub](https://github.com/quasimetric-learning)
  * **Contrastive RL** — Eysenbach, Zhang, Salakhutdinov, Levine, NeurIPS 2022. Official code at <https://github.com/google-research/google-research/tree/master/contrastive_rl> (Apache 2.0 monorepo; per-subdirectory star count not reported by GitHub). The critic parameterizes value as an inner product φ(s,a)ᵀψ(g) trained with an InfoNCE/NCE classification loss (built on Acme's JAX SAC agent), not as a distance regressed against integer targets. Converting it to your smooth-L1 (s,g) regression would change the method's identity and cost multiple days; not recommended. [arxiv](https://arxiv.org/pdf/2206.07568)
  * **Temporal Metric Distillation (TMD)** — Myers, Zheng, Eysenbach, Levine, 2025. Repo <https://github.com/vivekmyers/tmd-release>, a fork of OGBench. It uses a quasimetric head (same family as torchqmet) but inside a Monte-Carlo/TD offline-GCRL loop; the distance head is again just a quasimetric you can get from torchqmet, so the fork adds cost without new parameterization.
  * **Bilinear Value Networks (BVN)** — Hong, Yang, Agrawal, ICLR 2022. Repo <https://github.com/Improbable-AI/bvn> (official). Its factorization Q=φ(s,g)ᵀf(s,a) is action-conditioned and lives in a DDPG+HER stack; the pure-distance reduction is essentially the symmetric-embedding baseline you already have, so it is optional at best. [OpenReview](https://openreview.net/pdf?id=LedObtLmCjS)[GitHub](https://github.com/Improbable-AI/bvn)
  * **CLRS-30 neural algorithmic reasoning** — Veličković et al., ICML 2022. Repo <https://github.com/google-deepmind/clrs> (Apache 2.0; 539 stars; 116 forks; actively maintained into 2026; JAX/Haiku, pip-installable as `dm-clrs`). It contains reference BFS/Bellman-Ford shortest-path learners and is the gold standard for _length/size extrapolation_ , but its supervision is graph-trace/hint prediction over message-passing GNNs, not (s,g)→integer regression. It does not plug into your protocol without redefining the task, so treat it as related work / an optional extrapolation stress-test rather than a same-supervision baseline. [GitHub + 2](https://github.com/google-deepmind/clrs)


## Recommendations

  1. **Adopt`torchqmet` now** as a single dependency and instantiate IQE, MRN(Fixed) and PQE/DeepNorm heads on top of one shared encoder. This is the lowest-risk, highest-coverage move and directly delivers the reviewer-expected IQE and MRN.
  2. **Implement the two plain baselines natively** (scalar-head MLP + transformer on concat(s,g); symmetric V=−‖f(s)−f(g)‖). These are the controls that make your generalization and extrapolation claims legible.
  3. **Mandatory set for reviewers:** IQE, MRN, scalar-head regressor — exactly as you anticipated. **Add the symmetric embedding baseline** as a fourth near-free control; it isolates the value of asymmetry.
  4. **Optional:** PQE or DeepNorm/WideNorm as an ablation demonstrating IQE's optimization advantage within the quasimetric family.
  5. **Do not spend budget on** QRL, Contrastive RL, TMD or BVN as baselines — their distinctive parts are either training objectives (not distance parameterizations) or already reachable through torchqmet. Keep CLRS in "related work" and, if you want an extra extrapolation stressor, run its BFS learner as a separate size-generalization probe rather than a same-supervision baseline.


  * **Thresholds that would change this:** if a reviewer demands an inner-product/contrastive critic specifically, budget 2–3 days to port the Contrastive RL critic loss to supervised regression; if the original-authors' MRN numbers are contested, fall back from `torchqmet.MRNFixed` to Cranial-XIX/metric-residual-network's `src/model.py` and reconcile the L2-vs-squared fix.


## Caveats

  * **torchqmet internal file layout is unverified.** The public API (`torchqmet.IQE`, `.MRN`, `.MRNFixed`, `.PQE`, `.DeepNorm`, `.WideNorm`) and the two-tensor `forward(x, y)` signature are confirmed from the README, but the exact `.py` filenames and whether a `QuasimetricBase` exists in a `base.py` could not be read (GitHub tree pages are robots-blocked to automated fetching). Confirm by `git clone` before writing precise file-path references in the paper.
  * **Licenses:** torch-quasimetric = BSD-3-Clause; quasimetric-rl = MIT; clrs = Apache 2.0. **metric-residual-network and deepnorms show no LICENSE file** — usable for research but clarify reuse terms before redistribution.
  * **Contrastive RL star count** is not separately reported because it lives in the google-research monorepo; the monorepo-level count does not reflect the subdirectory.
  * **Commit-recency signals** were read from GitHub organization/repo pages (torch-quasimetric last commit Oct 31 2024; quasimetric-rl updated May 19 2025). Star counts are point-in-time (verified in this session) and will drift.
  * Fewer than five _distinct_ strong same-supervision baselines exist as separate mature repos; the practical answer is that **one package (torchqmet) supplies the whole quasimetric family** , and the remaining baselines are near-trivial to write. This is a feature, not a shortfall — but it means "five repos" overstates the sourcing; realistically it is two production-grade repos (torchqmet, and MRN/BVN for reference) plus two hand-written controls.


