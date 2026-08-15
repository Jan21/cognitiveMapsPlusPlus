---
url: https://grok.com/c/d78f71e6-9da0-404c-9cf1-003cc58a50d7?rid=37e53b75-b87b-43a8-aada-d019d8a33f43
source: grok
kind: normal
saved: 2026-08-11T20:02:24.724026
---

# Accumulated Latent Increments in Goal-Conditioned RL

_Kind: normal_

## User

Prior-work search: is a goal-conditioned value/distance function computed by ACCUMULATING PER-STEP LATENT INCREMENTS of an iterative network a novel idea, or has it been done?
 
CONTEXT (our method, described precisely so you can judge closeness):
We work in goal-conditioned reinforcement learning where the value function is a distance: conditioned on a goal state g, the value of state s is the estimated number of steps (BFS geodesic distance) from s to g. Our architecture computes this distance as follows. A shared encoder maps the current state and the goal state to sets of tokens. A single weight-shared transformer block is then applied for T recurrent steps; the goal and start tokens are re-injected into the sequence at every step (a "recall" mechanism). Crucially, the distance is NOT read out by a scalar regression head and NOT computed as a norm between two embeddings. Instead it is the accumulated path length of the latent trajectory itself: at each recurrent step we add the norms of the token displacements, cost += sum_i ||z_i(t+1) - z_i(t)||, and the final prediction is a learned positive scalar times this accumulated cost. In other words, the value is the integral of local latent "speed" along a learned iterative flow from the start representation toward the goal representation. Distinctive properties: (a) the number of steps T is fixed, yet the model extrapolates to distances far beyond the training range because the per-step increments carry the magnitude (no compute budget tied to distance); (b) training supervises only ground-truth shortest-path distances, in some settings only local/short-range pairs, and we evaluate extrapolation to longer distances and unseen environments; (c) no policy is learned, the network is a pure distance oracle.
 
MAIN QUESTION:
Has anyone previously proposed computing a goal-conditioned value or distance function as the sum/integral of per-iteration displacement norms (or other nonnegative local increments) of a recurrent or iterative network's hidden state? Find the closest prior work, whether an exact match, a partial anticipation, or a clearly related mechanism in a different domain.
 
SEARCH DIRECTIONS (cover each; report the closest matches with full citations):

Quasimetric and metric approaches to goal-conditioned RL: Quasimetric Reinforcement Learning (Tongzhou Wang, Phillip Isola), Interval Quasimetric Embeddings, Poisson Quasimetric Embeddings, Metric Residual Networks (Bo Liu et al.), DeepNorm/WideNorm (Pitis et al.), and any other asymmetric distance parameterizations for goal reaching. These read distance from an embedding pair, not from accumulated increments; check whether any variant accumulates iterative refinements.
Distance learning in goal-conditioned RL more broadly: Universal Value Function Approximators, Hindsight Experience Replay descendants that regress distances, Dynamical Distance Learning (Hartikainen et al.), Contrastive RL as implicit distance (Eysenbach et al.), Search on the Replay Buffer, successor representations and Laplacian/spectral state representations used as distances.
Iterative networks whose OUTPUT is an accumulation over compute steps: Adaptive Computation Time (Graves), PonderNet, Universal Transformers, deep equilibrium models applied to planning, Value Iteration Networks, Gated Path Planning Networks, neural algorithmic reasoning that executes Bellman-Ford or Dijkstra (Velickovic et al.), and recurrent "thinking" networks where iteration count tracks problem difficulty (Schwarzschild et al., Bansal et al.). Distinguish carefully: many of these iterate to refine an answer read out at the end; we instead SUM a quantity across the iterations. Find any that sum per-step costs into the prediction.
Distance as a path integral in a learned latent space: Riemannian geometry of deep generative models (Arvanitidis et al., "latent space oddity"), geodesic computation in VAE/GAN latent spaces, metric learning where distance is the integral of a local metric along a curve, neural ODEs where trajectory length or kinetic energy is penalized or read out, optimal transport and flow matching formulations where the transport cost itself is the quantity of interest.
Eikonal and PDE-style neural distance fields: neural eikonal solvers, signed distance functions trained with eikonal losses, physics-informed networks computing geodesic distances on manifolds or graphs, and any architecture whose recurrent dynamics implement front propagation or value iteration with distance accumulated along the propagation.
Cognitive science and neuroscience angles: cognitive map models where distance estimates come from mental simulation, models where estimated time or effort equals accumulated simulation steps, hippocampal replay or recall-based planning models, successor-representation accounts of distance judgments.
 
FOR EACH CANDIDATE WORK, report: (a) what quantity it computes; (b) the mechanical form of the distance readout (embedding norm, scalar head, accumulated increments, iteration count, PDE solve); (c) whether increments are accumulated across compute steps into the output; (d) whether it is goal-conditioned; (e) whether it trains on local/short-range supervision and tests long-range extrapolation; (f) a closeness score 1-5 relative to our design, with one sentence of justification.
 
VERDICT SECTION (required): list the 3 closest works overall; state exactly which components of our design they anticipate (iterative flow, increment-sum readout, goal re-injection, budget-free extrapolation, distance-only training) and which components remain unaccounted for in the literature; conclude with a clear novelty assessment and a list of additional search terms or communities we may have missed.
 
Prefer peer-reviewed ML venues (NeurIPS, ICML, ICLR, RLC, CoRL, AAAI), arXiv preprints, and relevant robotics and computational neuroscience literature. Cite concretely (authors, title, venue, year); do not pad with tangential goal-conditioned RL surveys.

## Grok

Dokončeno za 18s

No exact match was found. Computing a goal-conditioned distance/value as the sum/integral of per-iteration latent displacement norms (or other nonnegative local increments) of a recurrent/iterative network’s hidden states—with fixed compute budget 
𝑇
T yet magnitude-carrying increments enabling extrapolation, pure distance-oracle training on (often local) shortest-path supervision, and goal re-injection—is not present in the literature surveyed. Closest works anticipate isolated components (embedding norms, iterative refinement, path integrals in latent space, eikonal/PDE distances, algorithmic iteration) but not the combination of latent-trajectory increment accumulation as the distance readout itself.

1. Quasimetric / metric approaches to goal-conditioned RL

These parameterize asymmetric distances via embeddings or residuals and read distance from a pair (norm, max of positive differences, etc.). None accumulate iterative latent increments.

Quasimetric RL / Poisson Quasimetric Embeddings (PQE) / Interval Quasimetric Embeddings (IQE) (Tongzhou Wang, Phillip Isola et al.):
(a) Optimal goal-reaching value as quasimetric 
𝑑
(
𝑠
,
𝑔
)
d(s,g).
(b) Embedding-pair readout (Poisson or interval constructions).
(c) No accumulation of increments across compute steps.
(d) Goal-conditioned.
(e) Local costs → global quasimetric; some extrapolation analysis.
(f) Closeness 2: Strong inductive bias for distances and local-to-global recovery, but static embedding norm/parameterization, not iterative flow + sum of displacements.
Citations: Wang & Isola, “On the Learning and Learnability of Quasimetrics,” ICLR 2022 (arXiv:2206.15478); Wang et al., “Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning,” ICML 2023 (arXiv:2304.01203); Interval Quasimetric Embeddings (NeurIPS 2022 workshop).
Metric Residual Networks (MRN) (Bo Liu, Yihao Feng, Qiang Liu, Peter Stone):
(a) 
𝑄
(
𝑠
,
𝑎
,
𝑔
)
Q(s,a,g) decomposed as negated (metric + residual asymmetric).
(b) Explicit norm + max-positive residual on latent encodings of 
(
𝑠
,
𝑎
)
(s,a) and 
(
𝑠
,
𝑔
)
(s,g).
(c) No iterative accumulation.
(d) Goal-conditioned.
(e) Sample-efficient GCRL; triangle inequality bias.
(f) Closeness 2: Architectural bias toward metric structure and triangle inequality; still a one-shot embedding readout.
Citation: Liu et al., “Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning,” AAAI 2023 (arXiv:2208.08133).
DeepNorm / WideNorm (Pitis et al.):
(a) Norms/metrics respecting triangle inequality.
(b) Architectural norms (Deep/Wide).
(c) No.
(d) Used in multi-goal RL.
(e) Limited-data inductive bias.
(f) Closeness 1–2: Pure metric inductive bias, no recurrence or accumulation.
Citation: Pitis et al., “An Inductive Bias for Distances: Neural Nets that Respect the Triangle Inequality,” ICLR 2020 (arXiv:2002.05825).

No variant accumulates iterative refinements into the distance.

2. Distance learning in goal-conditioned RL more broadly

Standard regression, contrastive, or dynamical distances; readout is a scalar head or implicit occupancy, not latent-increment sum.

Dynamical Distance Learning (DDL) (Hartikainen et al.):
(a) Expected steps to goal (dynamical distance).
(b) Supervised regression on trajectories.
(c) No.
(d) Goal-conditioned (used for shaping/skills).
(e) Unsupervised interaction + limited preference labels; real-robot transfer.
(f) Closeness 2: Learns step-count distances from interaction data and uses them for long-horizon goals, but ordinary regression head.
Citation: Hartikainen et al., “Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery,” arXiv:1907.08225 (2020).
UVFA, HER descendants, Contrastive RL / successor representations / Laplacian:
Scalar heads, embedding similarities, or spectral distances. Iteration count or TD backups may appear, but not sum of latent 
∥
𝑧
𝑡
+
1
−
𝑧
𝑡
∥
∥z
t+1
	​

−z
t
	​

∥. Closeness ≤2.
3. Iterative networks whose output is an accumulation over compute steps

Most refine a final readout; a few accumulate ponder cost or value iterations, but not latent displacement norms as the predicted distance.

Value Iteration Networks (VIN) / Gated Path Planning Networks:
(a) Differentiable value iteration / planning.
(b) Convolutional recurrent updates; final value map or policy.
(c) Values are updated iteratively (Bellman-like), but the output is the refined value, not an explicit sum of per-step latent displacements.
(d) Can be goal-conditioned (navigation).
(e) Generalization to larger mazes.
(f) Closeness 3: Explicit iterative planning whose depth relates to horizon; closest “planning via iteration” family, yet readout is the converged value field, not path-length of the latent trajectory itself.
Citations: Tamar et al., “Value Iteration Networks,” NeurIPS 2016 (arXiv:1602.02867); Lee et al., “Gated Path Planning Networks,” ICML 2018 (arXiv:1806.06408).
Adaptive Computation Time (ACT) / PonderNet / Universal Transformers:
(a) Adaptive compute; final representation or prediction.
(b) Halting scores; weighted sum of intermediate states sometimes used for the representation, or ponder cost as regularization.
(c) Ponder time is accumulated, but it is a scalar compute budget, not the predicted distance; the distance/value is still read from the final (or mixture) state.
(d) Not specifically goal-conditioned distance oracles.
(e) Adaptive depth for difficulty.
(f) Closeness 2: Iteration count tracks difficulty and can be accumulated, but the quantity being predicted is not the integral of latent speed.
Citations: Graves, “Adaptive Computation Time for Recurrent Neural Networks,” arXiv:1603.08983; Dehghani et al., “Universal Transformers,” ICLR 2019.
Neural algorithmic reasoning (Bellman-Ford / Dijkstra imitators, Veličković et al. line):
(a) Shortest-path distances via message-passing iterations that mimic relaxation.
(b) Node embeddings updated by min-plus; final distance labels.
(c) Distances are maintained and updated (can be viewed as progressive accumulation of edge costs), but the network is trained to output the correct distance values, not to make the sum of its own latent displacements equal the distance.
(d) Source-conditioned.
(e) Extrapolation to larger graphs when properly aligned.
(f) Closeness 3: Algorithmic iteration that propagates distances; the mechanical form is still “output the distance field,” not “distance = ∫‖Δz‖ along the network’s latent trajectory.”
Closest papers: Veličković et al. neural execution of BF/Dijkstra; NBFNet (Zhu et al., “Neural Bellman-Ford Networks,” 2021).

Schwarzschild / Bansal-style recurrent “thinking” networks: iteration count correlates with difficulty; readout remains final-state classification/regression. Closeness 2.

4. Distance as a path integral in a learned latent space

This family is conceptually nearest on the “integral of local metric” axis, but the curve is usually optimized or integrated after learning a generative model / metric, not produced as the recurrent trajectory of the distance network itself.

Riemannian geometry of deep generative models / latent geodesics (Arvanitidis, Shao, Kumar, Fletcher, etc.):
(a) Geodesic length under the pull-back metric 
𝐺
(
𝑧
)
=
𝐽
⊤
𝐽
G(z)=J
⊤
J (or learned variants).
(b) Numerical integration of 
∫
∥
𝛾
˙
(
𝑡
)
∥
𝐺
 
𝑑
𝑡
∫∥
γ
˙
	​

(t)∥
G
	​

dt along a curve in latent space (ODE solve, graph approximation, energy minimization).
(c) Yes—path integral—but the path is not the hidden-state trajectory of an iterative distance oracle; it is a post-hoc geodesic on a fixed generator.
(d) Can be used for distances between points, not primarily goal-conditioned RL value functions.
(e) Quality of interpolation/extrapolation studied.
(f) Closeness 3: Distance = integral of local speed along a latent curve is exactly the geometric idea; the architecture and training (iterative network whose own latent flow carries the magnitude, supervised only on distances) differ.
Representative: Shao et al., “The Riemannian Geometry of Deep Generative Models,” arXiv:1711.08014; later works on efficient geodesic approximation and probability-density geodesics in diffusion latent space (CVPR 2025).

Neural ODEs with kinetic-energy / trajectory-length regularization (or readout): length/energy is penalized or used as a transport cost, not typically the goal-conditioned value itself. Closeness 2–3.

5. Eikonal and PDE-style neural distance fields
EikoNet / Neural Eikonal Solver (NES) / PINN eikonal solvers:
(a) Travel-time / geodesic distance fields satisfying 
∥
∇
𝑇
∥
=
1
/
𝑣
∥∇T∥=1/v.
(b) Direct regression of 
𝑇
(
𝑥
,
𝑠
)
T(x,s) or physics-informed residual; sometimes factored.
(c) No per-iteration latent-displacement accumulation of a recurrent net; the network solves the PDE.
(d) Source/goal-conditioned possible (two-point formulations).
(e) Continuous domains, generalization across velocity models.
(f) Closeness 2–3: Distance fields with local-to-global consistency (eikonal), but the computational mechanism is a PDE residual, not an iterative latent flow whose increments sum to distance.
Citations: Smith et al., “EikoNet,” arXiv:2004.00361; Grubas et al., Neural Eikonal Solver series.

Signed-distance functions with eikonal losses are analogous (level-set / surface distance) and likewise do not use the proposed readout.

6. Cognitive science / neuroscience

Cognitive-map, hippocampal-replay, and successor-representation models often equate estimated distance/time with the length or number of simulated steps or with SR expectations. They supply conceptual precedent for “distance = accumulated simulation effort” but are not concrete neural-network architectures that accumulate latent displacement norms of a transformer-style iterative block. Closeness 1–2 (inspirational, not mechanical).

Verdict: 3 closest works overall
Value Iteration Networks / Gated Path Planning Networks (Tamar et al. NeurIPS 2016; Lee et al. ICML 2018) — anticipate iterative flow / planning computation whose depth relates to horizon and can be goal-conditioned. Missing: increment-sum readout (they refine a value map), goal-token re-injection inside a shared transformer block, fixed-
𝑇
T magnitude-carrying extrapolation from local supervision only, pure distance-oracle training without policy.
Latent-space Riemannian geodesics / path-integral distances (Arvanitidis line, Shao et al. 2017+, later diffusion-latent geodesics) — anticipate distance = integral of local latent speed along a curve. Missing: the curve being the recurrent hidden-state trajectory of the distance network itself; the specific architecture (shared transformer + recall); training regime (local shortest-path supervision → long-range extrapolation with fixed compute); goal-conditioned RL setting.
Neural algorithmic reasoning for shortest paths (Bellman-Ford-style message passing) (Veličković et al. and NBFNet) — anticipate iterative accumulation of path costs that extrapolates when the algorithm is correctly implemented. Missing: the costs being latent displacement norms of the network’s own tokens rather than explicit edge weights; transformer-style latent flow with goal re-injection; the precise “distance = learned scalar × ∑‖Δz‖” readout.

Components anticipated somewhere: iterative latent flow / planning (VIN, algorithmic reasoners, Universal Transformers); path-integral / integral-of-local-metric idea (Riemannian latent geometry); local-to-global recovery and quasimetric bias (QRL/MRN/DeepNorm); adaptive or fixed-depth iteration tracking difficulty (ACT, thinking networks); pure distance supervision and some extrapolation (DDL, quasimetrics, NAR).

Components that remain essentially unaccounted for as a package:

Reading the goal-conditioned distance directly as the accumulated path length of the network’s own latent trajectory (∑‖token displacements‖ across recurrent steps of a weight-shared transformer).
Fixed compute budget 
𝑇
T whose per-step magnitudes carry the distance, enabling extrapolation far beyond training range without increasing compute.
Goal/start token re-injection (“recall”) at every step inside that iterative block.
Pure distance-oracle training (no policy) on (often only short-range) geodesic labels, evaluated on long-range / OOD environments.

Novelty assessment: The specific design—goal-conditioned distance as the integral of local latent “speed” along a learned iterative flow produced by a recurrent transformer with recall, trained only on distances and capable of budget-free magnitude extrapolation—is novel. Individual geometric, iterative, and quasimetric ideas exist, but their combination into this mechanical form of value readout does not appear in the peer-reviewed ML, robotics, or computational-neuroscience literature examined.

Additional search terms / communities that could still be fruitful:

“latent trajectory length” / “kinetic energy of hidden states” + value function or planning;
continuous-time value iteration / neural Hamilton-Jacobi-Bellman solvers that integrate along characteristics;
“path-cost accumulation in recurrent planners” or “differentiable Dijkstra with learned embeddings”;
cognitive-map models that explicitly integrate simulated effort (hippocampal replay literature, successor features with continuous time);
flow-matching / optimal-transport planners where the transport cost is the value;
“recurrent residual networks as metric spaces” or “transformer layers as geodesic flows.”

No exact or near-exact prior art was located; the closest partial anticipations leave the core increment-sum latent-flow readout and the fixed-
𝑇
T magnitude extrapolation regime as distinctive.

105 sources
