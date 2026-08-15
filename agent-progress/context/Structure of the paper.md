# NeurReps Extended Abstract --- Proposed Structure

## 1. Introduction: Goal-Conditioned Value Functions as Distance Estimators

### Motivation

-   Standard RL typically optimizes behavior for a particular reward
    function or task.
-   Generalist agents should instead be able to pursue arbitrary desired
    states.
-   Goal-conditioned RL formalizes this by conditioning behavior on
    both:
    -   the current state $s$
    -   a desired goal state $g$
-   The agent should answer: **How far am I from the state I want to
    reach?**

### Goal-conditioned value functions

-   Introduce the goal-conditioned value function $V(s,g)$.
-   Under suitable reward/cost formulations, $V(s,g)$ corresponds
    closely to a distance or cost-to-go between states.
-   This raises a representation question:
    -   **What inductive structure should we impose on a learned
        goal-conditioned value function?**

### Connection to structured distance models

-   Introduce prior work on metric/quasimetric representations of
    goal-conditioned values.
-   Discuss the idea that value functions have geometric structure.
-   Briefly position quasimetric approaches, including relevant work
    associated with Phillip Isola and collaborators.
-   Existing approaches typically obtain distance/value from
    relationships between learned state representations.

### Our perspective

-   Ask whether distance can instead emerge from the **computation
    performed between the two states**.
-   Central hypothesis:
    -   **Goal-conditioned value can be represented as cumulative
        displacement along an iteratively constructed latent
        trajectory.**
-   Rather than directly decoding a scalar value, the network "travels"
    through latent space.
-   The accumulated length of that computation becomes the predicted
    distance.

### Contributions

1.  A value-function parameterization based on **latent trajectory
    integration**.
2.  A depth-recurrent attention architecture implementing this
    computation over structured state representations.
3.  Empirical evaluation of its accuracy and generalization using both
    factored and image-based observations.

## 2. Value Estimation by Latent Trajectory Integration

### Core intuition

-   Interpret value estimation as an iterative process.
-   Start with a representation of the current state.
-   Repeated computation produces a sequence:
    $z_0 \rightarrow z_1 \rightarrow \cdots \rightarrow z_T$.
-   Each update represents movement through the model's latent space.

### Path-integral readout

-   Measure the magnitude of each latent update:
    $\Delta_t = \|z_{t+1}-z_t\|$.
-   Accumulate movement across iterations:
    $\hat d(s,g) \propto \sum_t \Delta_t$.
-   With multiple object/factor slots, integrate displacement across the
    relevant state tokens.
-   The resulting accumulated latent path length is the predicted goal
    distance/value.

### Why this parameterization?

-   The scalar prediction is tied directly to the model's internal
    computation.
-   No conventional scalar output head needs to independently decode
    distance.
-   Distance has an explicit compositional structure: local latent
    movements integrated into global distance.
-   Provides a connection between value functions, latent geometry,
    iterative computation, and mental/path simulation.

## 3. Model

### 3.1 State Representations

#### Factored observations

-   The underlying factors of the environment are directly observed.
-   Each entity/factor receives its own representation or slot.
-   Removes the perception problem.
-   Tests whether the model can learn the underlying distance
    computation itself.

#### Image-based observations

-   The model receives a rendered observation rather than explicit
    factors.
-   Relevant entities/factors must be recovered from pixels.
-   Slot-based representations provide a structured interface between
    perception and the distance model.
-   Slots learn to attend to different components of the scene.
-   Allows us to ask whether latent trajectory integration remains
    effective when the state representation itself must be learned.

### 3.2 Depth-Recurrent Transformer

-   Attention operates over the state/object slots.
-   The same Transformer block is repeatedly applied.
-   Weight sharing creates a **depth-recurrent computation**.
-   Each iteration can be interpreted as another step of latent
    computation.
-   Current state and goal information remain available throughout the
    recurrent process.
-   Number of recurrent iterations can be controlled independently of
    parameter count.

### 3.3 Distance Readout

-   Track how far state representations move during each recurrent
    iteration.
-   Sum latent displacement across recurrent steps and across relevant
    factor/object slots.
-   Apply the learned scaling required to map latent path length onto
    environment distance.
-   This integrated quantity is the predicted goal-conditioned
    value/distance.

### 3.4 Training

-   Ground truth is shortest-path distance between states.
-   Exact distances are obtained from graph search / BFS in the
    environment.
-   Train directly against these distances.
-   No trajectory demonstrations are required.
-   No explicit supervision of the intermediate latent trajectory is
    required.
-   The model must discover an internal computation whose accumulated
    movement predicts distance.

## 4. Experimental Setting

### Environment

-   Introduce the structured grid environment.
-   Multiple movable entities.
-   Constraints on how individual entities can move.
-   Links/couplings between entities.
-   Environment configuration changes the transition structure.
-   Consequently, geometric proximity does not trivially determine
    shortest-path distance.
-   Exact shortest-path distances remain computable with BFS.

### Why this environment?

-   Separates **visual/geometric closeness** from **transition
    distance**.
-   Requires understanding the latent transition structure.
-   Allows systematic manipulation of state distance, constraints,
    environment configurations, and observation modality.

### Evaluation questions

1.  **Can latent path integration accurately estimate goal distance?**
2.  **Does the learned representation capture the environment's
    transition structure?**
3.  **Does the model generalize to unseen environment configurations?**
4.  **Does it extrapolate to distances beyond those observed during
    training?**
5.  **Can the same principle operate with both factored and image-based
    observations?**

## 5. Results

### Distance estimation

-   Report MAE / correlation with true shortest-path distance.
-   Compare factored and image-based observations.
-   Show predicted versus true distance.

### Generalization

-   Held-out constraint configurations.
-   Structurally novel configurations.
-   Training/testing across different distance ranges.
-   Explicitly distinguish interpolation, compositional/configuration
    generalization, and length extrapolation.

### Baselines

Candidate baseline families to investigate: - Standard scalar
value/distance prediction head. - Metric embedding models. - Quasimetric
value-function models. - Relevant architectures such as IQE / MRN. -
Recurrent model with conventional final-state readout. - Ablation
replacing integrated latent displacement with a scalar decoder.

The most important comparison may be:

> **same backbone, different readout**

This isolates whether latent path integration itself contributes
anything beyond the recurrent architecture.

### Ablations

Prioritize only those that directly test the central claim: - integrated
path readout vs scalar head - recurrent vs non-recurrent computation -
weight sharing vs untied depth - different numbers of recurrent
iterations - start/goal recall vs no recall - factored vs learned visual
representations

## 6. Discussion and Conclusion

### Interpretation

-   Goal-conditioned value functions can be viewed as geometric objects.
-   Our proposal places the geometry not only in the representation of
    states, but in the **trajectory of the computation itself**.
-   The network estimates distance by accumulating its own internal
    movement.

### Relation to cognitive maps

-   Connect to cognitive-map interpretations of value and distance.
-   Possible connection to mental simulation / mental scanning.
-   Distance estimation as an internally generated trajectory rather
    than direct scalar decoding.

### What the experiments establish

-   Accuracy of the integrated-path representation.
-   Generalization across transition structures.
-   Applicability to both explicit structured states and learned visual
    representations.

### Limitations

-   Length extrapolation remains a particularly demanding test.
-   Internal latent trajectories should not automatically be interpreted
    as literal environment trajectories.
-   Results currently concern a controlled synthetic environment.
-   Stronger comparisons against structured metric/quasimetric
    approaches are important.

### Outlook

-   Stronger goal-conditioned RL benchmarks.
-   Integration with policy learning/planning.
-   Richer visual environments.
-   Analysis of latent trajectories themselves.
-   Relationship between computational path geometry and environmental
    transition geometry.

# Central Message

> **Instead of decoding goal distance from the final representation, we
> estimate it by integrating the distance traveled by the model's
> representations during iterative computation.**

Everything in the extended abstract should ultimately support this
claim.

------------------------------------------------------------------------

# Deep Research Prompt: Literature and Framing

## Objective

Conduct a deep, citation-driven literature review to help frame a
NeurReps extended abstract about a goal-conditioned value function whose
scalar distance prediction is obtained by **integrating the movement of
internal latent representations over recurrent computation**.

The purpose is not merely to collect related papers. The research should
determine: 1. the strongest intellectual framing of the contribution; 2.
the closest prior work and whether the proposed readout is genuinely
distinct; 3. the terminology and canonical claims used by the relevant
research communities; 4. the baselines and citations that reviewers are
likely to expect; 5. the claims we can safely make, and claims we should
avoid.

## Proposed method to be framed

We consider goal-conditioned prediction of shortest-path / cost-to-go
distance between a current state $s$ and goal $g$.

The model uses structured state representations (factored object/factor
slots, and an image-based version in which slots must recover factors
from pixels). A weight-shared Transformer block is applied recurrently
for multiple depth iterations. Start and goal information condition the
recurrent computation.

Rather than predicting distance using a conventional scalar output head,
or simply measuring a static distance between start and goal embeddings,
the model measures the displacement of its internal state
representations at every recurrent step and sums these displacements.
Schematically,

$$
z_0 \rightarrow z_1 \rightarrow \cdots \rightarrow z_T,
$$

with a readout of the form

$$
\hat d(s,g) = \alpha \sum_t \sum_i \|z_{t+1,i}-z_{t,i}\|,
$$

where $i$ indexes relevant state/object tokens and $\alpha$ is a learned
scale.

Thus the **trajectory of the computation itself becomes the distance
readout**.

The experiments use a controlled grid-based state space with movable
entities, movement constraints, couplings/links between entities, and
reconfigurable transition rules. True shortest-path distances are
computed exactly by BFS. Experiments examine distance accuracy,
generalization to held-out transition configurations, image versus
factored observations, and extrapolation beyond training distances.

## Research questions

### A. Goal-conditioned RL and value functions as distance

Find the foundational and strongest modern papers motivating
goal-conditioned RL / universal value functions.

Explain: - why goal-conditioned RL is important for generalist agents; -
when a goal-conditioned value function can be interpreted as distance,
cost-to-go, hitting time, shortest-path distance, or a related
quantity; - distinctions between discounted value, undiscounted
shortest-path cost, successor representations, reachability, and
temporal distance; - standard terminology that would make the
introduction precise rather than overstated.

Identify canonical citations and useful formulations from: - universal
value function approximators; - goal-conditioned RL; - goal-reaching /
multi-goal RL; - temporal distance and reachability learning.

### B. Metric and quasimetric structure of value functions

Investigate work treating goal-conditioned values as metrics,
pseudometrics, asymmetric metrics, or quasimetrics.

Pay particular attention to work by **Phillip Isola and collaborators**,
including quasimetric embeddings/value functions, and trace both the
papers they build on and subsequent work.

For every relevant method, explain: - what mathematical object
represents distance; - whether the distance is symmetric or
asymmetric; - how triangle inequality or other geometric properties are
imposed; - how the distance/value is read out; - what supervision is
used; - what generalization benefits are claimed.

Determine exactly how our recurrent latent-path integral differs from
these approaches.

### C. Learned latent distances and representation geometry

Survey methods where distance/cost/value is obtained from: - Euclidean
embedding distance; - contrastive representations; - successor
features/representations; - bisimulation metrics; - temporal-distance
embeddings; - graph/geodesic embeddings; - quasimetric embeddings; -
learned planning representations.

Distinguish **static endpoint geometry** from **geometry of a
computational trajectory**.

Ask explicitly whether any prior method defines its task output as the
accumulated arc length / total variation / path length of hidden-state
updates during inference.

### D. Recurrent and iterative neural computation

Survey: - Universal Transformers; - recurrent/depth-recurrent
Transformers; - weight-tied networks; - iterative refinement; - deep
equilibrium models where relevant; - recurrent reasoning models; -
adaptive computation / pondering; - latent-space planning and iterative
reasoning.

For each close method, determine whether intermediate hidden-state
motion is merely computation, is regularized/diagnosed, or is itself
used as the predicted scalar quantity.

We particularly need to know whether **summing norms of recurrent
hidden-state changes and training that sum as the primary prediction**
has appeared before.

### E. Path length, trajectory length, and neural computation

Search broadly across ML, neuroscience-inspired ML, representation
learning, dynamical systems, and interpretability for concepts such
as: - latent path length; - representation trajectory length; -
hidden-state trajectory length; - neural trajectory arc length; - total
variation of hidden states; - cumulative representation displacement; -
computational path length; - energy/path-integral readouts; -
action/path length in latent space.

Do not stop at keyword matches. For each candidate, determine exactly
what the accumulated movement is used for: - task output; - auxiliary
loss; - regularizer; - diagnostic; - stopping criterion; - complexity
measure; - planning cost.

This distinction is central to assessing novelty.

### F. Cognitive maps and mental simulation

Investigate conceptual connections to: - cognitive maps; - Tolman; -
hippocampal spatial representations; - predictive maps / successor
representations; - mental simulation; - Kosslyn-style mental scanning; -
neural representations of distance and goal direction.

Be conservative. Determine which connections are scientifically
defensible and useful for NeurReps framing, and which would be
speculative.

In particular, assess the claim that human mental scanning time scales
with imagined distance and whether it provides a reasonable motivation
for an iterative latent-distance computation.

### G. Object-centric / slot-based representation

Find the minimum literature needed to justify the image-based model: -
Slot Attention and major descendants; - object-centric representation
learning; - object-centric dynamics/world models; - slot-based
planning/reasoning.

We do **not** want object-centric learning to become the main story.
Identify 3--6 citations sufficient to position slots as a mechanism for
recovering structured factors from images.

### H. Baselines reviewers will expect

Recommend a compact baseline set appropriate for a four-page NeurReps
extended abstract.

Prioritize baselines that isolate the scientific claim rather than
simply maximizing the number of comparisons.

Consider: - scalar MLP/head on the same backbone; - static Euclidean
embedding distance; - quasimetric methods; - IQE; - MRN; - recurrent
final-state scalar decoder; - any stronger recent goal-distance
architecture you identify.

For each recommended baseline, state **what hypothesis it tests**.

### I. Novelty stress test

Act as a skeptical reviewer.

Try to falsify each of these possible novelty claims:

1.  "Goal-conditioned value can be represented as the integrated path
    length of recurrent latent computation."
2.  "The model's own internal movement can serve directly as its scalar
    distance prediction."
3.  "Fixed-parameter recurrent computation provides a natural mechanism
    for constructing such a latent path."
4.  "This differs fundamentally from measuring distance between static
    start and goal embeddings."

Find the closest counterexamples.

For each claim classify it as: - apparently novel; - novel only with
important qualification; - already established; - too broad to defend.

Give the exact qualification needed for a defensible paper claim.

## Desired output

Produce a structured research report with:

1.  **Executive framing recommendation**
    -   2--3 candidate ways to frame the paper.
    -   Recommend the strongest one.
2.  **Literature map**
    -   Organize papers into conceptual families rather than a
        chronological bibliography.
3.  **Closest prior work table**
    -   Paper
    -   Year
    -   Problem
    -   Representation
    -   Distance/value readout
    -   Iterative/recurrent?
    -   Uses hidden trajectory length?
    -   Role of trajectory length
    -   Key difference from our method
4.  **Canonical citations for the introduction**
    -   Approximately 8--15 papers.
    -   Explain in one sentence why each deserves to be cited.
5.  **Expected baselines**
    -   Rank as essential / useful / optional.
    -   Explain the scientific comparison provided by each.
6.  **Novelty assessment**
    -   Explicitly stress-test the latent-path-integral readout.
7.  **Suggested terminology** Compare terms such as:
    -   latent trajectory integration
    -   latent path integral
    -   computational path length
    -   recurrent representation path length
    -   internal trajectory length and recommend terminology that is
        accurate and unlikely to collide with established concepts.
8.  **Suggested introduction argument**
    -   Give the logical argument as bullet points.
    -   Do not write polished prose yet.
9.  **Claims to avoid**
    -   Identify statements that would be misleading, insufficiently
        supported, or likely to irritate expert reviewers.
10. **Open literature questions**
    -   Anything that remains uncertain after the search.

## Research standards

-   Prioritize primary papers over secondary summaries.
-   Verify claims against the actual paper whenever possible.
-   Include DOI, arXiv, OpenReview, proceedings, or official publication
    links.
-   Distinguish clearly between what a paper actually demonstrates and
    our interpretation of it.
-   Search backward through references and forward through citing work
    for the closest papers.
-   Include relevant work through **August 2026**.
-   Do not infer novelty merely because terminology differs.
-   Search for the underlying mathematical operation even when papers
    use different vocabulary.
-   Treat novelty as unproven until the closest competing methods have
    been inspected in detail.
