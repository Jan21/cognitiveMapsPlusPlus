---
summary: Plain-language explainer of the distance model (markdown twin of explainer.html): the world, inputs, model, supervision, novelty, results, glossary.
---

# How far is the goal? A neural network that measures distance by imagining the trip

**When:** 2026-08-12 08:14   **Repo:** cognitiveMapsPlusPlus   **Branch:** distance-model

This is the markdown twin of `distance_model/explainer.html`, written for a reader with no prior context. It explains the inputs, the model, the supervision, and the motivation of the distance-model project.

## 1. Motivation: why care about distance at all?

Almost everything an intelligent agent does starts with a quiet judgment of effort: is the coffee machine nearer than the kitchen downstairs, is this chess position close to a win, is the current draft far from the report I owe. Before any plan exists, something already estimates how far away the goal is.

In goal-directed reinforcement learning that judgment has a name: the value function. You give it two things, where you are now and where you want to be, and it returns a number. In our setting the number has a concrete meaning: the minimum number of moves needed to get from here to there. Distance, in moves.

Humans seem to compute such estimates by mental simulation. A classic 1978 experiment (Kosslyn's mental scanning) found that when people imagine traveling across a memorized map, their answer time grows linearly with the distance: farther places take longer to imagine. The mind seems to actually make the trip, at some bounded speed, and the effort of the imagined journey is the estimate.

This project builds a neural network in that spirit. It does not read the distance off a dial at the end of its computation. It takes an imagined journey inside its own head, and the distance is how far its thoughts traveled.

## 2. The world: a puzzle where distance is not geometry

To test whether a network truly understands distance, we need a world where distance cannot be eyeballed. Ours is a small grid, typically 6 by 6, occupied by a handful of pieces:

- **One free mover** that can always step up, down, left, or right.
- **Several constrained pieces**, each governed by a knob with four settings: free, horizontal moves only, vertical only, or locked in place.
- **Links** between pairs of constrained pieces. Linked pieces are welded together: they move as one rigid group, and the group can only make moves every member's knob allows.
- **A control cell** in the corner. When the free mover stands on it, it may spend moves changing knobs or toggling links, reconfiguring the rules of the world mid-journey.

This coupling is the point. The same two snapshots can be three moves apart or thirteen, depending on the knob and link settings: a locked piece linked to a free one freezes both, and sometimes the shortest route is a detour where the mover first walks to the control cell, unlocks something, and only then heads for the goal. Distance here is a property of the rules, not of straight-line geometry, so no simple formula gives the answer. The exact answer does exist, though: the world is small enough that breadth-first search (BFS) can count the true minimum number of moves between any two states. That exact count is our ground truth.

## 3. Inputs: what the model sees

Every query is a pair of states: the current one and the goal. A state is a short description of the world: the grid position of each piece, the knob setting of each constrained piece, and which links are active.

We feed this in three formats of increasing difficulty, to separate "can it compute distance" from "can it also perceive the scene":

| Format | What the network gets | What it must learn |
|---|---|---|
| factored | each piece's position handed over cleanly, as a separate slot | only the distance computation itself |
| bmask | a rendered picture of the grid, each piece still recoverable exactly | distance, from an image, with perception made easy |
| marker | the raw shared picture only | distance AND perception: it must learn by itself which blob on the canvas is which piece |

In every format, the knob and link settings also enter as one extra "rulebook" token attached to the state.

## 4. The model: the imagined journey

The model is a small transformer (the same family of network as language models, here a few million parameters). The state becomes a handful of tokens: one per piece, plus the rulebook token. Then the distinctive loop begins.

One transformer block is applied over and over, exactly T times (we use 14), with the same weights every time; think of it as fourteen moments of thought. At every step, two anchors are re-attached to the sequence: a copy of the goal and a copy of the original start. We call this recall; it keeps the imagination from forgetting where it came from and where it is heading while its working copy of the state drifts.

And here is the heart of the project. At each of the fourteen steps we measure how much the state tokens moved inside the network's internal representation space, and we add those movements up. Nothing else is read out. The final prediction is that accumulated internal mileage, times one learned scaling constant:

```
repeat 14 times:
    z     = TransformerBlock(tokens)
    cost += sum of ||how far every state token just moved||
    tokens = z, with goal and start re-attached

distance = scale * cost
```

Notice what is NOT here. There is no output neuron trained to emit "7". There is no ruler laid between an embedding of the start and an embedding of the goal. The trajectory of the computation itself is the answer, the way the duration of your imagined walk across a city is your estimate of its size.

**A fixed thought budget.** The network always thinks for exactly fourteen steps, whether the goal is two moves away or twenty. A farther goal does not get more thinking time; it makes the internal journey faster, each thought-step covering more representational ground. This matters for generalization: magnitude lives in the step sizes, not the step count, so reporting larger distances needs no extra compute.

## 5. Supervision: one signal only

Training is deliberately austere. For pairs of states we compute the true shortest-path count with exact graph search, and the network's prediction is pulled toward that number:

```
loss = smooth_L1( model(s, g),  true_BFS_distance(s, g) )
```

No rewards, no demonstrations, no route information, no hints about knobs or links beyond the raw state description, and no policy: the model is never asked to act, only to judge. In the harder experiments the training pairs are restricted (held-out constraint configurations, or, with the new `--Rtrain` flag, only nearby pairs), and the model is then tested on configurations or distances it never saw.

## 6. Why the readout is the interesting part

Neural networks that estimate distance-to-goal are not new. What is new is where the number comes from. Existing methods read distance in one of three ways:

1. **Output head**: a neuron at the end says the number.
2. **Embedding ruler**: embed both states, measure the gap between the two points.
3. **Iterate then decode**: run a recurrent network and read the answer from the final state.

Ours is a fourth way: sum the lengths of the network's own internal steps. The thinking is the measurement.

We checked the literature carefully, running four independent automated deep-research surveys (ChatGPT, Claude, Gemini, Grok) over reinforcement learning, geometry of neural representations, iterative reasoning networks, PDE-based distance solvers, and cognitive science. All four came back with the same finding: the ingredients exist separately, and the "total internal movement" quantity even appears in a few papers, but always in a supporting role: a training penalty, a diagnostic statistic, or a signal for when to stop computing. No prior work uses it as the trained output, and none combines it with the fixed thought budget. That readout appears to be new. (Full survey: `distance_model/prior_work.html`.)

## 7. Does it work?

Three fronts, from the completed experiments:

- **Accuracy.** With clean or lossless input, predictions land within roughly a tenth to a quarter of a move of the exact answer, on distances up to a dozen moves (cluster reference numbers: test MAE 0.10 to 0.29 across splits and seeds, correlation 0.98+). With the full image input, where it must also learn perception, the marker runs reach test MAE 0.07 to 0.25 on most splits with longer training.
- **Understanding the rules.** Given a state, it tends to rank positions reachable by a legal move as closer than positions that would require breaking a constraint, so the constraint system is reflected in its distances (this ranking metric is noisy across runs, though).
- **Generalization over configurations.** Tested on knob-and-link configurations held out of training entirely, including structurally new ones (multiple links where training only showed one) and higher-freedom worlds, distance estimates remain accurate.

**Honest open front: length extrapolation.** A new experiment trains only on distances up to 4 and tests up to 12. First result: within-range error 0.16 (good), beyond-range error 4.1 with no correlation (failure). This matches a known artifact from earlier probes in the project (a constant base cost folded into every step's increment) whose fix, a global-base lever, has not yet been ported into this clean experiment. A cluster sweep validating all of this on the same hardware as the original results is running now.

## 8. The bigger picture

This experiment is one piece of a larger project on cognitive maps: the internal representations that let animals and people navigate, plan, and judge effort without exhaustively simulating the world. The distance model shows one concrete, testable version of an old intuition from psychology: that estimating "how far" may literally be an act of imagination, and that the effort of the imagined journey can serve as the measurement itself. Next steps: the global-base port for length extrapolation, baselines against the strongest existing distance architectures (IQE, MRN, scalar head), and a write-up positioning the readout among its neighbors.

## Glossary

- **State**: a full snapshot of the world: every piece's position plus current knob and link settings.
- **Goal-conditioned**: the model's answer depends on a goal you hand it, so one network answers "how far to X" for every X.
- **Value function**: in reinforcement learning, the function scoring how good a situation is; here, negative distance-to-goal plays that role.
- **BFS / geodesic distance**: the exact minimum number of legal moves between two states, found by breadth-first graph search. Our ground truth.
- **Token**: one slot in a transformer's input; here, one piece of the world (or the rulebook) as a learned vector.
- **Weight-shared / recurrent**: the same block of weights applied repeatedly, one thought process iterated, rather than a stack of different layers.
- **Extrapolation**: answering correctly outside the training range: longer distances, unseen rule configurations.
