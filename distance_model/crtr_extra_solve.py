"""Evaluate frozen extra-task cases with the unchanged CRTR BestFSSolver.

Generated-node and expanded-node counters are reported independently. Budget
success means solved AND original nodes < budget, never <= or an alternate cap.
Already-solved roots are explicitly handled at the adapter boundary (one root,
zero expansions/actions), because the original solver omits that base case.
All nonterminal cases use original BestFSSolver and TrivialPolicy implementations.
"""

import argparse
import importlib
import json
from numbers import Integral
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

if __package__:
    from .sokoban_solve import pin_crtr_namespaces, file_sha256
    from .crtr_extra_envs import lights_toggle_matrix, digit_encode
else:
    from sokoban_solve import pin_crtr_namespaces, file_sha256
    from crtr_extra_envs import lights_toggle_matrix, digit_encode


BUDGETS = (50, 100, 500, 1000, 6000)
SIZES = {"lights": 7, "digit": 20}


def load_original_solver(root):
    """Pin namespaces before importing any CRTR search modules or model code."""
    root = pin_crtr_namespaces(root)
    modules = {}
    for name in ("search.solver", "search.goal_builder"):
        module = importlib.import_module(name)
        expected = root / (name.replace(".", "/") + ".py")
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError(f"wrong CRTR module source for {name}")
        modules[name] = module
    return (modules["search.solver"].BestFSSolver, modules["search.goal_builder"].TrivialPolicy,
            {name: str(Path(module.__file__).resolve()) for name, module in modules.items()})


class ExtraPuzzleEnv:
    """Stateless step(state, action) adapter consumed by original TrivialPolicy."""

    def __init__(self, env, size=None):
        if env not in SIZES:
            raise ValueError("env must be lights or digit")
        self.env = env
        self.size = SIZES[env] if size is None else size
        if not isinstance(self.size, Integral) or self.size < 1:
            raise ValueError("size must be a positive integer")
        self.initial = self.goal = None
        self.board = None
        self.masks = lights_toggle_matrix(self.size).T if env == "lights" else None

    def _flat(self, state):
        array = np.asarray(state)
        if array.size != self.size * self.size or array.dtype.kind not in "ui":
            raise ValueError("state must contain exactly size*size categorical integer tiles")
        flat = array.reshape(-1)
        if self.env == "lights":
            if np.any(flat > 1) or np.any(flat < 0):
                raise ValueError("Lights states must be binary")
        elif np.any(flat < 1) or np.any(flat > 12) or np.count_nonzero(flat > 6) != 1:
            raise ValueError("Digit states must have digits1..6 and exactly one marked digit7..12")
        return flat.astype(np.uint8, copy=False)

    def _digit_layout(self, flat):
        position = int(np.flatnonzero(flat > 6)[0])
        digits = flat.copy()
        digits[position] -= 6
        return digits.reshape(self.size, self.size), position

    def set_problem(self, initial, goal):
        initial, goal = self._flat(initial), self._flat(goal)
        if self.env == "digit":
            self.board, _ = self._digit_layout(initial)
            target_board, _ = self._digit_layout(goal)
            if not np.array_equal(self.board, target_board):
                raise ValueError("Digit state and goal must preserve the same digit board")
        self.initial, self.goal = tuple(map(int, initial)), tuple(map(int, goal))

    def reset(self):
        if self.initial is None:
            raise RuntimeError("set_problem must be called before reset")
        return self.initial

    def get_all_actions(self):
        return list(range(self.size * self.size if self.env == "lights" else 4))

    def step(self, state, action):
        if self.goal is None:
            raise RuntimeError("set_problem must be called before step")
        count = self.size * self.size if self.env == "lights" else 4
        if not isinstance(action, Integral) or isinstance(action, bool) or not 0 <= action < count:
            raise ValueError("invalid action index")
        flat = self._flat(state)
        if self.env == "lights":
            updated = flat ^ self.masks[action]
        else:
            board, position = self._digit_layout(flat)
            if not np.array_equal(board, self.board):
                raise ValueError("Digit layout changed during a case")
            row, col = divmod(position, self.size)
            jump = int(board[row, col])
            dr, dc = ((-1, 0), (0, 1), (0, -1), (1, 0))[action]
            rr, cc = row + jump * dr, col + jump * dc
            target = rr * self.size + cc if 0 <= rr < self.size and 0 <= cc < self.size else position
            updated = digit_encode(board, target).reshape(-1)
        successor = tuple(map(int, updated))
        done = successor == self.goal
        return successor, float(done), done, {}


class ExtraValueEstimator:
    def __init__(self, model, kind, device):
        self.model, self.kind, self.device = model, kind, device

    def construct_networks(self):
        self.model.eval()

    def get_solved_distance(self, state, goal):
        import torch
        states = torch.as_tensor(state, device=self.device, dtype=torch.uint8).reshape(1, -1)
        goals = torch.as_tensor(goal, device=self.device, dtype=torch.uint8).reshape(1, -1)
        with torch.inference_mode():
            output = self.model(states, goals)
            if not torch.isfinite(output).all():
                raise FloatingPointError("nonfinite solver model output")
            if self.kind == "joint":
                if output.shape != (1,):
                    raise ValueError("joint solver model must return a B-vector")
                value = output[0]
            else:
                if output.ndim != 2 or output.shape[0] != 1:
                    raise ValueError("supervised solver model must return BxBins logits")
                value = output.argmax(1)[0]
            score = float(value.item())
            if score < 0:
                raise ValueError("distance prediction must be nonnegative")
            return score


def make_solver(solver_class, policy_class, env, estimator, n_actions, max_tree_size):
    solver = solver_class(metric=None, shuffles=0, network=None, n_actions=n_actions,
        goal_builder_class=lambda shuffles: policy_class(shuffles=shuffles, env=lambda shuffles: env),
        max_tree_size=max_tree_size, max_tree_depth=-1, checkpoint_path=None,
        value_estimator_class=lambda network, checkpoint_path=None, metric=None: estimator)
    solver.construct_networks()
    return solver


def solve_case(solver, env, initial, goal, index):
    env.set_problem(initial, goal)
    start = time.time()
    if env.initial == env.goal:
        return dict(index=index, solved=True, nodes=1, generated_nodes=1, expanded_nodes=0,
                    length=0, original_solution_nodes=1, solver_invoked=False,
                    actions=[], seconds=time.time() - start,
                    finished_cause="already-solved root handled explicitly by adapter")
    solution, tree_metrics, root, actions, info = solver.solve(env.initial, env.goal)
    solved = solution is not None
    if solved:
        if actions is None:
            raise ValueError("original solver reported a solution without a trajectory")
        replay = env.initial
        for action in actions:
            replay, _, _, _ = env.step(replay, action)
        if replay != env.goal:
            raise ValueError("original solver's trajectory failed environment replay")
    nodes, expanded = int(tree_metrics["nodes"]), int(tree_metrics["expanded_nodes"])
    if nodes < 1 or expanded < 0:
        raise ValueError("invalid original solver node counters")
    return dict(index=index, solved=solved, nodes=nodes, generated_nodes=nodes,
                expanded_nodes=expanded, length=len(actions) if solved else None,
                original_solution_nodes=len(solution) if solved else None,
                solver_invoked=True, actions=[int(action) for action in actions] if solved else None,
                seconds=time.time() - start, tree_metrics=tree_metrics,
                finished_cause=info.get("finished_cause"))


def aggregate_records(records):
    if not records:
        raise ValueError("cannot aggregate zero cases")
    count = len(records)
    solved = [row for row in records if row["solved"]]
    return dict(boards_evaluated=count, solved=len(solved), solved_rate=len(solved) / count,
        mean_solution_length=float(np.mean([row["length"] for row in solved])) if solved else None,
        solved_rate_by_node_budget={str(budget): sum(row["nodes"] < budget for row in solved) / count
                                   for budget in BUDGETS},
        solved_rate_by_expanded_node_budget={str(budget): sum(row["expanded_nodes"] < budget for row in solved) / count
                                            for budget in BUDGETS})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("lights", "digit"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crtr-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--mode", choices=("greedy", "search"), required=True)
    parser.add_argument("--n-jobs", type=int, default=1000)
    parser.add_argument("--max-tree-size", type=int, default=6000)
    parser.add_argument("--out", type=Path, required=True, help="artifact prefix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.n_jobs, args.max_tree_size, args.torch_threads) < 1:
        parser.error("n-jobs, max-tree-size and torch-threads must be positive")
    prefix = args.out.resolve()
    outputs = {name: Path(str(prefix) + suffix) for name, suffix in
               (("config", ".config.json"), ("cases", ".cases.jsonl"), ("result", ".json"))}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite solver artifacts")
    root, checkpoint_path, cases_path = args.crtr_root.resolve(), args.checkpoint.resolve(), args.cases.resolve()
    for path in (checkpoint_path, cases_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    solver_class, policy_class, sources = load_original_solver(root)
    # Namespace pinning precedes all torch/model/network imports.
    import torch
    if __package__:
        from .crtr_extra_train import load_extra_checkpoint
    else:
        from crtr_extra_train import load_extra_checkpoint
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, checkpoint = load_extra_checkpoint(checkpoint_path, device=device, crtr_root=root)
    if checkpoint["env"] != args.env:
        raise ValueError("checkpoint environment does not match --env")
    with np.load(cases_path, allow_pickle=False) as archive:
        states, goals = archive["states"], archive["goals"]
    size = SIZES[args.env]
    if (states.dtype != np.uint8 or goals.dtype != np.uint8 or states.ndim != 3
            or states.shape[1:] != (size, size) or goals.shape != states.shape
            or len(states) < args.n_jobs):
        raise ValueError("cases must contain enough matching uint8 states/goals with the environment grid size")
    env = ExtraPuzzleEnv(args.env)
    # Validate every requested pair before evaluating any case.
    for index in range(args.n_jobs):
        env.set_problem(states[index], goals[index])
    env.set_problem(states[0], goals[0])
    n_actions = 1 if args.mode == "greedy" else (10 if args.env == "lights" else 4)
    estimator = ExtraValueEstimator(model, checkpoint["model"], device)
    solver = make_solver(solver_class, policy_class, env, estimator, n_actions, args.max_tree_size)
    for name in ("crtr_extra_solve", "crtr_extra_envs", "crtr_extra_train", "sokoban_solve", "sokoban_joint", "autoresearch_joint"):
        sources[name] = str(Path(__file__).with_name(name + ".py").resolve())
    if checkpoint["model"] != "joint":
        sources["networks"] = str(root / "networks.py")
    manifest = dict(env=args.env, model=checkpoint["model"], mode=args.mode, seed=args.seed,
        n_jobs=args.n_jobs, n_actions=n_actions, max_tree_size=args.max_tree_size,
        max_tree_depth=-1, budgets=list(BUDGETS), budget_rule="solved and original generated nodes < budget",
        initial_goal_rule="already solved: one root, zero expansions and zero actions; explicitly marked",
        smoke_only=args.n_jobs != 1000 or args.max_tree_size != 6000,
        device=str(device), checkpoint=str(checkpoint_path), cases=str(cases_path), crtr_root=str(root),
        checkpoint_sha256=file_sha256(checkpoint_path), cases_sha256=file_sha256(cases_path),
        sources=sources, source_sha256={name: file_sha256(path) for name, path in sources.items()},
        checkpoint_training_sources=checkpoint.get("source_sha256"),
        command=[sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with outputs["config"].open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("CONFIG " + json.dumps(manifest, sort_keys=True, allow_nan=False), flush=True)
    start = time.time()
    records = []
    with outputs["cases"].open("x") as stream:
        for index in range(args.n_jobs):
            # Exceptions propagate: partial case logs never become a final RESULT.
            record = solve_case(solver, env, states[index], goals[index], index)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
            print("CASE " + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    if len(records) != args.n_jobs:
        raise RuntimeError("incomplete solver run")
    result = {**manifest, **aggregate_records(records), "seconds": time.time() - start}
    with outputs["result"].open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("RESULT " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
