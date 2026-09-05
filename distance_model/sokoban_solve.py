"""Lean runner for the unchanged CRTR Sokoban solver and solve configs.

Pins CRTR's namespace packages before any ML imports, preventing an unrelated
regular search.py/utils.py/envs.py on site-packages paths from taking precedence.
Only Sokoban imports are loaded. The default 6000-node tree cap and original
strict nodes<1000 budget report are preserved; --max-tree-size is for explicitly
labelled small smoke checks and is recorded in every output.
"""

import argparse
import hashlib
from contextlib import ExitStack
import importlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types


def pin_crtr_namespaces(root):
    root = Path(root).resolve()
    for name in ("search", "utils", "envs"):
        if not (root / name).is_dir():
            raise FileNotFoundError(root / name)
    if not (root / "networks.py").is_file():
        raise FileNotFoundError(root / "networks.py")
    owned = ("search", "utils", "envs", "networks")
    loaded = [name for name in sys.modules if name.split(".", 1)[0] in owned]
    if loaded:
        raise RuntimeError(f"CRTR import names already loaded; use a fresh process: {loaded}")
    # Do not execute package __init__ files or consult other sys.path directories.
    for name in ("search", "utils", "envs"):
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(root / name)]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = module.__path__
        sys.modules[name] = module
    return root


def import_exact_file(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        existing = sys.modules[name]
        if getattr(existing, "__file__", None) and Path(existing.__file__).resolve() == path:
            return existing
        raise RuntimeError(f"Refusing preloaded module {name} from a different file")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_crtr_modules(root):
    root = pin_crtr_namespaces(root)
    modules = {"networks": import_exact_file("networks", root / "networks.py")}
    for name in (
        "utils.metric_logging", "utils.jax_rand", "search.value_function",
        "search.value_function_baseline", "search.goal_builder", "search.solve_job",
        "search.solver", "envs.sokoban.sokoban_env", "envs.sokoban.gen_problems_sokoban",
    ):
        module = importlib.import_module(name)
        expected = (root / (name.replace(".", "/") + ".py")).resolve()
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError(f"Wrong CRTR module path for {name}: {module.__file__}")
        modules[name] = module
    return modules


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solve_metrics(job):
    return {
        "solved": job.solved_boards,
        "boards_evaluated": job.all_boards,
        "solved_rate": job.solved_boards / job.all_boards,
        "solved_rate_by_node_budget": {
            str(k): v / job.all_boards for k, v in job.budget_solved.items()},
        "solved_rate_by_expanded_node_budget": {
            str(k): v / job.all_boards for k, v in job.budget_exp_solved.items()},
    }


def raw_supervised_state(checkpoint):
    """Extract our training wrapper while retaining the exact original network."""
    if not isinstance(checkpoint, dict) or checkpoint.get("model") != "supervised":
        raise ValueError("Expected a supervised-ours checkpoint with model='supervised'")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Supervised checkpoint has no state_dict")
    if any(not isinstance(key, str) or not key.startswith("network.") for key in state):
        raise ValueError("Every supervised wrapper key must have the exact network. prefix")
    return {key[len("network."):]: value for key, value in state.items()}


def run(job_class, seed, output_dir):
    """Same seeding, logger and SolveJob execution order as CRTR runner.run."""
    import random
    import numpy as np
    import torch
    from utils import metric_logging
    from utils.jax_rand import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    loggers = metric_logging.Loggers()
    loggers.register_logger(metric_logging.StdoutLogger(output_dir=str(output_dir)))
    loggers.log_property("seed", seed)
    job = job_class(loggers, output_dir=str(output_dir))
    job.execute()
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crtr-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=("joint", "supervised", "supervised-ours"), required=True)
    parser.add_argument("--mode", choices=("search", "greedy"), required=True)
    parser.add_argument("--n-jobs", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tree-size", type=int, default=6000,
                        help="6000 is the reference protocol; small values are smoke checks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boards", type=Path)
    args = parser.parse_args(argv)
    if args.n_jobs < 1 or args.max_tree_size < 1:
        parser.error("n-jobs and max-tree-size must be positive")
    root = args.crtr_root.resolve()
    checkpoint = args.checkpoint.resolve()
    boards = (args.boards or root / "training_datasets/sokoban_eval_boards/eval_boards.pkl").resolve()
    for path in (checkpoint, boards):
        if not path.is_file():
            raise FileNotFoundError(path)
    modules = load_crtr_modules(root)
    import gin
    import torch
    configured_run = gin.configurable(run)
    config = root / "configs" / "solve" / ("search" if args.mode == "search" else "no-search") / "supervised" / "sokoban.gin"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(module.__file__).resolve() for name, module in modules.items()}
    sources["sokoban_solve"] = Path(__file__).resolve()
    if args.kind == "joint":
        for name in ("value_estimator_sokoban_joint", "sokoban_joint", "autoresearch_joint"):
            sources[name] = Path(__file__).resolve().with_name(name + ".py")
    manifest = dict(kind=args.kind, mode=args.mode, n_jobs=args.n_jobs, seed=args.seed,
        max_tree_size=args.max_tree_size, reference_tree_cap=6000,
        smoke_only=args.max_tree_size != 6000 or args.n_jobs != 1000,
        checkpoint=str(checkpoint), boards=str(boards), crtr_root=str(root),
        sources={name: str(path) for name, path in sources.items()},
        source_sha256={name: file_sha256(path) for name, path in sources.items()},
        checkpoint_sha256=file_sha256(checkpoint), boards_sha256=file_sha256(boards))
    with (output / "solve_config.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print("SOKOBAN_SOLVE_CONFIG " + json.dumps(manifest, sort_keys=True), flush=True)
    with ExitStack() as stack:
        load_path = checkpoint
        if args.kind == "supervised-ours":
            wrapped = torch.load(checkpoint, map_location="cpu", weights_only=True)
            raw = raw_supervised_state(wrapped)
            temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="supervised-raw-")))
            load_path = temporary / "model.pt"
            torch.save(raw, load_path)
        bindings = [
            f"run.seed = {args.seed}",
            f"BestFSSolver.checkpoint_path = {str(load_path)!r}",
            f"CustomSokobanEnv.boards_path = {str(boards)!r}",
            f"SolveJob.n_actions = {12 if args.mode == 'search' else 1}",
            f"SolveJob.n_jobs = {args.n_jobs}",
            "SolveJob.n_parallel_workers = 1",
            f"SolveJob.batch_size = {min(200, args.n_jobs)}",
            f"BestFSSolver.max_tree_size = {args.max_tree_size}",
            "SolveJob.budget_checkpoints = [50, 100, 500, 1000]",
        ]
        if args.kind == "joint":
            import_exact_file("value_estimator_sokoban_joint",
                              Path(__file__).with_name("value_estimator_sokoban_joint.py"))
            bindings += ["SolveJob.network = None",
                         "BestFSSolver.value_estimator_class = @ValueEstimatorSokobanJoint"]
        gin.parse_config_files_and_bindings(config_files=[str(config)], bindings=bindings)
        with (output / "hyperparameters.txt").open("x") as stream:
            stream.write(gin.config_str() + "\n\nBindings:\n" + "\n".join(bindings) + "\n")
        job = configured_run(output_dir=str(output))
        result = {**manifest, **solve_metrics(job)}
        with (output / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        print("RESULT " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return result


if __name__ == "__main__":
    main()

