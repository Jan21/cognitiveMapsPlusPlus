"""Direct-distance Sokoban transfer on frozen clean-BFS NPZ banks.

Banks contain states/goals: uint8 (N,144), dist: float32 (N,), optionally
metadata_json/provenance JSON scalars. NPZ members load once into host memory;
NumPy cannot memory-map compressed archive members. No trajectories are sampled
or labels regenerated here. Evaluation during training uses validation only.

The joint model returns only accumulated latent motion. The supervised control
imports the original CRTR networks.LNConvNet and learns 150 CE distance bins.
Both models draw identical IID rows and joint D4 transformations for a given seed.

Solver API: model, metadata = load_joint_checkpoint(path, device="cpu").
model(flat_states, flat_goals) accepts matching B144 tile tensors and returns B.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .autoresearch_joint import JointPixelInteg
except ImportError:
    from autoresearch_joint import JointPixelInteg


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenBank:
    def __init__(self, path, require_solved_goal=True):
        self.path = Path(path).resolve()
        metadata = {}
        with np.load(self.path, allow_pickle=False, mmap_mode="r") as archive:
            self.states = archive["states"]
            self.goals = archive["goals"]
            self.dist = archive["dist"]
            for key in ("metadata_json", "provenance_json", "provenance", "metadata"):
                if key in archive:
                    value = archive[key]
                    if value.ndim == 0 and value.dtype.kind in "US":
                        text = value.item()
                        if isinstance(text, bytes):
                            text = text.decode()
                        try:
                            metadata[key] = json.loads(text)
                        except json.JSONDecodeError:
                            metadata[key] = text
        if (self.states.dtype != np.uint8 or self.goals.dtype != np.uint8
                or self.states.ndim != 2 or self.states.shape[1] != 144
                or self.goals.shape != self.states.shape or len(self.states) == 0):
            raise ValueError("bank states/goals must be nonempty matched uint8 (N,144) arrays")
        if (self.dist.dtype != np.float32 or self.dist.shape != (len(self.states),)
                or not np.isfinite(self.dist).all() or np.any(self.dist < 0)):
            raise ValueError("bank dist must be finite nonnegative float32 (N,)")
        # Bound validation temporaries for large banks.
        for start in range(0, len(self.states), 16384):
            states, goals = self.states[start:start + 16384], self.goals[start:start + 16384]
            if np.any(states > 6) or np.any(goals > 6):
                raise ValueError("CRTR tile IDs must be in 0..6")
            if not np.array_equal(states == 0, goals == 0):
                raise ValueError("paired boards must preserve walls")
            state_targets = (states == 2) | (states == 3) | (states == 6)
            goal_targets = (goals == 2) | (goals == 3) | (goals == 6)
            if not np.array_equal(state_targets, goal_targets):
                raise ValueError("paired boards must preserve target locations")
            if require_solved_goal:
                expected = np.ones_like(states)
                expected[states == 0] = 0
                expected[state_targets] = 3
                if not np.array_equal(goals, expected):
                    raise ValueError("goals must be synthetic agent-less solved frames preserving walls and targets")
        self.provenance = dict(path=str(self.path), sha256=file_sha256(self.path), rows=len(self.states),
            distance_min=float(self.dist.min()), distance_max=float(self.dist.max()), metadata=metadata,
            require_solved_goal=bool(require_solved_goal))

    def __len__(self):
        return len(self.states)


def onehot_grid(flat):
    if flat.ndim != 2 or flat.shape[1] != 144:
        raise ValueError("Sokoban inputs must have shape (B,144)")
    return F.one_hot(flat.long(), num_classes=7).reshape(-1, 12, 12, 7).permute(0, 3, 1, 2).float()


def augment_pair(states, goals, codes):
    """Apply one of eight D4 transforms to each pair, preserving state/goal alignment."""
    if states.shape != goals.shape or states.shape != (len(codes), 144):
        raise ValueError("augmentation expects matched B144 pairs and one D4 code per pair")
    codes = np.asarray(codes)
    if np.any((codes < 0) | (codes > 7)):
        raise ValueError("D4 codes must be in 0..7")
    left, right = states.reshape(-1, 12, 12), goals.reshape(-1, 12, 12)
    output_left, output_right = torch.empty_like(left), torch.empty_like(right)
    for code in range(8):
        indices = torch.as_tensor(np.flatnonzero(codes == code), device=states.device)
        if not len(indices):
            continue
        a = torch.rot90(left[indices], code % 4, (-2, -1))
        b = torch.rot90(right[indices], code % 4, (-2, -1))
        if code >= 4:
            a, b = a.flip(-1), b.flip(-1)
        output_left[indices], output_right[indices] = a, b
    return output_left.flatten(1), output_right.flatten(1)


def draw_batch(bank, rng, batch_size, augment=True):
    indices = rng.integers(0, len(bank), size=batch_size)
    # Draw codes even with augmentation disabled to keep IID row streams matched.
    codes = rng.integers(0, 8, size=batch_size)
    states = torch.from_numpy(bank.states[indices])
    goals = torch.from_numpy(bank.goals[indices])
    if augment:
        states, goals = augment_pair(states, goals, codes)
    return states, goals, torch.from_numpy(bank.dist[indices]), indices, codes


class SokobanJoint(nn.Module):
    def __init__(self, width=64, T=8, attention_heads=2):
        super().__init__()
        self.core = JointPixelInteg(in_channels=7, width=width, T=T, kernel_size=3,
                                   tied=True, blocks=1, stride=1, reinject=True,
                                   attention_heads=attention_heads)

    def forward(self, states, goals):
        return self.core(onehot_grid(states), onehot_grid(goals))


class SupervisedPair(nn.Module):
    def __init__(self, crtr_root):
        super().__init__()
        if not crtr_root:
            raise ValueError("--crtr-root is required for the original supervised network")
        root = Path(crtr_root).resolve()
        if not (root / "networks.py").is_file():
            raise ValueError("--crtr-root must contain the original networks.py")
        sys.path.insert(0, str(root))
        try:
            networks = importlib.import_module("networks")
        finally:
            sys.path.pop(0)
        if Path(networks.__file__).resolve() != root / "networks.py":
            raise ValueError("a different networks module was already imported")
        self.network = networks.LNConvNet(input_size=14, repr_dim=150, hidden_size=64,
                                           depth=8, baseline=True)
        self.source_path = root / "networks.py"

    def forward(self, states, goals):
        return self.network(torch.cat((states, goals), dim=1))


def load_joint_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("model") != "joint":
        raise ValueError("expected a sokoban_joint joint-model checkpoint")
    model = SokobanJoint(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat((starts + ends - 1) / 2, ends - starts)
    return ranks


def correlation(left, right):
    left, right = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    # Constant targets/predictions have undefined correlation, represented as JSON null.
    return float(np.clip(np.dot(left, right) / denominator, -1, 1)) if denominator else None


def distance_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (truth.ndim != 1 or prediction.shape != truth.shape or not len(truth)
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError("metrics require nonempty, matching, finite vectors")
    error = prediction - truth
    return dict(n=len(truth), mae=float(np.abs(error).mean()), bias=float(error.mean()),
                rmse=float(np.sqrt(np.square(error).mean())), pearson=correlation(truth, prediction),
                spearman=correlation(average_ranks(truth), average_ranks(prediction)))


def evaluate(model, bank, kind, device, batch_size):
    predictions = {"distance": []} if kind == "joint" else {"argmax": [], "expectation": []}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(bank), batch_size):
                states = torch.from_numpy(bank.states[start:start + batch_size]).to(device)
                goals = torch.from_numpy(bank.goals[start:start + batch_size]).to(device)
                output = model(states, goals)
                if not torch.isfinite(output).all():
                    raise FloatingPointError("nonfinite evaluation output")
                if kind == "joint":
                    predictions["distance"].append(output.cpu().numpy())
                else:
                    predictions["argmax"].append(output.argmax(1).float().cpu().numpy())
                    expectation = (output.softmax(1) * torch.arange(150, device=device)).sum(1)
                    predictions["expectation"].append(expectation.cpu().numpy())
    finally:
        model.train(was_training)
    arrays = {name: np.concatenate(parts).astype(np.float32) for name, parts in predictions.items()}
    return {name: distance_metrics(bank.dist, array) for name, array in arrays.items()}, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("joint", "supervised"), default="joint")
    parser.add_argument("--targets", choices=("bfs", "gap"), default="bfs",
                        help="training/validation bank targets; final test always requires synthetic solved goals")
    parser.add_argument("--train-bank", type=Path)
    parser.add_argument("--val-bank", type=Path, required=True)
    parser.add_argument("--test-bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="artifact path prefix")
    parser.add_argument("--crtr-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--eval-bs", type=int, default=128)
    parser.add_argument("--evalevery", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--gradclip", type=float, default=1.)
    parser.add_argument("--augment", type=int, choices=(0, 1), default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.eval_only != bool(args.checkpoint):
        parser.error("--checkpoint and --eval-only must be used together; training resumes are not supported")
    if not args.eval_only and (args.train_bank is None or args.steps < 1):
        parser.error("training requires --train-bank and positive --steps")
    if min(args.bs, args.eval_bs, args.torch_threads) < 1 or min(args.warmup, args.evalevery) < 0:
        parser.error("batch sizes/threads must be positive; warmup/evalevery nonnegative")
    if not all(math.isfinite(value) and value > 0 for value in (args.lr, args.gradclip)):
        parser.error("lr and gradclip must be positive and finite")
    paths = [path.resolve() for path in (args.train_bank, args.val_bank, args.test_bank) if path is not None]
    if len(paths) != len(set(paths)):
        parser.error("train, validation, and test banks must use distinct paths")
    prefix = args.out.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {name: Path(str(prefix) + suffix) for name, suffix in
               (("checkpoint", ".pt"), ("result", ".json"), ("predictions", ".predictions.npz"),
                ("validation_log", ".validation.jsonl"))}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    banks = {"validation": FrozenBank(args.val_bank, require_solved_goal=args.targets == "bfs"),
             "test": FrozenBank(args.test_bank, require_solved_goal=True)}
    if args.train_bank is not None:
        banks["train"] = FrozenBank(args.train_bank, require_solved_goal=args.targets == "bfs")
    if args.model == "supervised":
        for bank in banks.values():
            if np.any(bank.dist >= 150) or np.any(bank.dist != np.floor(bank.dist)):
                raise ValueError("the exact 150-bin supervised control requires integer distances in 0..149")
    config = dict(width=args.width, T=args.T, attention_heads=args.attention_heads)
    if args.model == "joint":
        if args.eval_only:
            model, loaded = load_joint_checkpoint(args.checkpoint, device)
            config = loaded["config"]
        else:
            model = SokobanJoint(**config).to(device)
    else:
        model = SupervisedPair(args.crtr_root).to(device)
        config = dict(input_size=14, repr_dim=150, hidden_size=64, depth=8, baseline=True)
        if args.eval_only:
            loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
            if isinstance(loaded, dict) and "state_dict" in loaded:
                if loaded.get("model") != "supervised":
                    raise ValueError("checkpoint is not the supervised Sokoban control")
                model.load_state_dict(loaded["state_dict"], strict=True)
            else:
                model.network.load_state_dict(loaded, strict=True)
    training = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.eval_only:
        training["steps"] = 0
    provenance = {name: bank.provenance for name, bank in banks.items()}
    sources = {"sokoban_joint.py": file_sha256(__file__),
               "autoresearch_joint.py": file_sha256(Path(__file__).with_name("autoresearch_joint.py"))}
    if args.model == "supervised":
        sources["networks.py"] = file_sha256(model.source_path)
    print("CONFIG " + json.dumps(dict(model=args.model, config=config, training=training, banks=provenance), sort_keys=True), flush=True)
    trace = hashlib.sha256()
    start = time.time()
    with outputs["validation_log"].open("x") as validation_log:
        if not args.eval_only:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            for step in range(1, args.steps + 1):
                states, goals, distances, indices, codes = draw_batch(banks["train"], rng, args.bs, bool(args.augment))
                trace.update(indices.astype("<i8").tobytes())
                trace.update(codes.astype(np.uint8).tobytes())
                states, goals, distances = states.to(device), goals.to(device), distances.to(device)
                lr = args.lr * min(1., step / max(1, args.warmup))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                prediction = model(states, goals)
                loss = F.smooth_l1_loss(prediction, distances) if args.model == "joint" else F.cross_entropy(prediction, distances.long())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite training loss at step {step}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.gradclip, error_if_nonfinite=True)
                optimizer.step()
                if step == 1 or step % 1000 == 0:
                    print("TRAIN " + json.dumps(dict(step=step, loss=float(loss.detach()), seconds=time.time()-start)), flush=True)
                if args.evalevery and step % args.evalevery == 0:
                    metrics, _ = evaluate(model, banks["validation"], args.model, device, args.eval_bs)
                    event = dict(step=step, bank="validation", metrics=metrics, loss=float(loss.detach()), lr=lr)
                    validation_log.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                    validation_log.flush()
                    print("VALIDATION " + json.dumps(event, sort_keys=True, allow_nan=False), flush=True)
    checkpoint = dict(format_version=1, model=args.model, config=config, training=training,
        state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
        banks=provenance, source_sha256=sources, sample_trace_sha256=trace.hexdigest(),
        versions=dict(torch=str(torch.__version__), numpy=str(np.__version__)))
    if not args.eval_only:
        # Freeze the final checkpoint BEFORE looking at any test prediction.
        torch.save(checkpoint, outputs["checkpoint"])
    final_metrics, final_predictions = {}, {}
    for name in ("validation", "test"):
        metrics, predictions = evaluate(model, banks[name], args.model, device, args.eval_bs)
        final_metrics[name] = metrics
        final_predictions[name + "_truth"] = banks[name].dist
        for estimator, values in predictions.items():
            final_predictions[name + "_" + estimator] = values
    result = dict(model=args.model, config=config, training=training, nskip=0,
        params=sum(parameter.numel() for parameter in model.parameters()),
        banks=provenance, source_sha256=sources, sample_trace_sha256=trace.hexdigest(),
        final_checkpoint=str(args.checkpoint.resolve() if args.eval_only else outputs["checkpoint"]),
        checkpoint_sha256=file_sha256(args.checkpoint if args.eval_only else outputs["checkpoint"]),
        seconds=time.time() - start, **final_metrics)
    np.savez_compressed(outputs["predictions"], **final_predictions,
                        provenance_json=np.array(json.dumps(provenance, sort_keys=True)))
    outputs["result"].write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("RESULT " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
