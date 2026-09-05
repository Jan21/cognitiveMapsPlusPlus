"""Frozen-bank direct-distance transfer for reconstructed CRTR extra tasks.

LightsOut: 7x7 binary tiles. DigitJumper: 20x20 categorical vocabulary 0..12,
with digits 1..6 and the agent's tile marked by +6. Joint models one-hot each
endpoint without coordinates; their ONLY distance output is accumulated latent
motion. Original CRTR baselines receive concatenated flat numeric tile arrays.

Banks contain matched uint8 states/goals (N,H,W), optionally (N,1,H,W), exact
float32 dist (N,), and optionally gap (N,). --targets gap selects gap for training
and validation only. Final test always uses exact dist. No data augmentation.

Public solver API: model, metadata = load_extra_checkpoint(path, device='cpu',
crtr_root=None). All wrappers accept flat Bx(HW), BHW or B1HW matched tensors.
Joint returns B distances; supervised/cnn/dense return Bx50 or Bx400 CE logits.
LightsOut supervised is the main CNN control; dense is an extra original 8x512
LNDenseNet control on concatenated numeric endpoint tiles.
The latter require the exact CRTR networks.py source at load time.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .autoresearch_joint import JointPixelInteg
    from .sokoban_joint import distance_metrics, file_sha256
else:
    from autoresearch_joint import JointPixelInteg
    from sokoban_joint import distance_metrics, file_sha256


ENVIRONMENTS = {"lights": {"size": 7, "classes": 2, "bins": 50, "lr": 1e-4},
                "digit": {"size": 20, "classes": 13, "bins": 400, "lr": 3e-4}}


def environment(env):
    if env not in ENVIRONMENTS:
        raise ValueError("env must be lights or digit")
    return ENVIRONMENTS[env]


def flat_tiles(value, env):
    size = environment(env)["size"]
    if ((value.ndim == 2 and value.shape[1:] == (size * size,))
            or (value.ndim == 3 and value.shape[1:] == (size, size))
            or (value.ndim == 4 and value.shape[1:] == (1, size, size))):
        return value.reshape(value.shape[0], size * size)
    raise ValueError(f"{env} inputs must be Bx{size * size}, Bx{size}x{size} or Bx1x{size}x{size}")


def onehot_grid(value, env):
    spec = environment(env)
    flat = flat_tiles(value, env)
    return F.one_hot(flat.long(), spec["classes"]).reshape(
        -1, spec["size"], spec["size"], spec["classes"]).permute(0, 3, 1, 2).float()


class ExtraJoint(nn.Module):
    def __init__(self, env, width=64, T=8, attention_heads=2):
        super().__init__()
        self.env = env
        self.core = JointPixelInteg(in_channels=environment(env)["classes"], width=width,
            T=T, attention_heads=attention_heads, kernel_size=3, tied=True,
            blocks=1, stride=1, reinject=True)

    def forward(self, states, goals, Trun=None, ret_states=False):
        return self.core(onehot_grid(states, self.env), onehot_grid(goals, self.env),
                         Trun=Trun, ret_states=ret_states)


def import_crtr_networks(crtr_root):
    if crtr_root is None:
        raise ValueError("--crtr-root is required for supervised/cnn/dense models")
    path = Path(crtr_root).resolve() / "networks.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    # A unique exact-file name avoids .pth-injected or preloaded networks modules.
    name = "_crtr_extra_networks_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError("conflicting exact CRTR networks module")
        return module, path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, path


def supervised_config(env, kind):
    spec = environment(env)
    if kind == "cnn" and env != "digit":
        raise ValueError("the optional cnn control is DigitJumper-only")
    if kind == "dense" and env != "lights":
        raise ValueError("the optional dense control is LightsOut-only")
    if kind not in ("supervised", "cnn", "dense"):
        raise ValueError("expected supervised, cnn or dense")
    if kind == "dense" or (env == "digit" and kind == "supervised"):
        return dict(network_class="LNDenseNet", input_size=2 * spec["size"] ** 2,
                    repr_dim=spec["bins"], hidden_size=512, depth=8)
    return dict(network_class="LNConvNet", input_size=2 * spec["classes"],
                repr_dim=spec["bins"], hidden_size=64, depth=8, baseline=True)


class ExtraSupervised(nn.Module):
    def __init__(self, env, kind, crtr_root):
        super().__init__()
        self.env, self.kind = env, kind
        networks, self.source_path = import_crtr_networks(crtr_root)
        config = supervised_config(env, kind)
        self.network = getattr(networks, config["network_class"])(
            **{key: value for key, value in config.items() if key != "network_class"})

    def forward(self, states, goals):
        states, goals = flat_tiles(states, self.env), flat_tiles(goals, self.env)
        if states.shape != goals.shape:
            raise ValueError("state/goal shapes must match")
        return self.network(torch.cat((states, goals), dim=1).float())


class FrozenExtraBank:
    def __init__(self, path, env, targets="exact"):
        spec = environment(env)
        if targets not in ("exact", "gap"):
            raise ValueError("targets must be exact or gap")
        self.path = Path(path).resolve()
        metadata = {}
        with np.load(self.path, allow_pickle=False) as archive:
            self.states, self.goals, self.dist = archive["states"], archive["goals"], archive["dist"]
            if targets == "gap" and "gap" not in archive:
                raise ValueError("gap targets require a gap member in the frozen bank")
            self.targets = archive["gap"] if targets == "gap" else self.dist
            for name in ("metadata_json", "provenance_json"):
                if name in archive and archive[name].ndim == 0 and archive[name].dtype.kind in "US":
                    text = archive[name].item()
                    try:
                        metadata[name] = json.loads(text)
                    except (ValueError, TypeError):
                        metadata[name] = str(text)
        if self.states.ndim == 4 and self.states.shape[1] == 1:
            self.states = self.states[:, 0]
        if self.goals.ndim == 4 and self.goals.shape[1] == 1:
            self.goals = self.goals[:, 0]
        shape = (spec["size"], spec["size"])
        if (self.states.dtype != np.uint8 or self.goals.dtype != np.uint8
                or self.states.ndim != 3 or self.states.shape[1:] != shape
                or self.goals.shape != self.states.shape or len(self.states) == 0):
            raise ValueError(f"bank states/goals must be matched nonempty uint8 (N,{shape[0]},{shape[1]})")
        for name, values in (("dist", self.dist), (targets, self.targets)):
            if (values.dtype != np.float32 or values.shape != (len(self.states),)
                    or not np.isfinite(values).all() or np.any(values < 0)):
                raise ValueError(f"bank {name} must be finite nonnegative float32 (N,)")
        for start in range(0, len(self.states), 8192):
            if (np.any(self.states[start:start + 8192] >= spec["classes"])
                    or np.any(self.goals[start:start + 8192] >= spec["classes"])):
                raise ValueError(f"{env} tile IDs must be in 0..{spec['classes'] - 1}")
        self.provenance = dict(path=str(self.path), sha256=file_sha256(self.path),
            rows=len(self.states), env=env, target_member="dist" if targets == "exact" else "gap",
            distance_min=float(self.dist.min()), distance_max=float(self.dist.max()), metadata=metadata)

    def __len__(self):
        return len(self.states)


def draw_batch(bank, rng, batch_size):
    indices = rng.integers(0, len(bank), size=batch_size)
    return (torch.from_numpy(bank.states[indices]), torch.from_numpy(bank.goals[indices]),
            torch.from_numpy(bank.targets[indices]), indices)


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
                    expectation = (output.softmax(1) * torch.arange(output.shape[1], device=device)).sum(1)
                    predictions["expectation"].append(expectation.cpu().numpy())
    finally:
        model.train(was_training)
    arrays = {name: np.concatenate(parts).astype(np.float32) for name, parts in predictions.items()}
    return {name: distance_metrics(bank.targets, values) for name, values in arrays.items()}, arrays


def load_extra_checkpoint(path, device="cpu", crtr_root=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("format") != "crtr_extra"
            or checkpoint.get("model") not in ("joint", "supervised", "cnn", "dense")):
        raise ValueError("expected a crtr_extra checkpoint")
    kind, env = checkpoint["model"], checkpoint["env"]
    if kind == "joint":
        model = ExtraJoint(env, **checkpoint["config"])
    else:
        root = crtr_root or checkpoint["training"].get("crtr_root")
        model = ExtraSupervised(env, kind, root)
        if checkpoint["config"] != supervised_config(env, kind):
            raise ValueError("checkpoint architecture differs from exact supervised configuration")
        expected = checkpoint.get("source_sha256", {}).get("networks.py")
        if expected is not None and file_sha256(model.source_path) != expected:
            raise ValueError("CRTR networks.py source differs from checkpoint")
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("lights", "digit"), required=True)
    parser.add_argument("--model", choices=("joint", "supervised", "cnn", "dense"), default="joint")
    parser.add_argument("--targets", choices=("exact", "gap"), default="exact")
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--val-bank", type=Path, required=True)
    parser.add_argument("--test-bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="artifact path prefix")
    parser.add_argument("--crtr-root", type=Path)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--eval-bs", type=int, default=128)
    parser.add_argument("--evalevery", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gradclip", type=float, default=1.)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.lr is None:
        args.lr = environment(args.env)["lr"]
    if min(args.steps, args.bs, args.eval_bs, args.torch_threads) < 1 or args.evalevery < 0:
        parser.error("steps/batch sizes/threads must be positive; evalevery must be nonnegative")
    if not all(math.isfinite(value) and value > 0 for value in (args.lr, args.gradclip)):
        parser.error("lr and gradclip must be finite and positive")
    paths = [path.resolve() for path in (args.train_bank, args.val_bank, args.test_bank)]
    if len(set(paths)) != 3:
        parser.error("train/validation/test banks must use distinct paths")
    prefix = args.out.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {name: Path(str(prefix) + suffix) for name, suffix in (
        ("checkpoint", ".pt"), ("result", ".json"), ("predictions", ".predictions.npz"),
        ("validation_log", ".validation.jsonl"))}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)  # Model initialization cannot consume this RNG.
    device = torch.device(args.device)
    banks = {"train": FrozenExtraBank(args.train_bank, args.env, args.targets),
             "validation": FrozenExtraBank(args.val_bank, args.env, args.targets),
             "test": FrozenExtraBank(args.test_bank, args.env, "exact")}
    if args.model != "joint":
        for bank in banks.values():
            if (np.any(bank.targets != np.floor(bank.targets))
                    or np.any(bank.targets >= environment(args.env)["bins"])):
                raise ValueError("supervised CE targets must be integer distances inside the configured bins")
    if args.model == "joint":
        config = dict(width=args.width, T=args.T, attention_heads=args.attention_heads)
        model = ExtraJoint(args.env, **config).to(device)
    else:
        config = supervised_config(args.env, args.model)
        model = ExtraSupervised(args.env, args.model, args.crtr_root).to(device)
    training = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    training["augmentation"] = "none"
    sources = {name: file_sha256(Path(__file__).with_name(name)) for name in
               ("crtr_extra_train.py", "autoresearch_joint.py", "sokoban_joint.py")}
    if args.model != "joint":
        sources["networks.py"] = file_sha256(model.source_path)
    provenance = {name: bank.provenance for name, bank in banks.items()}
    print("CONFIG " + json.dumps(dict(env=args.env, model=args.model, config=config,
                                     training=training, banks=provenance), sort_keys=True), flush=True)
    trace = hashlib.sha256()
    start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    with outputs["validation_log"].open("x") as log:
        for step in range(1, args.steps + 1):
            states, goals, distances, indices = draw_batch(banks["train"], rng, args.bs)
            trace.update(indices.astype("<i8").tobytes())
            states, goals, distances = states.to(device), goals.to(device), distances.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(states, goals)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite training output at step {step}")
            loss = (F.smooth_l1_loss(prediction, distances) if args.model == "joint"
                    else F.cross_entropy(prediction, distances.long()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradclip, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 1000 == 0:
                print("TRAIN " + json.dumps(dict(step=step, loss=float(loss.detach()),
                                                seconds=time.time() - start)), flush=True)
            if args.evalevery and step % args.evalevery == 0:
                metrics, _ = evaluate(model, banks["validation"], args.model, device, args.eval_bs)
                event = dict(step=step, bank="validation", metrics=metrics, loss=float(loss.detach()), lr=args.lr)
                log.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                log.flush()
                print("VALIDATION " + json.dumps(event, sort_keys=True, allow_nan=False), flush=True)
    state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state_dict.values()):
        raise FloatingPointError("nonfinite final checkpoint state")
    checkpoint = dict(format="crtr_extra", format_version=1, env=args.env, model=args.model,
        config=config, training=training, state_dict=state_dict, banks=provenance,
        source_sha256=sources, sample_trace_sha256=trace.hexdigest(),
        versions=dict(torch=str(torch.__version__), numpy=str(np.__version__)))
    # Freeze final weights before any test predictions; never select by test.
    torch.save(checkpoint, outputs["checkpoint"])
    metrics, predictions = {}, {}
    for name in ("validation", "test"):
        metrics[name], values = evaluate(model, banks[name], args.model, device, args.eval_bs)
        predictions[name + "_truth"] = banks[name].targets
        for estimator, array in values.items():
            predictions[name + "_" + estimator] = array
    result = dict(env=args.env, model=args.model, config=config, training=training, nskip=0,
        params=sum(parameter.numel() for parameter in model.parameters()), banks=provenance,
        source_sha256=sources, sample_trace_sha256=trace.hexdigest(),
        final_checkpoint=str(outputs["checkpoint"]), checkpoint_sha256=file_sha256(outputs["checkpoint"]),
        seconds=time.time() - start, **metrics)
    np.savez_compressed(outputs["predictions"], **predictions,
                        provenance_json=np.array(json.dumps(provenance, sort_keys=True)))
    with outputs["result"].open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("RESULT " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()

