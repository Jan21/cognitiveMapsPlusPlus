"""Run Switchyard trials with shared CPU pools and an independent validation bank.

Preparation does not import torch or allocate a GPU::

    python3 autoresearch_trial.py --pool-cache /shared/sw-pools --prepare \
        --eval-bank validation --train --G 11 --ngate 3 --nlever 2 --nchute 1 \
        --nmaps 683 --poolq 6800 --Rmax 36 --bfsmax 38360 --split map --seed 0

Remove --prepare and append normal Switchyard training/model flags to train.
Cache files are pickle artifacts produced by this program: use a trusted directory.
"""

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np

try:
    from . import switchyard as sw
except ImportError:
    import switchyard as sw


VALIDATION_SEED_OFFSET = 1_000_000
CACHE_VERSION = 1
_MAKE_YARDS = sw.make_yards


def progress(event, **fields):
    print(event + " " + json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                    **fields}, sort_keys=True), flush=True)


def _canonical(value):
    """Canonical, typed JSON representation including every map attribute."""
    if isinstance(value, np.ndarray):
        return {"array_dtype": value.dtype.str, "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, dict):
        pairs = [[_canonical(key), _canonical(item)] for key, item in value.items()]
        return {"dict": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, tuple):
        return {"tuple": [_canonical(item) for item in value]}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((_canonical(item) for item in value),
                              key=lambda item: json.dumps(item, sort_keys=True))}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported pool fingerprint value: {type(value).__name__}")


class PoolCache:
    """Cache the exact pool and generator post-state under an NFS-compatible lock."""

    def __init__(self, directory, build_fn):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.build_fn = build_fn
        source = inspect.getsource(build_fn) + inspect.getsource(sw.Yard)
        self.source_hash = hashlib.sha256(source.encode()).hexdigest()
        self.builds = self.hits = 0

    def key(self, a, rng, yards, mapids, Rcap, cot=False):
        # Only selected maps affect a pool. This deliberately shares training pools
        # between historical/validation runs despite their different held-out maps.
        identity = dict(version=CACHE_VERSION, source=self.source_hash, numpy=np.__version__,
            sampling=dict(poolq=a.poolq, bfsmax=getattr(a, "bfsmax", 200000),
                          Rcap=Rcap, cot=bool(cot),
                          ncot=(getattr(a, "ncot", 0) or a.T) if cot else 0),
            rng=rng.bit_generator.state, mapids=list(mapids),
            maps=[vars(yards[int(index)]) for index in dict.fromkeys(mapids)])
        data = json.dumps(_canonical(identity), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode()).hexdigest()

    def __call__(self, a, rng, yards, mapids, Rcap, cot=False):
        key = self.key(a, rng, yards, mapids, Rcap, cot=cot)
        path = self.directory / (key + ".pkl")
        with (self.directory / (key + ".lock")).open("a+b") as lock:
            progress("CACHE_WAIT", key=key, path=str(path))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                with path.open("rb") as stream:
                    entry = pickle.load(stream)
                if entry["key"] != key:
                    raise ValueError(f"Pool cache key mismatch in {path}")
                rng.bit_generator.state = entry["rng_state"]
                self.hits += 1
                progress("CACHE_HIT", key=key, pairs=len(entry["pool"][0]))
                return entry["pool"]
            progress("CACHE_BUILD", key=key, maps=len(mapids), queries=a.poolq,
                     Rcap=Rcap, cot=bool(cot))
            pool = self.build_fn(a, rng, yards, mapids, Rcap, cot=cot)
            entry = dict(key=key, pool=pool, rng_state=copy.deepcopy(rng.bit_generator.state))
            descriptor, temporary = tempfile.mkstemp(prefix=key + ".", suffix=".tmp",
                                                     dir=self.directory)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    pickle.dump(entry, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.builds += 1
            progress("CACHE_SAVED", key=key, pairs=len(pool[0]), path=str(path))
            return pool


def make_bank_yards(a, eval_bank="validation", wire1=None, noplate=None,
                    make_fn=_MAKE_YARDS):
    """Keep original training maps and replace held-out indices for validation."""
    yards, train_ids, eval_ids = make_fn(a, wire1=wire1, noplate=noplate)
    if eval_bank == "historical":
        return yards, train_ids, eval_ids
    if eval_bank != "validation":
        raise ValueError(f"Unknown evaluation bank: {eval_bank}")
    fresh_args = argparse.Namespace(**{**vars(a), "seed": a.seed + VALIDATION_SEED_OFFSET})
    fresh_yards, _, _ = make_fn(fresh_args, wire1=wire1, noplate=noplate)
    for index in eval_ids:
        yards[index] = fresh_yards[index]
    return yards, train_ids, eval_ids


def prepare_pools(a):
    """Mirror the trainer's pool construction using CPU-only imports."""
    yards, train_ids, eval_ids = sw.make_yards(a)
    small_ids = []
    if a.gcurr:
        if a.gcurr >= a.G or (a.G - a.gcurr) % 2 or a.lencurr or a.curriculum:
            raise ValueError("gcurr requires a smaller grid with even margin and no other curriculum")
        small, small_train, _ = sw.make_yards(argparse.Namespace(**{**vars(a), "G": a.gcurr}))
        offset = len(yards)
        yards += [sw.pad_yard(small[index], a.G) for index in small_train]
        small_ids = list(range(offset, len(yards)))
    cap = a.Rtrain or a.Rmax
    if a.cotsup > 0 and (a.lencurr or a.curriculum):
        raise ValueError("cotsup is incompatible with curricula")
    sw.build_pool(a, np.random.default_rng(a.seed), yards, train_ids, cap, cot=a.cotsup > 0)
    if small_ids:
        sw.build_pool(a, np.random.default_rng(a.seed + 77), yards, small_ids, cap)
    if a.lencurr and a.Rtrain > 8:
        caps = sorted({8, min(12, a.Rtrain), a.Rtrain})
        for index, phase_cap in enumerate(caps[:-1]):
            sw.build_pool(a, np.random.default_rng(a.seed + 40 + index), yards, train_ids, phase_cap)
    if a.curriculum:
        easy, _, _ = sw.make_yards(a, wire1=True, noplate=True)
        middle, _, _ = sw.make_yards(a, wire1=False, noplate=True)
        sw.build_pool(a, np.random.default_rng(a.seed + 7), easy, train_ids, cap)
        sw.build_pool(a, np.random.default_rng(a.seed + 8), middle, train_ids, cap)
    sw.build_pool(a, np.random.default_rng(a.seed + 99), yards, eval_ids, a.Rmax)
    progress("PREPARE_DONE", G=a.G, seed=a.seed, eval_bank=a.eval_bank)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--pool-cache", required=True)
    parser.add_argument("--eval-bank", choices=("validation", "historical"), default="validation")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--fast-bfs", action="store_true",
                        help="use the verified exact Numba BFS (requires numba)")
    parser.add_argument("--torch-threads", type=int, default=2)
    options, remaining = parser.parse_known_args(argv)
    if options.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if options.prepare and "--train" not in remaining:
        remaining.append("--train")
    if not options.prepare:
        import torch
        torch.set_num_threads(options.torch_threads)
    original_build, original_make, original_train = sw.build_pool, sw.make_yards, sw.train
    original_bfs = sw.Yard.bfs
    original_argv = sys.argv
    cache = PoolCache(options.pool_cache, original_build)
    if options.fast_bfs:
        try:
            from .autoresearch_bfs import bfs as compiled_bfs
        except ImportError:
            from autoresearch_bfs import bfs as compiled_bfs
        sw.Yard.bfs = compiled_bfs
    sw.build_pool = cache
    sw.make_yards = lambda a, wire1=None, noplate=None: make_bank_yards(
        a, options.eval_bank, wire1=wire1, noplate=noplate, make_fn=original_make)

    def run(a):
        a.eval_bank = options.eval_bank
        a.pool_cache = str(cache.directory.resolve())
        a.torch_threads = options.torch_threads
        a.fast_bfs = options.fast_bfs
        progress("TRIAL_CONFIG", eval_bank=options.eval_bank, prepare=options.prepare,
                 pool_cache=a.pool_cache, torch_threads=options.torch_threads,
                 validation_seed_offset=VALIDATION_SEED_OFFSET, args=vars(a))
        return prepare_pools(a) if options.prepare else original_train(a)

    sw.train = run
    try:
        sys.argv = [original_argv[0], *remaining]
        sw.main()
    finally:
        sys.argv = original_argv
        sw.build_pool, sw.make_yards, sw.train = original_build, original_make, original_train
        sw.Yard.bfs = original_bfs


if __name__ == "__main__":
    main()
