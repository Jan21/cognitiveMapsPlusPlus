"""Prepare exact shared label pools using four CPU-only subprocesses on the DGX."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    specs = [(11, 0, "validation", 9), (13, 0, "validation", 11),
             (11, 0, "historical", 0), (13, 0, "historical", 0),
             (11, 1, "historical", 0), (13, 1, "historical", 0)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}

    def prepare(spec):
        grid, seed, bank, curriculum = spec
        label = f"prepare_G{grid}_s{seed}_{bank}"
        command = [sys.executable, str(root / "src/autoresearch_trial.py"),
                   "--pool-cache", str(root / "pools"), "--eval-bank", bank,
                   "--prepare", "--fast-bfs", "--train", "--G", str(grid),
                   "--ngate", "3", "--nlever", "2", "--nchute", "1", "--nmaps", "683",
                   "--poolq", "6800", "--Rmax", str(36 if grid == 11 else 44),
                   "--bfsmax", str(38360 if grid == 11 else 72903), "--split", "map",
                   "--seed", str(seed), "--gcurr", str(curriculum)]
        start = time.time()
        with (root / "logs" / (label + ".log")).open("x") as log:
            print(json.dumps({"event": "PREP_START", "name": label, "command": command}), flush=True)
            result = subprocess.run(command, cwd=root / "src", env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=7200)
        record = {"event": "PREP_END", "name": label, "exit_code": result.returncode,
                  "seconds": round(time.time() - start, 1)}
        print(json.dumps(record), flush=True)
        if result.returncode == 0:
            (root / (label + ".done")).write_text(json.dumps(record) + "\n")
        return result.returncode

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(prepare, specs))
    if any(results):
        raise SystemExit("One or more pool preparations failed; inspect individual logs")
    print("ALL_POOLS_PREPARED", flush=True)


if __name__ == "__main__":
    main()
