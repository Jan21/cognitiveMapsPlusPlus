"""Start one persistent worker on each currently idle DGX V100; no busy GPUs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from autoresearch_loop import gpu_idle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    cards = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name",
                                    "--format=csv,noheader"], text=True)
    launched = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    for uuid, name in csv.reader(cards.strip().splitlines(), skipinitialspace=True):
        uuid, name = uuid.strip(), name.strip()
        if "V100" not in name or "Display" in name:
            continue
        idle, reason = gpu_idle(uuid)
        if not idle:
            print(json.dumps({"skipped_gpu": uuid, "reason": reason}), flush=True)
            continue
        command = [sys.executable, "-u", str(root / "src/autoresearch_loop.py"),
                   "worker", "--root", str(root), "--group", "dgx", "--gpu", uuid,
                   "--ready-file", str(root / "prepare_G11_s0_validation.done")]
        logfile = root / "logs" / ("dgx_worker_" + uuid + ".log")
        with logfile.open("x") as log:
            child = subprocess.Popen(command, cwd=root / "src", env=env,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        launched.append({"gpu": uuid, "pid": child.pid, "log": str(logfile),
                         "command": command})
    with (root / "dgx_launch.json").open("x") as stream:
        json.dump(launched, stream, indent=2)
    print(json.dumps({"launched": launched}), flush=True)


if __name__ == "__main__":
    main()
