"""Durable, bounded direct-distance campaign; workers never launch other workers.

Usage: init --root ROOT --hours 24; worker --root ROOT --group dgx --gpu GPU-UUID;
status --root ROOT. All hosts must see the same root via NFS with flock support.
Create ROOT/STOP to stop workers and terminate only their own training children.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time


TERMINAL = {"completed", "failed", "expired", "interrupted"}
TRIAL_TIMEOUT = 8 * 3600
POLL_SECONDS = 30


def atomic_json(path, data):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def state_transaction(root):
    root = Path(root)
    with (root / "state.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((root / "state.json").read_text())
        yield state
        atomic_json(root / "state.json", state)


def initial_specs(G):
    joint = {"--research-model": "joint", "--cnnw": 64, "--cnnk": 3,
             "--rtied": 1, "--rblocks": 1, "--rstride": 1}
    context = {"--research-model": "context", "--d": 128, "--layers": 2,
               "--cnnw": 64, "--cnnk": 3, "--cnndepth": 3}
    return [
        ("coat64", "coat", {"--extonly": "coat", "--extw": 64, "--T": 4}),
        ("joint64T4", "joint", {**joint, "--T": 4}),
        ("joint64T8", "joint", {**joint, "--T": 8}),
        ("joint96T4untied", "joint", {**joint, "--T": 4, "--cnnw": 96, "--rtied": 0}),
        ("joint64T8attn2", "joint", {**joint, "--T": 8, "--rattn": 2}),
        ("context128T4", "context", {**context, "--T": 4}),
        ("context128T8", "context", {**context, "--T": 8}),
        ("pixels128T4", "integ", {"--d": 128, "--layers": 1, "--T": 4,
            "--readout": "pixels", "--cnnk": 3, "--cnnw": 64}),
        ("gcurr", "integ", {"--gcurr": G - 2, "--slots": 16, "--d": 256,
            "--layers": 3, "--T": 4, "--cnnk": 1, "--cnnw": 64,
            "--cnndepth": 2, "--objch": 1, "--readout": "xattn"}),
    ]


def add_task(state, group, model, result_key, flags, phase="discovery", seed=0,
             variant=None, parent=None, now=None, G=None):
    root = Path(state["root"])
    G = state["groups"][group]["G"] if G is None else G
    variant = variant or model
    scale_part = f"-G{G}" if phase == "confirmation" else ""
    identifier = f"{group}-{phase}{scale_part}-{variant}-s{seed}"
    if any(task["id"] == identifier for task in state["tasks"]):
        raise ValueError(f"Duplicate task {identifier}")
    bank = "historical" if phase == "confirmation" else "validation"
    steps = 160000 if phase == "confirmation" else 80000
    base = {"--G": G, "--ngate": 3, "--nlever": 2, "--nchute": 1,
            "--Rmax": 36 if G == 11 else 44, "--bfsmax": 38360 if G == 11 else 72903,
            "--nmaps": 683, "--poolq": 6800, "--steps": steps, "--bs": 128,
            "--warmup": 2000, "--gradclip": 1, "--lr": .001, "--heads": 4,
            "--enc": "pureimage", "--split": "map", "--seed": seed,
            "--evalevery": 20000, "--evalbs": 128, **flags}
    # Bank, steps, and seed belong to the protocol and cannot be overridden by a variant.
    base.update({"--steps": steps, "--seed": seed,
                 "--evalevery": 0 if phase == "confirmation" else 20000})
    prefix = str(root / "artifacts" / identifier)
    base.update({"--save": prefix, "--dumppred": prefix, "--tag": identifier})
    command = [sys.executable, "-u", str(root / "src" / "autoresearch_trial.py"),
               "--pool-cache", str(root / "pools"), "--eval-bank", bank,
               "--torch-threads", "2", "--fast-bfs", "--train", "--nobaseline"]
    for key, value in base.items():
        command.extend((key, str(value)))
    task = dict(id=identifier, group=group, G=G, model=model, variant=variant,
                result_key=result_key, flags=copy.deepcopy(flags), phase=phase,
                seed=seed, steps=steps, eval_bank=bank, status="pending", command=command,
                parent=parent, created_at=time.time() if now is None else now,
                outfile=str(root / "logs" / (identifier + ".log")), result=None,
                metrics=None, start=None, end=None, pid=None, pgid=None, gpu=None, host=None)
    state["tasks"].append(task)
    return task


def new_campaign(root, hours, now=None):
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("hours must be positive and finite")
    now = time.time() if now is None else now
    root = Path(root).resolve()
    hashes = {str(path.relative_to(root / "src")): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((root / "src").glob("*.py"))}
    state = dict(version=1, root=str(root), created_at=now, deadline=now + hours * 3600,
                 trial_timeout=TRIAL_TIMEOUT, source_sha256=hashes,
                 groups={"dgx": {"G": 11, "phase": "discovery"},
                         "a40": {"G": 13, "phase": "discovery"}},
                 tasks=[], workers={}, decisions=[])
    for group, details in state["groups"].items():
        for model, key, flags in initial_specs(details["G"]):
            add_task(state, group, model, key, flags, now=now)
    return state


def valid_metrics(metrics):
    if not isinstance(metrics, dict) or metrics.get("nskip") != 0:
        return False
    for key in ("test_mae", "test_corr"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
    return metrics["test_mae"] >= 0 and -1 <= metrics["test_corr"] <= 1


def parse_result(path, returncode, result_key):
    if returncode != 0:
        raise ValueError(f"process exited with code {returncode}; RESULT is ineligible")
    result = None
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
    if not isinstance(result, dict) or not valid_metrics(result.get(result_key)):
        raise ValueError("missing/invalid final RESULT: finite test_mae/test_corr and nskip=0 required")
    # Reject nonstandard JSON numbers anywhere before durable strict-JSON storage.
    json.dumps(result, allow_nan=False)
    metrics = {key: result[result_key][key] for key in ("test_mae", "test_corr", "nskip")}
    return metrics, result


def comparison(task, anchor):
    if task["status"] != "completed" or not valid_metrics(task.get("metrics")):
        return {"task": task["id"], "eligible": False,
                "reason": task.get("error", task["status"])}
    metrics, baseline = task["metrics"], anchor["metrics"]
    dc = metrics["test_corr"] - baseline["test_corr"]
    dm = metrics["test_mae"] - baseline["test_mae"]
    return dict(task=task["id"], eligible=True, corr_delta=dc, mae_delta=dm,
                mae_ratio=metrics["test_mae"] / baseline["test_mae"] if baseline["test_mae"] else None,
                passes=(metrics["test_corr"] > baseline["test_corr"] + .005
                        or metrics["test_mae"] < baseline["test_mae"] * .97),
                near=(metrics["test_corr"] >= baseline["test_corr"] - .03
                      or metrics["test_mae"] <= baseline["test_mae"] * 1.15))


def advance_group(state, group, now=None):
    """Advance exactly once per completed stage, under the caller's state lock."""
    now = time.time() if now is None else now
    details = state["groups"][group]
    phase = details["phase"]
    if phase == "done" or now >= state["deadline"]:
        return
    tasks = [task for task in state["tasks"] if task["group"] == group]
    stage = [task for task in tasks if task["phase"] == phase]
    if not stage or any(task["status"] not in TERMINAL for task in stage):
        return
    decision = dict(time=now, group=group, phase=phase)
    state["decisions"].append(decision)
    if phase == "confirmation":
        anchors = {(task["G"], task["seed"]): task for task in stage if task["model"] == "coat64"}
        comparisons = []
        for task in stage:
            if task["model"] == "coat64":
                continue
            anchor = anchors[(task["G"], task["seed"])]
            if anchor["status"] == "completed" and valid_metrics(anchor.get("metrics")):
                comparisons.append(comparison(task, anchor))
            else:
                comparisons.append(dict(task=task["id"], eligible=False, reason="confirmation anchor failed"))
        decision.update(reason="confirmation stage complete", comparisons=comparisons)
        details["phase"] = "done"
        return
    anchor = next(task for task in tasks if task["phase"] == "discovery" and task["model"] == "coat64")
    if anchor["status"] != "completed" or not valid_metrics(anchor.get("metrics")):
        decision.update(reason="discovery anchor failed; no selection", error=anchor.get("error"))
        details["phase"] = "done"
        return
    candidates = [task for task in tasks if task["model"] != "coat64"
                  and task["phase"] in ("discovery", "mutation")]
    comparisons = [comparison(task, anchor) for task in candidates]
    decision["comparisons"] = comparisons
    passing_ids = {item["task"] for item in comparisons if item.get("passes")}
    passing = [task for task in candidates if task["id"] in passing_ids]
    if passing:
        best_corr = max(passing, key=lambda task: (task["metrics"]["test_corr"], -task["metrics"]["test_mae"]))
        best_mae = min(passing, key=lambda task: (task["metrics"]["test_mae"], -task["metrics"]["test_corr"]))
        selected = [best_corr] + ([] if best_mae["id"] == best_corr["id"] else [best_mae])
        for source in [anchor, *selected]:
            for G in (11, 13):
                for seed in (0, 1):
                    flags = {**source["flags"]}
                    if "--gcurr" in flags:
                        flags["--gcurr"] = G - 2
                    add_task(state, group, source["model"], source["result_key"], flags,
                             phase="confirmation", seed=seed, variant=source["variant"],
                             parent=source["id"], now=now, G=G)
        details["phase"] = "confirmation"
        decision.update(reason="validation candidates passed final-metric threshold",
                        selected=[task["id"] for task in selected])
        return
    near_ids = {item["task"] for item in comparisons if item.get("near")}
    near = [task for task in candidates if task["id"] in near_ids]
    if phase == "discovery" and near:
        source = max(near, key=lambda task: (task["metrics"]["test_corr"], -task["metrics"]["test_mae"]))
        for suffix, adjustment in (("lr5e4", {"--lr": .0005}), ("cosine", {"--cosine": 1})):
            add_task(state, group, source["model"], source["result_key"], {**source["flags"], **adjustment},
                     phase="mutation", variant=source["variant"] + "_" + suffix, parent=source["id"], now=now)
        details["phase"] = "mutation"
        decision.update(reason="no pass; bounded near-miss mutation round", selected=[source["id"]])
    else:
        details["phase"] = "done"
        decision["reason"] = "no passing candidate after mutation" if phase == "mutation" else "no pass or near miss"


def gpu_idle(uuid):
    """Fail closed on unknown cards, unavailable telemetry, or any observed use."""
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
        raise ValueError("--gpu must be a full physical GPU UUID")
    query = subprocess.run(["nvidia-smi", "--id=" + uuid,
        "--query-gpu=uuid,name,memory.used,utilization.gpu,display_active",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=15)
    rows = list(csv.reader(query.stdout.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("ambiguous GPU telemetry")
    found, name, memory, utilization, display = [value.strip() for value in rows[0]]
    compute_card = re.search(r"\b(A40|A100|A800|H100|H200|H800|V100|L40S?|L4|B100|B200|B300|P100|T4|GH200|GB200|GB300)\b", name)
    if (found != uuid or not compute_card or "1080" in name
            or display.lower() not in ("disabled", "no", "inactive")):
        return False, f"ineligible compute card: {rows[0]}"
    try:
        memory_value, utilization_value = float(memory), float(utilization)
    except ValueError:
        return False, f"unavailable GPU telemetry: {rows[0]}"
    if (not math.isfinite(memory_value) or not 0 <= memory_value <= 20
            or not math.isfinite(utilization_value) or utilization_value != 0):
        return False, f"GPU occupied: memory={memory} MiB utilization={utilization}%"
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=15)
    for row in csv.reader(processes.stdout.strip().splitlines(), skipinitialspace=True):
        if row and row[0].strip() == uuid:
            return False, "GPU has an existing compute process"
    return True, name


def verify_sources(state):
    source = Path(state["root"]) / "src"
    current = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(source.glob("*.py"))}
    if current != state["source_sha256"]:
        raise ValueError("source SHA256 snapshot changed; refusing to run a mixed-source campaign")


def terminate_child(child):
    """Only a Popen child created with start_new_session=True is accepted here."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def worker(root, group, uuid, ready_file=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
        raise ValueError("--gpu must be a physical GPU UUID")
    timeout_program = shutil.which("timeout")
    if timeout_program is None:
        raise RuntimeError("GNU timeout is required to enforce deadlines independently of the worker")
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    host, worker_pid = socket.gethostname(), os.getpid()
    identity = f"{host}:{worker_pid}:{uuid}"
    gpu_lock = (root / ("gpu-" + uuid + ".lock")).open("a+")
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        gpu_lock.close()
        print(json.dumps({"event": "worker_exit", "reason": "GPU already has a campaign worker", "gpu": uuid}), flush=True)
        return 2
    def heartbeat(state, task=None, status="idle"):
        state["workers"][identity] = dict(host=host, pid=worker_pid, gpu=uuid, group=group,
            heartbeat=time.time(), task=task, status=status)
    exit_reason = "complete"
    try:
        while True:
            now = time.time()
            with state_transaction(root) as state:
                heartbeat(state)
                deadline = state["deadline"]
                if now >= deadline:
                    for task in state["tasks"]:
                        if task["status"] == "pending":
                            task.update(status="expired", end=now, error="campaign deadline")
                advance_group(state, group, now)
                finished = state["groups"][group]["phase"] == "done"
            if stop.is_set() or (root / "STOP").exists() or now >= deadline:
                exit_reason = "stop requested" if now < deadline else "campaign deadline"
                break
            if finished:
                break
            if ready_file and not Path(ready_file).exists():
                stop.wait(POLL_SECONDS)
                continue
            idle, reason = gpu_idle(uuid)
            if not idle:
                exit_reason = reason
                break
            selected = None
            with state_transaction(root) as state:
                verify_sources(state)
                abandoned = [task["id"] for task in state["tasks"]
                    if task["status"] == "running" and task.get("gpu") == uuid
                    and task.get("worker") != identity]
                if abandoned:
                    raise ValueError(f"prior worker left unresolved tasks on this GPU: {abandoned}")
                if time.time() < state["deadline"] and not (root / "STOP").exists():
                    selected = next((task for task in state["tasks"]
                        if task["group"] == group and task["status"] == "pending"), None)
                    if selected is not None:
                        selected.update(status="running", start=time.time(), gpu=uuid,
                                        host=host, worker=identity, worker_pid=worker_pid)
                        heartbeat(state, selected["id"], "running")
                        selected = copy.deepcopy(selected)
            if selected is None:
                stop.wait(POLL_SECONDS)
                continue
            child = None
            outcome = {"status": "failed", "error": "trial did not start"}
            try:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "OMP_NUM_THREADS": "2",
                       "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
                with open(selected["outfile"], "x") as log:
                    trial_deadline = min(deadline, selected["start"] + TRIAL_TIMEOUT)
                    timeout_seconds = trial_deadline - time.time()
                    if timeout_seconds <= 0:
                        raise TimeoutError("campaign/trial deadline reached before subprocess launch")
                    launch_command = [timeout_program, "--signal=TERM", "--kill-after=10s",
                                      f"{timeout_seconds:.6f}s", *selected["command"]]
                    # The timeout supervisor inherits the flock descriptor, so an
                    # orphaned training child keeps its GPU reserved until expiry.
                    child = subprocess.Popen(launch_command, cwd=root / "src", env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(gpu_lock.fileno(),))
                    with state_transaction(root) as state:
                        task = next(task for task in state["tasks"] if task["id"] == selected["id"])
                        task.update(pid=child.pid, pgid=child.pid,
                                    launch_command=launch_command, timeout_seconds=timeout_seconds,
                                    environment={key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
                    interrupted = None
                    while child.poll() is None:
                        if stop.is_set() or (root / "STOP").exists():
                            interrupted = "stop requested"
                        elif time.time() >= trial_deadline:
                            interrupted = "campaign deadline" if time.time() >= deadline else "8-hour trial timeout"
                        if interrupted:
                            terminate_child(child)
                            break
                        with state_transaction(root) as state:
                            heartbeat(state, selected["id"], "running")
                        stop.wait(min(POLL_SECONDS, max(.01, trial_deadline - time.time())))
                if interrupted:
                    outcome = dict(status="interrupted", error=interrupted, returncode=child.returncode)
                else:
                    metrics, result = parse_result(selected["outfile"], child.returncode, selected["result_key"])
                    outcome = dict(status="completed", metrics=metrics, result=result, returncode=child.returncode)
            except Exception as exc:
                if child is not None:
                    terminate_child(child)
                outcome = dict(status="failed", error=f"{type(exc).__name__}: {exc}",
                               returncode=child.returncode if child else None)
            finally:
                with state_transaction(root) as state:
                    task = next(task for task in state["tasks"] if task["id"] == selected["id"])
                    task.update(outcome, end=time.time())
                    heartbeat(state)
                    advance_group(state, group)
                print(json.dumps({"event": "trial_end", "task": selected["id"], **outcome}), flush=True)
    except Exception as exc:
        exit_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with state_transaction(root) as state:
            heartbeat(state, status="exited")
            state["workers"][identity]["reason"] = exit_reason
        gpu_lock.close()
        print(json.dumps({"event": "worker_exit", "worker": identity, "reason": exit_reason}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--hours", type=float, default=24)
    work = subparsers.add_parser("worker")
    work.add_argument("--root", type=Path, required=True)
    work.add_argument("--group", choices=("dgx", "a40"), required=True)
    work.add_argument("--gpu", required=True)
    work.add_argument("--ready-file", type=Path)
    status = subparsers.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "init":
        args.root.mkdir(parents=True, exist_ok=True)
        for directory in ("logs", "artifacts", "pools", "src"):
            (args.root / directory).mkdir(exist_ok=True)
        with (args.root / "state.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (args.root / "state.json").exists():
                parser.error("state.json already exists; refusing to overwrite campaign")
            state = new_campaign(args.root, args.hours)
            atomic_json(args.root / "state.json", state)
        print(json.dumps({"root": str(args.root.resolve()), "tasks": len(state["tasks"]), "deadline": state["deadline"]}), flush=True)
    elif args.action == "worker":
        return worker(args.root, args.group, args.gpu, args.ready_file)
    else:
        with (args.root / "state.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            state = json.loads((args.root / "state.json").read_text())
        state["counts"] = dict(Counter(task["status"] for task in state["tasks"]))
        state["seconds_remaining"] = max(0, state["deadline"] - time.time())
        print(json.dumps(state, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
