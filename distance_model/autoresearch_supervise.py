"""Bounded replacement supervisor for the frozen direct-distance campaign.

Deploy OUTSIDE ROOT/src (for example ROOT/autoresearch_supervise.py), so the
frozen worker's source hashes remain valid. Campaign state is strictly read-only.
No process is killed; all training runs through the original worker and timeout.
"""

import argparse
import fcntl
import getpass
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time


POLL_SECONDS = 30
DGX_RESTART_LIMIT = 64
SLURM_SUBMISSION_LIMIT = 16
SLURM_COOLDOWN = 120


def read_state(root):
    with (root / "state.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        return json.loads((root / "state.json").read_text())


def unresolved_gpus(state):
    return {task.get("gpu") for task in state["tasks"]
            if task["status"] == "running" and task.get("gpu")}


def replacement_slots(pending, active):
    return max(0, min(pending, 8 - active))


def active_slurm_slots(output, tracked_jobs):
    """--array is mandatory: reject compact ranges instead of undercounting."""
    active = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) != 3:
            raise ValueError(f"malformed squeue row: {line!r}")
        job_id, name, status = (field.strip() for field in fields)
        if job_id.split("_", 1)[0] not in tracked_jobs:
            continue
        if not re.fullmatch(r"\d+(?:_\d+)?", job_id):
            raise ValueError(f"squeue did not expand tracked array: {job_id}")
        if status not in ("CD", "CA", "F", "TO", "NF", "OOM"):
            active.add(job_id)
    return len(active)


def replay_budget(events):
    budget = dict(dgx_restarts=0, last_gpu_launch={}, slurm_submissions=0,
                  last_slurm_submit=0., submitted_jobs=set(), slurm_pending_submission=False)
    for event in events:
        if event["event"] == "dgx_launch_attempt":
            budget["dgx_restarts"] += 1
            budget["last_gpu_launch"][event["gpu"]] = event["time"]
        elif event["event"] == "slurm_submit_attempt":
            budget["slurm_submissions"] += 1
            budget["last_slurm_submit"] = event["time"]
            budget["slurm_pending_submission"] = True
        elif event["event"] == "slurm_submitted":
            budget["submitted_jobs"].add(event["job_id"])
            budget["slurm_pending_submission"] = False
    return budget


def load_original_worker(root):
    spec = importlib.util.spec_from_file_location("frozen_autoresearch_loop", root / "src" / "autoresearch_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dgx_allowlist(root):
    entries = json.loads((root / "dgx_launch.json").read_text())
    if not isinstance(entries, list) or len(entries) != 8:
        raise ValueError("dgx_launch.json must identify the original eight eligible GPUs")
    cards = {}
    for entry in entries:
        uuid = entry["gpu"]
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid) or uuid in cards:
            raise ValueError("invalid or duplicate approved GPU UUID")
        cards[uuid] = entry["command"][0]
    return cards


def supervise(root, site, slurm_jobs):
    root = root.resolve()
    if Path(__file__).resolve().parent == root / "src":
        raise ValueError("deploy this supervisor outside the frozen ROOT/src directory")
    group = "dgx" if site == "dgx" else "a40"
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    supervisor_lock = (root / f"supervisor-{site}.lock").open("a+")
    try:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        supervisor_lock.close()
        raise RuntimeError(f"a {site} supervisor is already running")
    log_path = root / "logs" / f"supervisor-{site}.jsonl"
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()] if log_path.exists() else []
    budget = replay_budget(events)
    tracked_jobs = set(slurm_jobs) | budget["submitted_jobs"]
    children = []
    with log_path.open("a", buffering=1) as log:
        def record(event, **fields):
            item = dict(event=event, time=time.time(), site=site, host=socket.gethostname(), **fields)
            log.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            print(json.dumps(item, sort_keys=True), flush=True)

        try:
            original = load_original_worker(root)
            cards = dgx_allowlist(root) if site == "dgx" else {}
            record("supervisor_start", pid=os.getpid(), tracked_jobs=sorted(tracked_jobs),
                   dgx_restart_count=budget["dgx_restarts"], slurm_submission_count=budget["slurm_submissions"])
            if budget["slurm_pending_submission"]:
                record("ambiguous_submission", reason="prior submit attempt has no recorded job ID; further submissions blocked")
            while not stop.is_set():
                # Reap only our already-exited workers; never signal any process.
                children = [child for child in children if child.poll() is None]
                try:
                    state = read_state(root)
                    now = time.time()
                    if (root / "STOP").exists() or now >= state["deadline"] or state["groups"][group]["phase"] == "done":
                        record("supervisor_done", reason="STOP, deadline, or completed group")
                        break
                    original.verify_sources(state)
                    pending = sum(task["group"] == group and task["status"] == "pending" for task in state["tasks"])
                    record("heartbeat", pending=pending, phase=state["groups"][group]["phase"],
                           seconds_remaining=state["deadline"] - now)
                    if pending and site == "dgx" and budget["dgx_restarts"] < DGX_RESTART_LIMIT:
                        launches_remaining = pending
                        for uuid, python in cards.items():
                            if stop.is_set() or budget["dgx_restarts"] >= DGX_RESTART_LIMIT or launches_remaining <= 0:
                                break
                            current = read_state(root)
                            if (root / "STOP").exists() or time.time() >= current["deadline"]:
                                break
                            if not any(task["group"] == group and task["status"] == "pending" for task in current["tasks"]):
                                break
                            if uuid in unresolved_gpus(current) or time.time() - budget["last_gpu_launch"].get(uuid, 0) < POLL_SECONDS:
                                continue
                            with (root / f"gpu-{uuid}.lock").open("a+") as lock:
                                try:
                                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BlockingIOError:
                                    continue
                                idle, reason = original.gpu_idle(uuid)
                                if not idle or "V100" not in reason:
                                    record("gpu_deferred", gpu=uuid, reason=reason)
                                    continue
                            # Release our probe lock before the frozen worker takes
                            # its own lock and independently rechecks telemetry.
                            command = [python, "-u", str(root / "src" / "autoresearch_loop.py"),
                                       "worker", "--root", str(root), "--group", "dgx", "--gpu", uuid,
                                       "--ready-file", str(root / "prepare_G11_s0_validation.done")]
                            budget["dgx_restarts"] += 1
                            budget["last_gpu_launch"][uuid] = time.time()
                            worker_log = root / "logs" / f"dgx_auto_{budget['dgx_restarts']:03d}_{time.time_ns()}_{uuid}.log"
                            record("dgx_launch_attempt", gpu=uuid, command=command, log=str(worker_log))
                            env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                   "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
                            with worker_log.open("x") as stream:
                                child = subprocess.Popen(command, cwd=root / "src", env=env,
                                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
                            children.append(child)
                            launches_remaining -= 1
                            record("dgx_launched", gpu=uuid, pid=child.pid, command=command, log=str(worker_log))
                    elif (pending and site == "slurm" and budget["slurm_submissions"] < SLURM_SUBMISSION_LIMIT
                          and not budget["slurm_pending_submission"]):
                        query = ["squeue", "--array", "--noheader", "--user", getpass.getuser(), "--format=%i|%j|%t"]
                        queue = subprocess.run(query, check=True, capture_output=True, text=True, timeout=20)
                        active = active_slurm_slots(queue.stdout, tracked_jobs)
                        needed = replacement_slots(pending, active)
                        if needed and time.time() - budget["last_slurm_submit"] >= SLURM_COOLDOWN:
                            current = read_state(root)
                            if (root / "STOP").exists() or time.time() >= current["deadline"]:
                                break
                            current_pending = sum(task["group"] == group and task["status"] == "pending" for task in current["tasks"])
                            needed = replacement_slots(current_pending, active)
                            if needed:
                                command = ["sbatch", "--parsable", f"--array=0-{needed - 1}%{needed}",
                                           "--job-name=ar-direct-auto", str(root / "src" / "autoresearch_workers.sbatch")]
                                budget["slurm_submissions"] += 1
                                budget["last_slurm_submit"] = time.time()
                                budget["slurm_pending_submission"] = True
                                record("slurm_submit_attempt", command=command, slots=needed, active=active)
                                submitted = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
                                job_id = submitted.stdout.strip().split(";", 1)[0]
                                if not re.fullmatch(r"\d+", job_id):
                                    raise ValueError(f"unrecognized sbatch response: {submitted.stdout!r}")
                                tracked_jobs.add(job_id)
                                record("slurm_submitted", job_id=job_id, command=command, slots=needed)
                                budget["slurm_pending_submission"] = False
                except Exception as exc:
                    record("error", message=f"{type(exc).__name__}: {exc}")
                    if site == "slurm" and budget["slurm_pending_submission"]:
                        record("ambiguous_submission", reason="submit outcome uncertain; further submissions blocked")
                stop.wait(POLL_SECONDS)
            record("supervisor_exit", pid=os.getpid(), dgx_restart_count=budget["dgx_restarts"],
                   slurm_submission_count=budget["slurm_submissions"])
        finally:
            supervisor_lock.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--site", choices=("dgx", "slurm"), required=True)
    parser.add_argument("--slurm-jobs", nargs="*", default=[])
    args = parser.parse_args(argv)
    job_ids = {part for value in args.slurm_jobs for part in value.split(",") if part}
    if any(not re.fullmatch(r"\d+", job) for job in job_ids):
        parser.error("--slurm-jobs must contain numeric parent array IDs")
    if args.site == "slurm" and not job_ids:
        parser.error("--slurm-jobs is required to count existing campaign allocations")
    return supervise(args.root, args.site, job_ids)


if __name__ == "__main__":
    raise SystemExit(main())
