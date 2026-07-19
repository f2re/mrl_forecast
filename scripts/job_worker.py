#!/usr/bin/env python3
"""Run queued MRL jobs one at a time on the local machine."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jobs import JobStore  # noqa: E402

STOP_REQUESTED = False


def _request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_job(store: JobStore, job: dict) -> None:
    details = store.get(job["id"], include_command=True)
    command = details["command"]
    log_path = Path(details["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        store.set_pid(job["id"], process.pid)
        while process.poll() is None:
            if STOP_REQUESTED or store.is_cancelling(job["id"]):
                _terminate(process)
                break
            time.sleep(0.5)
        return_code = process.poll()
        if return_code is None:
            return_code = 143
        store.finish(job["id"], return_code)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    store = JobStore()
    store.mark_interrupted()

    while not STOP_REQUESTED:
        job = store.claim_next()
        if job is None:
            if args.once:
                return 0
            time.sleep(max(args.poll_seconds, 0.2))
            continue
        try:
            run_job(store, job)
        except Exception as exc:
            store.finish(job["id"], 1, error=str(exc))
        if args.once:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
