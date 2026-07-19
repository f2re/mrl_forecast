#!/usr/bin/env python3
"""Unified command line entry point for MRL Forecast workflows."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Sequence

ROOT = pathlib.Path(__file__).resolve().parent


def _run(relative_path: str, arguments: Sequence[str], env: dict | None = None) -> int:
    command = [sys.executable, str(ROOT / relative_path), *[str(value) for value in arguments]]
    result = subprocess.run(command, cwd=ROOT, env=env or os.environ.copy())
    return int(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrl",
        description="MRL Forecast: загрузка, подготовка, обучение, инференс и диагностика",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Проверить окружение и источники")
    doctor.add_argument("--check-aws", action="store_true")
    doctor.add_argument("--check-dwd", action="store_true")
    doctor.add_argument("--station", default="KOKX")
    doctor.add_argument("--dwd-station", default="ess")
    doctor.add_argument("--date", default="2024-05-20")

    download = commands.add_parser("download", help="Скачать открытый радарный архив")
    download.add_argument("--source", choices=("noaa", "dwd"), default="noaa")
    download.add_argument("--station", required=True)
    download.add_argument("--date", required=True)
    download.add_argument("--count", type=int, default=100)
    download.add_argument("--output", default="data/raw/archive")

    prepare = commands.add_parser("prepare", help="Построить quality-aware dataset")
    prepare.add_argument("--archive-dir", required=True)
    prepare.add_argument("--output-dir", default="data/processed_archive")
    prepare.add_argument("--seq-len", type=int, default=8)
    prepare.add_argument("--grid-profile", choices=("canonical", "legacy"), default="canonical")
    prepare.add_argument("--time-step-minutes", type=int, default=15)

    train = commands.add_parser("train", help="Обучить ConvLSTM или MRL-PhysEvolution")
    train.add_argument("--data-dirs", required=True)
    train.add_argument("--architecture", choices=("phys-evolution", "convlstm"), default="phys-evolution")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--val-split", type=float, default=0.2)
    train.add_argument("--test-split", type=float, default=0.1)
    train.add_argument("--input-length", type=int, default=4)
    train.add_argument("--target-length", type=int, default=4)
    train.add_argument("--base-channels", type=int, default=16)
    train.add_argument("--hidden-channels", type=int, default=24)
    train.add_argument("--output-dir", default="models/registry")
    train.add_argument("--no-balanced-sampling", action="store_true")

    infer = commands.add_parser("infer", help="Выполнить прогноз из терминала")
    infer.add_argument("--model-path", required=True)
    infer.add_argument("--source", choices=("aws", "local", "demo"), default="aws")
    infer.add_argument("--station", default="KOKX")
    infer.add_argument("--local-dir", default="data/processed")
    infer.add_argument("--output-dir", default="data/predictions")

    catalog = commands.add_parser("catalog", help="Управлять SQLite-каталогом")
    catalog.add_argument("catalog_args", nargs=argparse.REMAINDER)

    benchmark = commands.add_parser("benchmark", help="Измерить CPU latency и RAM")
    benchmark.add_argument("--model-path", required=True)
    benchmark.add_argument("--threads", type=int, default=0)
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--save", action="store_true")

    serve = commands.add_parser("serve", help="Запустить веб-интерфейс")
    serve.add_argument("--model-path")
    serve.add_argument("--port", type=int, default=5005)
    serve.add_argument("--debug", action="store_true")

    worker = commands.add_parser("worker", help="Запустить локальный job worker")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--poll-seconds", type=float, default=1.0)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "doctor":
        command = ["--station", args.station, "--dwd-station", args.dwd_station, "--date", args.date]
        if args.check_aws:
            command.append("--check-aws")
        if args.check_dwd:
            command.append("--check-dwd")
        return _run("scripts/doctor.py", command)

    if args.command == "download":
        script = "src/download_archive.py" if args.source == "noaa" else "src/download_dwd_archive.py"
        return _run(
            script,
            [
                "--station", args.station,
                "--date", args.date,
                "--count", args.count,
                "--output", args.output,
            ],
        )

    if args.command == "prepare":
        return _run(
            "src/make_dataset.py",
            [
                "--archive-dir", args.archive_dir,
                "--output-dir", args.output_dir,
                "--seq-len", args.seq_len,
                "--grid-profile", args.grid_profile,
                "--time-step-minutes", args.time_step_minutes,
            ],
        )

    if args.command == "train":
        command = [
            "--data-dirs", args.data_dirs,
            "--architecture", args.architecture,
            "--epochs", args.epochs,
            "--batch-size", args.batch_size,
            "--lr", args.lr,
            "--val-split", args.val_split,
            "--test-split", args.test_split,
            "--input-length", args.input_length,
            "--target-length", args.target_length,
            "--base-channels", args.base_channels,
            "--hidden-channels", args.hidden_channels,
            "--output-dir", args.output_dir,
        ]
        if args.no_balanced_sampling:
            command.append("--no-balanced-sampling")
        return _run("src/train_nowcasting_model.py", command)

    if args.command == "infer":
        return _run(
            "src/run_inference.py",
            [
                "--model-path", args.model_path,
                "--source", args.source,
                "--station", args.station,
                "--local-dir", args.local_dir,
                "--output-dir", args.output_dir,
            ],
        )

    if args.command == "catalog":
        command = args.catalog_args or ["summary"]
        return _run("scripts/catalog.py", command)

    if args.command == "benchmark":
        command = [
            "--model-path", args.model_path,
            "--threads", args.threads,
            "--warmup", args.warmup,
            "--repeats", args.repeats,
        ]
        if args.save:
            command.append("--save")
        return _run("scripts/benchmark_cpu.py", command)

    if args.command == "serve":
        env = os.environ.copy()
        env["PORT"] = str(args.port)
        env["DEBUG"] = "true" if args.debug else "false"
        if args.model_path:
            env["NOWCAST_MODEL_CHECKPOINT"] = args.model_path
        return _run("src/web_app.py", [], env=env)

    if args.command == "worker":
        command = ["--poll-seconds", args.poll_seconds]
        if args.once:
            command.append("--once")
        return _run("scripts/job_worker.py", command)

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
