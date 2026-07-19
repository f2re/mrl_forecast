#!/usr/bin/env python3
"""Inspect, configure, probe and sample-download radar sources."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from international_sources import KnmiRadarSource  # noqa: E402
from source_access import CredentialStore  # noqa: E402
from source_registry import build_default_source_registry  # noqa: E402


DEFAULT_REPORT_PATH = ROOT / "data" / "source_health.json"


def _date_range(value: str):
    if not value:
        return None, None
    date = datetime.date.fromisoformat(value)
    start = datetime.datetime.combine(date, datetime.time.min, tzinfo=datetime.UTC)
    end = start + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
    return start, end


def _source_kwargs(args) -> dict:
    start, end = _date_range(args.date)
    result = {
        "prefix": args.prefix,
        "limit": args.limit,
        "station": args.station,
        "collection": args.collection,
        "start": start,
        "end": end,
    }
    return {key: value for key, value in result.items() if value not in (None, "")}


def _create_source(registry, args):
    if args.source == "knmi-radar":
        return registry.create(
            args.source,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
        )
    return registry.create(args.source)


def _select_descriptions(registry, source_id: str):
    descriptions = registry.describe()
    if source_id == "all":
        return descriptions
    return [item for item in descriptions if item["source_id"] == source_id]


def _print_registration(capabilities) -> None:
    print(f"Источник: {capabilities.source_id}")
    print(f"Режим доступа: {capabilities.access_mode}")
    if capabilities.registration_url:
        print(f"Регистрация: {capabilities.registration_url}")
    if capabilities.credential_env:
        print(f"Переменная/секрет: {capabilities.credential_env}")
    if capabilities.registration_steps:
        print("Порядок регистрации:")
        for index, step in enumerate(capabilities.registration_steps, start=1):
            print(f"  {index}. {step}")


def _save_report(path_value: str, reports: list[dict]) -> Path:
    path = Path(path_value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checked_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "reports": reports,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("list", "info", "probe", "configure", "sample"),
        default="probe",
    )
    parser.add_argument("--source", default="all")
    parser.add_argument("--download-test", action="store_true")
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--no-save-report", action="store_true")
    parser.add_argument("--output-dir", default="data/source_samples")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--station", default="")
    parser.add_argument("--collection", default="volume")
    parser.add_argument("--date", default="")
    parser.add_argument("--dataset-name", default=KnmiRadarSource.DEFAULT_DATASET)
    parser.add_argument("--dataset-version", default=KnmiRadarSource.DEFAULT_VERSION)
    args = parser.parse_args()

    credentials = CredentialStore()
    registry = build_default_source_registry(credentials=credentials)
    if args.source != "all" and args.source not in registry.source_ids():
        parser.error(
            f"Unknown source {args.source}. Available: {', '.join(registry.source_ids())}"
        )

    if args.action == "list":
        print(json.dumps(_select_descriptions(registry, args.source), indent=2, ensure_ascii=False))
        return 0

    if args.action == "info":
        if args.source == "all":
            parser.error("--action info requires one --source")
        _print_registration(registry.capabilities(args.source))
        return 0

    if args.action == "configure":
        if args.source == "all":
            parser.error("--action configure requires one --source")
        capabilities = registry.capabilities(args.source)
        _print_registration(capabilities)
        if not capabilities.credential_env:
            print("Этот источник не использует токен, сохраняемый MRL Forecast.")
            return 2
        path = credentials.configure(capabilities)
        print(f"Ключ сохранён в {path}; значение не выводится.")
        return 0

    if args.action == "probe":
        kwargs = _source_kwargs(args)
        kwargs["download_test"] = args.download_test
        if args.source == "all":
            report = registry.probe_all(
                active_only=args.active_only,
                max_workers=args.max_workers,
                **kwargs,
            )
        elif args.source == "knmi-radar":
            source = _create_source(registry, args)
            result = source.probe(**kwargs)
            report = [result.to_metadata()]
        else:
            report = [registry.probe(args.source, **kwargs)]
        if not args.no_save_report:
            report_path = _save_report(args.report_path, report)
            print(f"Отчёт сохранён: {report_path}", file=sys.stderr)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0 if all(item["status"] not in {"unavailable"} for item in report) else 1

    if args.source == "all":
        parser.error("--action sample requires one --source")
    source = _create_source(registry, args)
    if not hasattr(source, "list_files") or not hasattr(source, "download"):
        raise RuntimeError(f"Source {args.source} does not implement automated download")
    files = source.list_files(**_source_kwargs(args))
    if not files:
        raise RuntimeError(f"Source {args.source} returned no downloadable files")
    result = source.download(files[0], args.output_dir)
    print(
        json.dumps(
            {"remote": files[0].to_metadata(), "download": result},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
