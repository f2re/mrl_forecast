"""Prepare versioned training sequences from observed radar archives."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
from typing import Optional

import numpy as np

from config import FORECAST_STEP_MINUTES
from event_catalog import class_counts, summarize_sequence
from metadata_utils import load_metadata, save_metadata
from radar_catalog import RadarCatalog
from radar_pipeline import RadarDecodeError, RadarPipeline, RadarPipelineConfig

NEXRAD_TIMESTAMP = re.compile(r"^[A-Z0-9]{4}(\d{8})_(\d{6})")
DWD_TIMESTAMP = re.compile(r"DBZH_00-(\d{12})", re.IGNORECASE)
SAMPLE_FORMAT = "npz-radar-quality-v2"
QUALITY_FIELDS = (
    "valid_mask",
    "coverage_mask",
    "clutter_mask",
    "interpolation_weight",
)
SPLIT_BLOCK_HOURS = 3


def _timestamp_from_path(path: pathlib.Path) -> datetime.datetime:
    nexrad_match = NEXRAD_TIMESTAMP.match(path.name.upper())
    if nexrad_match:
        return datetime.datetime.strptime(
            "".join(nexrad_match.groups()),
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=datetime.UTC)

    dwd_match = DWD_TIMESTAMP.search(path.name)
    if dwd_match:
        return datetime.datetime.strptime(
            dwd_match.group(1),
            "%Y%m%d%H%M",
        ).replace(tzinfo=datetime.UTC)

    raise RadarDecodeError(
        f"Observation timestamp is not encoded in {path.name}; file mtime is not accepted"
    )


def _split_group(timestamp: datetime.datetime, segment_index: int) -> str:
    """Return a chronological block used to keep adjacent windows together."""

    block_hour = (timestamp.hour // SPLIT_BLOCK_HOURS) * SPLIT_BLOCK_HOURS
    return f"{timestamp:%Y%m%d}_s{segment_index:03d}_h{block_hour:02d}"


def _pipeline_for_legacy_arguments(
    grid_shape: tuple[int, int, int],
    grid_limits: tuple[tuple[float, float], ...],
    time_step_minutes: int,
) -> RadarPipeline:
    return RadarPipeline(
        RadarPipelineConfig(
            height=grid_shape[1],
            width=grid_shape[2],
            radius_km=max(abs(grid_limits[1][0]), abs(grid_limits[1][1])) / 1000.0,
            vertical_limit_m=grid_limits[0][1],
            time_step_minutes=time_step_minutes,
        )
    )


def _source_pipeline(pipeline: RadarPipeline, source_type: str) -> RadarPipeline:
    """Attach only the decoder required by the source; gridding stays shared."""

    if source_type == "dwd-open-data":
        from dwd_source import DWDOpenDataAdapter

        return RadarPipeline(
            config=pipeline.config,
            radar_reader=DWDOpenDataAdapter._read_odim,
        )
    return pipeline


def regular_frame_segments(frames, step_minutes: int, tolerance_minutes: int = 4):
    """Select approximately regular frames and split sequences at observation gaps."""
    segments = []
    current_segment = []
    minimum_step = datetime.timedelta(minutes=step_minutes - tolerance_minutes)
    maximum_step = datetime.timedelta(minutes=step_minutes + tolerance_minutes)
    for frame in sorted(frames, key=lambda item: item.timestamp_utc):
        if not current_segment:
            current_segment = [frame]
            continue
        difference = frame.timestamp_utc - current_segment[-1].timestamp_utc
        if difference < minimum_step:
            continue
        if difference <= maximum_step:
            current_segment.append(frame)
            continue
        segments.append(current_segment)
        current_segment = [frame]
    if current_segment:
        segments.append(current_segment)
    return segments


def _sequence_arrays(frames) -> dict[str, np.ndarray]:
    return {
        "reflectivity": np.stack([frame.data for frame in frames], axis=0).astype(np.float32),
        "valid_mask": np.stack([frame.valid_mask for frame in frames], axis=0).astype(bool),
        "coverage_mask": np.stack([frame.coverage_mask for frame in frames], axis=0).astype(bool),
        "clutter_mask": np.stack([frame.clutter_mask for frame in frames], axis=0).astype(bool),
        "interpolation_weight": np.stack(
            [frame.interpolation_weight for frame in frames],
            axis=0,
        ).astype(np.float32),
    }


def _save_sequence(
    path: pathlib.Path,
    arrays: dict[str, np.ndarray],
    frames,
) -> None:
    timestamps = np.asarray(
        [frame.timestamp_utc.isoformat() for frame in frames],
        dtype="U32",
    )
    np.savez_compressed(path, **arrays, timestamps_utc=timestamps)


def process_archive_directory(
    archive_dir,
    output_root,
    sequence_length=8,
    grid_shape=(1, 256, 256),
    grid_limits=((0, 10000), (-250000.0, 250000.0), (-250000.0, 250000.0)),
    pipeline: Optional[RadarPipeline] = None,
    grid_profile: str = "canonical",
    time_step_minutes: int = FORECAST_STEP_MINUTES,
):
    """Grid observed radar files and save quality-aware sequences and metadata."""
    archive_path = pathlib.Path(archive_dir).resolve()
    files = sorted(
        path
        for path in archive_path.iterdir()
        if path.is_file() and not path.name.endswith("_MDM") and path.name != "metadata.json"
    )
    if not files:
        raise ValueError(f"No valid radar files found in {archive_dir}")

    source_meta = load_metadata(str(archive_path))
    if not source_meta or source_meta.get("type") != "raw_data":
        raise ValueError(f"Raw archive metadata is missing or invalid in {archive_dir}")
    if source_meta.get("status") != "completed":
        raise ValueError(f"Raw archive is not completed: {source_meta.get('status', 'unknown')}")
    station = source_meta.get("station")
    if not station:
        raise ValueError("Raw archive metadata does not contain a station")
    source_type = source_meta.get("source", "aws")

    if pipeline is not None:
        radar_pipeline = pipeline
    elif grid_profile == "canonical":
        radar_pipeline = RadarPipeline(
            config=RadarPipelineConfig.canonical(time_step_minutes=time_step_minutes)
        )
    elif grid_profile == "legacy":
        radar_pipeline = _pipeline_for_legacy_arguments(
            grid_shape, grid_limits, time_step_minutes
        )
    else:
        raise ValueError(f"Unknown grid profile: {grid_profile}")
    radar_pipeline = _source_pipeline(radar_pipeline, source_type)

    dataset_id = f"dataset_{station}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = pathlib.Path(output_root) / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "dataset",
        "source_path": str(archive_path),
        "source_type": source_type,
        "station": station,
        "sequence_length": sequence_length,
        "sample_format": SAMPLE_FORMAT,
        "quality_fields": list(QUALITY_FIELDS),
        "grid_profile": grid_profile,
        "time_step_minutes": time_step_minutes,
        "split_block_hours": SPLIT_BLOCK_HOURS,
        "pipeline": radar_pipeline.metadata(),
        "status": "processing",
        "sample_count": 0,
    }
    save_metadata(str(output_dir), metadata)

    frames = []
    frame_manifest = []
    errors = []
    for path in files:
        try:
            frame = radar_pipeline.process_file(
                str(path),
                timestamp_utc=_timestamp_from_path(path),
                station=station,
                source=source_type,
            )
            frames.append(frame)
            frame_manifest.append(
                {
                    "source_file": path.name,
                    "timestamp_utc": frame.timestamp_utc.isoformat(),
                    "status": frame.status,
                    "quality": frame.quality_summary(),
                    "qc": frame.qc,
                    "provenance": frame.provenance,
                }
            )
            print(f"Successfully gridded {path.name}")
        except RadarDecodeError as exc:
            errors.append({"source_file": path.name, "error": str(exc)})
            print(f"Error processing {path.name}: {exc}")

    if len(frames) < sequence_length:
        metadata["status"] = "failed"
        metadata["error"] = "Not enough valid observed files"
        metadata["decode_errors"] = errors
        save_metadata(str(output_dir), metadata)
        raise ValueError("Not enough valid observed files to form a sequence")

    segments = regular_frame_segments(
        frames,
        step_minutes=radar_pipeline.metadata()["time_step_minutes"],
    )
    frame_indices = {id(frame): index for index, frame in enumerate(frames)}
    sequences = []
    for segment_index, segment in enumerate(segments):
        for index in range(len(segment) - sequence_length + 1):
            selected = segment[index : index + sequence_length]
            if any(frame.status != "observed" for frame in selected):
                raise ValueError("Production datasets cannot contain non-observed frames")
            filename = f"seq_{len(sequences):04d}.npz"
            arrays = _sequence_arrays(selected)
            _save_sequence(output_dir / filename, arrays, selected)
            statistics = summarize_sequence(arrays["reflectivity"], arrays["valid_mask"])
            sequences.append(
                {
                    "file": filename,
                    "format": SAMPLE_FORMAT,
                    "segment": segment_index,
                    "split_group": _split_group(selected[0].timestamp_utc, segment_index),
                    "frame_indices": [frame_indices[id(frame)] for frame in selected],
                    "start_time_utc": selected[0].timestamp_utc.isoformat(),
                    "end_time_utc": selected[-1].timestamp_utc.isoformat(),
                    "event_class": statistics["event_class"],
                    "statistics": statistics,
                    "quality": {
                        "mean_valid_fraction": float(arrays["valid_mask"].mean()),
                        "mean_coverage_fraction": float(arrays["coverage_mask"].mean()),
                        "mean_clutter_fraction": float(arrays["clutter_mask"].mean()),
                        "mean_interpolation_weight": float(arrays["interpolation_weight"].mean()),
                    },
                }
            )
    if not sequences:
        metadata["status"] = "failed"
        metadata["error"] = "No regular observed sequences"
        save_metadata(str(output_dir), metadata)
        raise ValueError("No regular observed sequences after cadence quality control")

    counts = class_counts(item["event_class"] for item in sequences)
    split_groups = sorted({item["split_group"] for item in sequences})
    manifest = {
        "pipeline": radar_pipeline.metadata(),
        "sample_format": SAMPLE_FORMAT,
        "quality_fields": list(QUALITY_FIELDS),
        "station": station,
        "frames": frame_manifest,
        "sequences": sequences,
        "decode_errors": errors,
        "regular_segment_count": len(segments),
        "split_group_count": len(split_groups),
        "class_counts": counts,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    metadata["status"] = "completed"
    metadata["sample_count"] = len(sequences)
    metadata["observed_frame_count"] = len(frames)
    metadata["selected_frame_count"] = sum(len(segment) for segment in segments)
    metadata["regular_segment_count"] = len(segments)
    metadata["split_group_count"] = len(split_groups)
    metadata["decode_error_count"] = len(errors)
    metadata["class_counts"] = counts
    save_metadata(str(output_dir), metadata)
    try:
        catalog = RadarCatalog()
        catalog.index_archive(str(archive_path))
        catalog.index_dataset(str(output_dir))
        metadata["catalog_indexed"] = True
    except Exception as exc:
        metadata["catalog_indexed"] = False
        metadata["catalog_error"] = str(exc)
    save_metadata(str(output_dir), metadata)

    print(f"Saved {len(sequences)} quality-aware sequences to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", required=True, help="Directory with raw radar files")
    parser.add_argument("--output-dir", default="data/processed_archive")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--grid-profile", choices=("canonical", "legacy"), default="canonical")
    parser.add_argument("--time-step-minutes", type=int, default=FORECAST_STEP_MINUTES)
    args = parser.parse_args()
    process_archive_directory(
        args.archive_dir,
        args.output_dir,
        args.seq_len,
        grid_profile=args.grid_profile,
        time_step_minutes=args.time_step_minutes,
    )
