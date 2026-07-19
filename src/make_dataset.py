"""Prepare versioned training sequences from observed radar archives."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
from typing import Optional

import numpy as np

from metadata_utils import load_metadata, save_metadata
from radar_pipeline import RadarDecodeError, RadarPipeline, RadarPipelineConfig

NEXRAD_TIMESTAMP = re.compile(r"^[A-Z0-9]{4}(\d{8})_(\d{6})")
SAMPLE_FORMAT = "npz-reflectivity-mask-v1"


def _timestamp_from_path(path: pathlib.Path) -> datetime.datetime:
    match = NEXRAD_TIMESTAMP.match(path.name.upper())
    if not match:
        raise RadarDecodeError(
            f"Observation timestamp is not encoded in {path.name}; file mtime is not accepted"
        )
    return datetime.datetime.strptime(
        "".join(match.groups()),
        "%Y%m%d%H%M%S",
    ).replace(tzinfo=datetime.UTC)


def _pipeline_for_legacy_arguments(
    grid_shape: tuple[int, int, int],
    grid_limits: tuple[tuple[float, float], ...],
) -> RadarPipeline:
    return RadarPipeline(
        RadarPipelineConfig(
            height=grid_shape[1],
            width=grid_shape[2],
            radius_km=max(abs(grid_limits[1][0]), abs(grid_limits[1][1])) / 1000.0,
            vertical_limit_m=grid_limits[0][1],
        )
    )


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


def _save_sequence(path: pathlib.Path, frames) -> None:
    reflectivity = np.stack([frame.data for frame in frames], axis=0).astype(np.float32)
    valid_mask = np.stack([frame.valid_mask for frame in frames], axis=0).astype(bool)
    timestamps = np.asarray(
        [frame.timestamp_utc.isoformat() for frame in frames],
        dtype="U32",
    )
    np.savez_compressed(
        path,
        reflectivity=reflectivity,
        valid_mask=valid_mask,
        timestamps_utc=timestamps,
    )


def process_archive_directory(
    archive_dir,
    output_root,
    sequence_length=8,
    grid_shape=(1, 256, 256),
    grid_limits=((0, 10000), (-250000.0, 250000.0), (-250000.0, 250000.0)),
    pipeline: Optional[RadarPipeline] = None,
):
    """Grid observed radar files and save masked sequences plus provenance metadata."""
    archive_path = pathlib.Path(archive_dir)
    files = sorted(
        path
        for path in archive_path.iterdir()
        if path.is_file() and not path.name.endswith("_MDM") and path.name != "metadata.json"
    )
    if not files:
        raise ValueError(f"No valid radar files found in {archive_dir}")

    source_meta = load_metadata(archive_dir)
    if not source_meta or source_meta.get("type") != "raw_data":
        raise ValueError(f"Raw archive metadata is missing or invalid in {archive_dir}")
    if source_meta.get("status") != "completed":
        raise ValueError(f"Raw archive is not completed: {source_meta.get('status', 'unknown')}")
    station = source_meta.get("station")
    if not station:
        raise ValueError("Raw archive metadata does not contain a station")

    radar_pipeline = pipeline or _pipeline_for_legacy_arguments(grid_shape, grid_limits)
    dataset_id = f"dataset_{station}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = pathlib.Path(output_root) / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "dataset",
        "source_path": archive_dir,
        "source_type": source_meta.get("source", "aws"),
        "station": station,
        "sequence_length": sequence_length,
        "sample_format": SAMPLE_FORMAT,
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
                source=metadata["source_type"],
            )
            frames.append(frame)
            frame_manifest.append(
                {
                    "source_file": path.name,
                    "timestamp_utc": frame.timestamp_utc.isoformat(),
                    "status": frame.status,
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
            _save_sequence(output_dir / filename, selected)
            sequence_valid_fraction = float(
                np.mean(np.stack([frame.valid_mask for frame in selected], axis=0))
            )
            sequences.append(
                {
                    "file": filename,
                    "format": SAMPLE_FORMAT,
                    "segment": segment_index,
                    "frame_indices": [frame_indices[id(frame)] for frame in selected],
                    "start_time_utc": selected[0].timestamp_utc.isoformat(),
                    "end_time_utc": selected[-1].timestamp_utc.isoformat(),
                    "valid_fraction": sequence_valid_fraction,
                }
            )
    if not sequences:
        metadata["status"] = "failed"
        metadata["error"] = "No regular observed sequences"
        save_metadata(str(output_dir), metadata)
        raise ValueError("No regular observed sequences after cadence quality control")

    manifest = {
        "pipeline": radar_pipeline.metadata(),
        "sample_format": SAMPLE_FORMAT,
        "station": station,
        "frames": frame_manifest,
        "sequences": sequences,
        "decode_errors": errors,
        "regular_segment_count": len(segments),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    metadata["status"] = "completed"
    metadata["sample_count"] = len(sequences)
    metadata["observed_frame_count"] = len(frames)
    metadata["selected_frame_count"] = sum(len(segment) for segment in segments)
    metadata["regular_segment_count"] = len(segments)
    metadata["decode_error_count"] = len(errors)
    save_metadata(str(output_dir), metadata)
    print(f"Saved {len(sequences)} masked sequences to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", required=True, help="Directory with raw NEXRAD Level II files")
    parser.add_argument("--output-dir", default="data/processed_archive")
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()
    process_archive_directory(args.archive_dir, args.output_dir, args.seq_len)
