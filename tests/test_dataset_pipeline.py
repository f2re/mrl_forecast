import json
import os
import sys
import tempfile
import unittest
import datetime
from pathlib import Path

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from make_dataset import process_archive_directory, regular_frame_segments  # noqa: E402
from radar_pipeline import PIPELINE_VERSION, RadarFrame  # noqa: E402
from train_nowcasting_model import temporal_split_indices  # noqa: E402


class _FixturePipeline:
    def metadata(self):
        return {
            "pipeline_version": PIPELINE_VERSION,
            "product": "lowest_elevation_reflectivity",
            "units": "dBZ",
            "time_step_minutes": 10,
            "grid": {"width": 4, "height": 4, "radius_km": 250.0, "crs": "local_aeqd"},
        }

    def process_file(self, path, *, timestamp_utc, station, source):
        value = float(Path(path).name[-1])
        data = np.full((4, 4), value, dtype=np.float32)
        return RadarFrame(
            data=data,
            valid_mask=np.ones((4, 4), dtype=bool),
            timestamp_utc=timestamp_utc,
            station=station,
            source=source,
            qc={"pipeline_version": PIPELINE_VERSION, "valid_fraction": 1.0},
            provenance={"path": path},
        )


class DatasetPipelineTest(unittest.TestCase):
    def test_archive_processing_saves_pipeline_metadata_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            output = root / "processed"
            archive.mkdir()
            for index in range(3):
                (archive / f"KOKX20240520_00{index}000_V06").write_bytes(b"fixture")
            (archive / "metadata.json").write_text(
                json.dumps(
                    {
                        "type": "raw_data",
                        "station": "KOKX",
                        "status": "completed",
                    }
                ),
                encoding="utf-8",
            )

            dataset_dir = process_archive_directory(
                str(archive),
                str(output),
                sequence_length=2,
                pipeline=_FixturePipeline(),
            )

            metadata = json.loads((Path(dataset_dir) / "metadata.json").read_text(encoding="utf-8"))
            manifest = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["pipeline"]["pipeline_version"], PIPELINE_VERSION)
            self.assertEqual(metadata["sample_count"], 2)
            self.assertEqual(len(manifest["frames"]), 3)
            self.assertEqual(len(manifest["sequences"]), 2)

    def test_temporal_split_leaves_gap_for_overlapping_windows(self):
        train_indices, validation_indices = temporal_split_indices(
            sample_count=20,
            overlap_frames=7,
            val_fraction=0.25,
        )

        self.assertEqual(train_indices, list(range(8)))
        self.assertEqual(validation_indices, list(range(15, 20)))
        self.assertLess(max(train_indices) + 7, min(validation_indices))

    def test_regular_segments_do_not_bridge_observation_gaps(self):
        start = datetime.datetime(2026, 5, 30, tzinfo=datetime.UTC)
        frames = [
            RadarFrame(
                data=np.zeros((2, 2), dtype=np.float32),
                valid_mask=np.ones((2, 2), dtype=bool),
                timestamp_utc=start + datetime.timedelta(minutes=minutes),
                station="KOKX",
                source="fixture",
            )
            for minutes in (0, 5, 10, 20, 60, 70)
        ]

        segments = regular_frame_segments(frames, step_minutes=10, tolerance_minutes=4)

        self.assertEqual([[frame.timestamp_utc.minute for frame in segment] for segment in segments], [[0, 10, 20], [0, 10]])


if __name__ == "__main__":
    unittest.main()
