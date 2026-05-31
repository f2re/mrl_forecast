import datetime
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from radar_pipeline import (  # noqa: E402
    PIPELINE_VERSION,
    DemoRadarAdapter,
    RadarDecodeError,
    RadarFrame,
    RadarPipeline,
    RadarPipelineConfig,
    RadarSequence,
    RadarSourceError,
)


class RadarPipelineTest(unittest.TestCase):
    def test_frame_preserves_mask_and_reports_qc(self):
        pipeline = RadarPipeline()
        masked = np.ma.array(
            [[-5.0, 10.0], [20.0, 80.0]],
            mask=[[False, False], [True, False]],
        )

        frame = pipeline.frame_from_grid(
            masked,
            timestamp_utc=datetime.datetime(2026, 5, 30, tzinfo=datetime.UTC),
            station="KOKX",
            source="fixture",
        )

        self.assertEqual(frame.status, "observed")
        self.assertEqual(frame.data[0, 0], -5.0)
        self.assertFalse(frame.valid_mask[1, 0])
        self.assertEqual(frame.qc["masked_pixels"], 1)
        self.assertEqual(frame.qc["pipeline_version"], PIPELINE_VERSION)

    def test_process_file_raises_decode_error_instead_of_generating_demo_grid(self):
        pipeline = RadarPipeline(radar_reader=mock.Mock(side_effect=ValueError("bad radar")))

        with self.assertRaisesRegex(RadarDecodeError, "bad radar"):
            pipeline.process_file(
                "broken-radar-file",
                timestamp_utc=datetime.datetime(2026, 5, 30, tzinfo=datetime.UTC),
                station="KOKX",
                source="fixture",
            )

    def test_demo_adapter_is_explicit_and_marked_demo(self):
        sequence = DemoRadarAdapter(grid_size=(8, 8)).get_latest_sequence(3)

        self.assertIsInstance(sequence, RadarSequence)
        self.assertEqual(sequence.status, "demo")
        self.assertEqual(sequence.stack().shape, (3, 8, 8))
        self.assertTrue(all(frame.status == "demo" for frame in sequence.frames))

    def test_sequence_rejects_non_observed_frames_for_operational_stack(self):
        frame = RadarFrame(
            data=np.zeros((4, 4), dtype=np.float32),
            valid_mask=np.ones((4, 4), dtype=bool),
            timestamp_utc=datetime.datetime(2026, 5, 30, tzinfo=datetime.UTC),
            station="KOKX",
            source="fixture",
            product="reflectivity",
            status="decode_failed",
        )

        sequence = RadarSequence(frames=[frame], source="fixture", status="decode_failed")

        with self.assertRaisesRegex(RadarSourceError, "decode_failed"):
            sequence.stack(require_observed=True)

    def test_default_config_is_serializable_metadata(self):
        metadata = RadarPipelineConfig().to_metadata()

        self.assertEqual(metadata["pipeline_version"], PIPELINE_VERSION)
        self.assertEqual(metadata["units"], "dBZ")
        self.assertEqual(metadata["time_step_minutes"], 10)
        self.assertEqual(metadata["grid"]["width"], 256)


if __name__ == "__main__":
    unittest.main()

