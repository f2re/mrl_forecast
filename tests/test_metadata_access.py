import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from metadata_utils import load_metadata  # noqa: E402


class MetadataAccessTest(unittest.TestCase):
    def test_unverified_source_changes_effective_dataset_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            dataset = root / "dataset"
            raw.mkdir()
            dataset.mkdir()
            (raw / "metadata.json").write_text(
                json.dumps(
                    {
                        "type": "raw_data",
                        "access": {"source_id": "fmi-s3", "training_allowed": False},
                    }
                ),
                encoding="utf-8",
            )
            (dataset / "metadata.json").write_text(
                json.dumps(
                    {
                        "type": "dataset",
                        "status": "completed",
                        "source_path": str(raw),
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_metadata(str(dataset))

            self.assertEqual(metadata["status"], "unverified_source")
            self.assertFalse(metadata["source_training_allowed"])
            self.assertIn("training_block_reason", metadata)

    def test_verified_source_remains_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            dataset = root / "dataset"
            raw.mkdir()
            dataset.mkdir()
            (raw / "metadata.json").write_text(
                json.dumps(
                    {
                        "type": "raw_data",
                        "access": {"source_id": "fixture", "training_allowed": True},
                    }
                ),
                encoding="utf-8",
            )
            (dataset / "metadata.json").write_text(
                json.dumps(
                    {
                        "type": "dataset",
                        "status": "completed",
                        "source_path": str(raw),
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_metadata(str(dataset))

            self.assertEqual(metadata["status"], "completed")
            self.assertTrue(metadata["source_training_allowed"])


if __name__ == "__main__":
    unittest.main()
