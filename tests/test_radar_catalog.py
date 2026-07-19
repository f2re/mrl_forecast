import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from radar_catalog import RadarCatalog  # noqa: E402


class RadarCatalogTest(unittest.TestCase):
    def test_indexes_raw_observation_with_checksum_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            archive.mkdir()
            filename = "KOKX20240520_001500_V06"
            (archive / filename).write_bytes(b"radar fixture")
            (archive / "metadata.json").write_text(
                json.dumps(
                    {
                        "type": "raw_data",
                        "source": "aws",
                        "station": "KOKX",
                        "date": "2024-05-20",
                        "status": "completed",
                        "files": [filename],
                    }
                ),
                encoding="utf-8",
            )
            catalog = RadarCatalog(str(root / "catalog.sqlite3"))

            catalog.index_archive(str(archive))
            summary = catalog.summary()
            observations = catalog.list_observations()

            self.assertEqual(summary["observations"], 1)
            self.assertEqual(observations[0]["station"], "KOKX")
            self.assertEqual(observations[0]["timestamp_utc"], "2024-05-20T00:15:00+00:00")
            self.assertEqual(len(observations[0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
