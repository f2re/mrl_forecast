import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from international_sources import DmiRadarSource, KnmiRadarSource  # noqa: E402
from source_access import (  # noqa: E402
    CredentialStore,
    SourceAccessError,
    SourceProbeResult,
)
from source_registry import build_default_source_registry  # noqa: E402


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _DmiSession:
    def get(self, url, **_kwargs):
        if url.endswith("/collections"):
            return _Response({"collections": [{"id": "volume"}]})
        return _Response(
            {
                "features": [
                    {
                        "id": "20260719T120000_station.h5",
                        "properties": {
                            "datetime": "2026-07-19T12:00:00Z",
                            "stationId": "06194",
                        },
                        "asset": {"data": {"href": "https://example.invalid/radar.h5"}},
                    }
                ]
            }
        )


class SourceAccessTest(unittest.TestCase):
    def test_credentials_are_stored_outside_repo_with_private_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.json"
            store = CredentialStore(str(path))
            previous = os.environ.pop("TEST_RADAR_TOKEN", None)
            try:
                store.set("TEST_RADAR_TOKEN", "secret-value")
                self.assertEqual(store.get("TEST_RADAR_TOKEN"), "secret-value")
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["TEST_RADAR_TOKEN"], "secret-value")
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            finally:
                os.environ.pop("TEST_RADAR_TOKEN", None)
                if previous is not None:
                    os.environ["TEST_RADAR_TOKEN"] = previous

    def test_probe_report_redacts_secret_fields_and_signed_queries(self):
        report = SourceProbeResult(
            source_id="fixture",
            status="available",
            reachable=True,
            can_list=True,
            can_download=True,
            credential_state="present",
            sample={
                "url": "https://example.invalid/file.h5?signature=secret",
                "Authorization": "secret-key",
                "metadata": {
                    "href": "https://example.invalid/other.h5?token=secret",
                    "key": "public/object/key.h5",
                },
            },
        ).to_metadata()

        self.assertEqual(report["sample"]["url"], "https://example.invalid/file.h5")
        self.assertEqual(report["sample"]["Authorization"], "<redacted>")
        self.assertEqual(
            report["sample"]["metadata"]["href"],
            "https://example.invalid/other.h5",
        )
        self.assertEqual(report["sample"]["metadata"]["key"], "public/object/key.h5")

    def test_knmi_reports_missing_key_without_network_call(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.pop("KNMI_API_KEY", None)
            try:
                source = KnmiRadarSource(
                    credentials=CredentialStore(str(Path(directory) / "credentials.json"))
                )
                probe = source.probe()
                self.assertEqual(probe.status, "credential_required")
                self.assertFalse(probe.can_download)
                with self.assertRaises(SourceAccessError):
                    source.list_files(limit=1)
            finally:
                if previous is not None:
                    os.environ["KNMI_API_KEY"] = previous

    def test_dmi_probe_parses_downloadable_stac_asset(self):
        source = DmiRadarSource(session=_DmiSession())
        probe = source.probe(download_test=False, station="06194")

        self.assertEqual(probe.status, "available")
        self.assertTrue(probe.can_list)
        self.assertTrue(probe.can_download)
        self.assertEqual(probe.sample["station_id"], "06194")

    def test_registry_exposes_active_and_manual_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = build_default_source_registry(
                credentials=CredentialStore(str(Path(directory) / "credentials.json"))
            )
            descriptions = {item["source_id"]: item for item in registry.describe()}

        self.assertIn("opera-ord", descriptions)
        self.assertIn("fmi-s3", descriptions)
        self.assertIn("dmi-radar", descriptions)
        self.assertIn("knmi-radar", descriptions)
        self.assertIn("ncradar-cao", descriptions)
        self.assertEqual(descriptions["knmi-radar"]["credential_state"], "missing")
        self.assertFalse(descriptions["ncradar-cao"]["download_supported"])


if __name__ == "__main__":
    unittest.main()
