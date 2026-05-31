import datetime
import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from adapters import NOAAAWSAdapter  # noqa: E402
from radar_pipeline import RadarSourceError  # noqa: E402


class _FailingAwsConnection:
    def get_avail_scans(self, *args, **kwargs):
        raise OSError("aws unavailable")


class AdapterTest(unittest.TestCase):
    def test_aws_adapter_raises_source_error_instead_of_returning_demo_data(self):
        adapter = NOAAAWSAdapter(conn=_FailingAwsConnection())

        with self.assertRaisesRegex(RadarSourceError, "aws unavailable"):
            adapter.get_latest_sequence(
                4,
                station_code="kokx",
                end_time=datetime.datetime(2026, 5, 30, tzinfo=datetime.UTC),
            )

    def test_aws_adapter_forces_public_bucket_region(self):
        previous_default_region = os.environ.get("AWS_DEFAULT_REGION")
        previous_region = os.environ.get("AWS_REGION")
        os.environ["AWS_DEFAULT_REGION"] = "ru-central1"
        os.environ["AWS_REGION"] = "ru-central1"
        try:
            NOAAAWSAdapter(conn=_FailingAwsConnection())
            self.assertEqual(os.environ["AWS_DEFAULT_REGION"], "us-east-1")
            self.assertEqual(os.environ["AWS_REGION"], "us-east-1")
        finally:
            if previous_default_region is None:
                os.environ.pop("AWS_DEFAULT_REGION", None)
            else:
                os.environ["AWS_DEFAULT_REGION"] = previous_default_region
            if previous_region is None:
                os.environ.pop("AWS_REGION", None)
            else:
                os.environ["AWS_REGION"] = previous_region


if __name__ == "__main__":
    unittest.main()

