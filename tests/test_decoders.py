import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bufr_decoder import MRLBufrDecoder  # noqa: E402
from nexrad_decoder import NEXRADDecoder  # noqa: E402
from radar_pipeline import RadarDecodeError  # noqa: E402


class DecoderTrustBoundaryTest(unittest.TestCase):
    def test_invalid_bufr_is_rejected(self):
        with tempfile.NamedTemporaryFile() as radar_file:
            radar_file.write(b"not-bufr")
            radar_file.flush()
            with self.assertRaises(RadarDecodeError):
                MRLBufrDecoder().decode(radar_file.name)

    def test_invalid_nexrad_is_rejected(self):
        with tempfile.NamedTemporaryFile() as radar_file:
            radar_file.write(b"not-nexrad")
            radar_file.flush()
            with self.assertRaises(RadarDecodeError):
                NEXRADDecoder().decode(radar_file.name)


if __name__ == "__main__":
    unittest.main()

