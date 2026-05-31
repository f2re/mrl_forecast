import os
import sys
import unittest
import json
import tempfile
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import web_app  # noqa: E402


class _UniformForecastModel(torch.nn.Module):
    def forward(self, input_tensor):
        batch, _, _, height, width = input_tensor.shape
        output = torch.full((batch, 4, 1, height, width), 0.35)
        return output, None


class WebAppTest(unittest.TestCase):
    def setUp(self):
        self.previous_model = web_app.model
        web_app.model = _UniformForecastModel()

    def tearDown(self):
        web_app.model = self.previous_model

    def test_demo_mode_is_explicit_and_uniform_forecast_is_rejected(self):
        client = web_app.app.test_client()

        response = client.post(
            "/api/predict",
            data={"source_type": "demo", "ftp_station": "kokx"},
        )

        self.assertEqual(response.status_code, 422)
        payload = response.get_json()
        self.assertIn("однород", payload["error"].lower())
        self.assertEqual(payload["source_status"], "demo")

    def test_registry_model_must_be_completed_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "metadata.json").write_text(
                json.dumps({"type": "model", "status": "training"}),
                encoding="utf-8",
            )

            self.assertFalse(web_app.is_model_usable(str(model_dir)))


if __name__ == "__main__":
    unittest.main()
