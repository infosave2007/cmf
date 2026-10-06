import unittest
from unittest.mock import patch

try:
    import app
    from fastapi.testclient import TestClient
except ImportError:  # Dependency-free logic tests also run in the Rust repo CI image.
    TestClient = None
    app = None


@unittest.skipUnless(app is not None, "Gradio dependencies are not installed")
class SparkAppTests(unittest.TestCase):
    def test_landing_page_renders_without_loading_weights(self):
        response = TestClient(app.demo.app).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Spark Lab", response.text)

    def test_download_reports_progress_without_boolean_coercion(self):
        class Progress:
            def __init__(self):
                self.events = []

            def __bool__(self):
                raise AssertionError("Progress must not be coerced to bool")

            def __call__(self, value, *, desc):
                self.events.append((value, desc))

        progress = Progress()
        with patch.object(app, "hf_hub_download", return_value="/tmp/spark.cmf"):
            path = app.CortiqService._download("Spark 1.7B · fastest", progress)
        self.assertEqual(path, "/tmp/spark.cmf")
        self.assertEqual(progress.events[0][0], 0.04)


if __name__ == "__main__":
    unittest.main()
