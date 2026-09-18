"""Exercise the Streamlit workflow with real pipeline results."""

from pathlib import Path
import unittest

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def image_bytes(with_page: bool) -> bytes:
    image = np.full((600, 800, 3), 155, np.uint8)
    if with_page:
        corners = np.int32([[160, 70], [680, 100], [650, 535], [125, 505]])
        cv2.fillConvexPoly(image, corners, (245, 245, 245))
        for y in range(170, 440, 25):
            cv2.line(image, (240, y), (570, y), (45, 45, 45), 2)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not build test image")
    return encoded.tobytes()


class FrontendWorkflowTests(unittest.TestCase):
    def test_webcam_action_reveals_camera_input(self) -> None:
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertEqual(len(app.get("camera_input")), 0)

        next(button for button in app.button if button.label == "Use Webcam").click().run()
        self.assertTrue(app.session_state["show_camera"])
        self.assertEqual(len(app.get("camera_input")), 1)

        next(button for button in app.button if button.label == "Use Webcam").click().run()
        self.assertFalse(app.session_state["show_camera"])
        self.assertEqual(len(app.get("camera_input")), 0)
        self.assertFalse(app.exception)

    def test_success_filter_and_reset(self) -> None:
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("file_uploader")), 1)

        app.get("file_uploader")[0].upload("page.png", image_bytes(True), "image/png").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["source_name"], "page.png")
        self.assertIsNone(app.session_state["result"])
        self.assertTrue(app.session_state["use_a4_ratio"])

        next(button for button in app.button if button.label == "Scan & rectify").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["result"].success)
        self.assertEqual(app.session_state["result"].detection.method, "contour")
        self.assertIn("Contour selected", app.session_state["diagnostics"])

        app.radio[0].set_value("grayscale").run()
        output = app.session_state["result"].enhanced
        self.assertTrue(np.array_equal(output[:, :, 0], output[:, :, 1]))
        self.assertEqual(app.session_state["rendered_mode"], "grayscale")

        next(button for button in app.button if button.label == "Scan another document").click().run()
        self.assertIsNone(app.session_state["source_bytes"])
        self.assertIsNone(app.session_state["result"])
        self.assertEqual(app.session_state["enhance_mode"], "bw_scan")
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertFalse(app.exception)

    def test_replace_and_detection_failure(self) -> None:
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        app.get("file_uploader")[0].upload("first.png", image_bytes(True), "image/png").run()
        next(button for button in app.button if button.label == "Choose different").click().run()
        app.get("file_uploader")[0].upload("no-page.png", image_bytes(False), "image/png").run()
        self.assertEqual(app.session_state["source_name"], "no-page.png")
        self.assertIsNone(app.session_state["result"])

        next(button for button in app.button if button.label == "Scan & rectify").click().run()
        self.assertFalse(app.session_state["result"].success)
        self.assertIn("Could not reliably detect", app.session_state["result"].message)
        self.assertIn("Contour pass=", app.session_state["diagnostics"])
        self.assertFalse(app.exception)

    def test_invalid_image_has_recovery_action(self) -> None:
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        app.get("file_uploader")[0].upload("broken.jpg", b"not an image", "image/jpeg").run()
        self.assertIsNone(app.session_state["result"])
        self.assertIn("Choose another photo", [button.label for button in app.button])
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
