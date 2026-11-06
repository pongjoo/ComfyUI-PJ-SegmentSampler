import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pj_validation_media", ROOT / "scripts" / "validate_h3_repair.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
validation.torch = torch


class ValidationMediaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.options = SimpleNamespace(reference_video=None, reference_video_start=0.,
                                       reference_image=[], frames=5, width=32, height=32)

    def test_product_images_preserve_order_and_pixels(self):
        for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
            path = self.root / f"ref_{index}.png"
            Image.new("RGB", (32, 48), color).save(path)
            self.options.reference_image.append(str(path))
        video, images = validation.reference_inputs(self.options)
        self.assertIsNone(video)
        self.assertEqual(images.shape, (2, 48, 32, 3))
        self.assertTrue(torch.equal(images[:, 0, 0], torch.tensor([[1., 0., 0.], [0., 1., 0.]])))

    def test_mismatched_reference_batch_has_actionable_error(self):
        for index, size in enumerate(((32, 32), (48, 32))):
            path = self.root / f"ref_{index}.png"
            Image.new("RGB", size).save(path)
            self.options.reference_image.append(str(path))
        with self.assertRaisesRegex(ValueError, "share dimensions"):
            validation.reference_inputs(self.options)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_video_window_uses_requested_start_and_rejects_short_source(self):
        frames = np.zeros((60, 32, 32, 3), dtype=np.uint8)
        frames[:30, :, :, 0] = 255
        frames[30:, :, :, 1] = 255
        path = self.root / "reference.mkv"
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                        "-video_size", "32x32", "-framerate", "30", "-i", "pipe:0", "-c:v", "ffv1", str(path)],
                       input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        self.options.reference_video = str(path)
        self.options.reference_video_start = 1.
        video, images = validation.reference_inputs(self.options)
        self.assertIsNone(images)
        self.assertEqual(video.shape, (5, 32, 32, 3))
        self.assertTrue(torch.all(video[..., 1] == 1))
        self.assertTrue(torch.all(video[..., (0, 2)] == 0))
        self.options.frames = 73
        with self.assertRaisesRegex(ValueError, "does not cover 73 frames"):
            validation.reference_inputs(self.options)


if __name__ == "__main__":
    unittest.main()
