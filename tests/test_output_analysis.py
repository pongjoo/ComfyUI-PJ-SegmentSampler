import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pj_output_analysis", ROOT / "scripts" / "analyze_segment_outputs.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class OutputAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.before = [torch.zeros(1, 24, 8, 2, 2), torch.zeros(1, 32, 2, 24)]
        self.after = [x.clone() for x in self.before]
        self.after[0][:, :, 3:5] = 0.25
        self.manifest = {
            "baseline": {"seed": 42, "denoise": 0.4, "case": "synthetic contract check"},
            "streams": {
                "video": {"time_dim": 2, "editable_intervals": [[2, 6]], "protected_intervals": [[2, 3], [5, 6]]},
                "audio": {"time_dim": -1, "editable_intervals": []},
            },
            "timeline": {"total_frames": 22, "delivered_intervals": [[0, 10], [10, 22]]},
        }

    def check_report(self, expected="pass", **kwargs):
        result = analysis.analyze_outputs(self.before, self.after, self.manifest, **kwargs)
        self.assertEqual(result["status"], expected, result)
        json.dumps(result, allow_nan=False)
        return result

    def test_local_change_preserves_both_boundaries_and_audio(self):
        result = self.check_report()
        self.assertEqual(result["streams"]["video"]["changed_steps"], [3, 4])
        self.assertTrue(result["streams"]["video"]["protected_exact"])
        self.assertEqual(result["streams"]["audio"]["max_abs_diff"], 0)
        self.assertEqual(result["baseline"], self.manifest["baseline"])

    def test_corrupt_boundary_outside_region_and_audio_each_fail(self):
        for stream, step in ((0, 2), (0, 5), (0, 7), (1, 12)):
            with self.subTest(stream=stream, step=step):
                self.after = [x.clone() for x in self.before]
                axis = 2 if stream == 0 else -1
                self.after[stream].select(axis, step).fill_(1)
                result = self.check_report("fail")
                name = "video" if stream == 0 else "audio"
                self.assertEqual(result["streams"][name]["forbidden_changed_steps"], [step])

    def test_spatial_mask_preserves_pixels_inside_editable_time(self):
        mask = torch.ones(1, 1, 8, 2, 2)
        mask[..., 0, 0] = 0
        self.after[0][..., 0, 0] = 0
        self.check_report(edit_masks=mask)
        self.after[0][0, 0, 3, 0, 0] = 0.1
        result = self.check_report("fail", edit_masks=mask)
        self.assertEqual(result["streams"]["video"]["forbidden_changed_steps"], [3])

    def test_nan_and_infinity_fail_in_baseline_or_candidate(self):
        for side in ("before", "after"):
            for value in (float("nan"), float("inf")):
                with self.subTest(side=side, value=value):
                    self.setUp()
                    getattr(self, side)[0][0, 0, 3, 0, 0] = value
                    result = self.check_report("fail")
                    self.assertFalse(result["streams"]["video"]["finite"])

    def test_shape_and_dtype_must_not_change(self):
        self.after[0] = self.after[0][:, :, :-1]
        self.check_report("fail")
        self.setUp()
        self.after[0] = self.after[0].half()
        self.assertIn("video: dtype changed", self.check_report("fail")["hard_failures"])

    def test_invalid_editable_and_protected_intervals(self):
        cases = ([[2, 9]], [[3, 2]], [[2.0, 4]], [[True, 3]], [[2, 4], [3, 6]], [[4, 6], [2, 3]])
        for field in ("editable_intervals", "protected_intervals"):
            for value in cases:
                with self.subTest(field=field, intervals=value):
                    self.setUp()
                    self.manifest["streams"]["video"][field] = value
                    self.check_report("fail")

    def test_delivery_gaps_overlaps_wrong_order_and_truncation_fail(self):
        for value in ([[0, 9], [10, 22]], [[0, 11], [10, 22]], [[10, 22], [0, 10]], [[0, 21]], []):
            with self.subTest(intervals=value):
                self.manifest["timeline"]["delivered_intervals"] = value
                self.check_report("fail")

    def test_missing_stream_metadata_does_not_silently_skip_audio(self):
        del self.manifest["streams"]["audio"]
        self.check_report("fail")

    def test_invalid_schema_and_nonfinite_configuration_produce_json_failures(self):
        for manifest in ([], {"timeline": []}, {"baseline": {"denoise": float("nan")}}):
            with self.subTest(manifest=manifest):
                result = analysis.analyze_outputs(self.before, self.after, manifest)
                self.assertEqual(result["status"], "fail")
                json.dumps(result, allow_nan=False)

    def test_static_decoded_frames_are_clues_not_duplicate_delivery_failures(self):
        frames = torch.full((22, 4, 6, 3), 0.5)
        result = self.check_report(before_frames=frames, after_frames=frames)
        self.assertEqual(result["quality_clues"]["after"]["identical_frame_runs"], [[0, 22]])
        self.assertEqual(result["quality_clues"]["intensity_delta"], [0.0] * 22)
        self.assertEqual(result["hard_failures"], [])

    def test_decoder_missing_frames_and_nonfinite_values_fail(self):
        self.check_report("fail", after_frames=torch.zeros(21, 4, 6, 3))
        frames = torch.zeros(22, 4, 6, 3)
        frames[3, 1, 1] = float("nan")
        self.check_report("fail", after_frames=frames)

    def test_brightness_jump_is_measurement_not_quality_verdict(self):
        before = torch.zeros(22, 4, 6, 3, dtype=torch.uint8)
        after = before.clone()
        after[11:] = 255
        result = self.check_report(before_frames=before, after_frames=after)
        self.assertEqual(result["quality_clues"]["after"]["adjacent_mean_abs_diff"][10], 1.0)
        self.assertEqual(result["quality_clues"]["intensity_delta"][11], 1.0)
        self.assertEqual(result["quality_clues"]["paired_mean_abs_diff"][11], 1.0)
        self.assertEqual(result["quality_clues"]["after"]["largest_frame_changes"][0]["frame"], 11)

    def test_frames_at_different_scales_are_not_compared_as_quality_change(self):
        before = torch.zeros(22, 4, 6, 3)
        after = torch.zeros(22, 8, 12, 3)
        result = self.check_report(before_frames=before, after_frames=after)
        self.assertNotIn("edge_energy_delta", result["quality_clues"])

    def test_safe_saved_formats_and_cli_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            old, new, manifest, output = [directory / name for name in ("old.pt", "new.pt", "manifest.json", "report.json")]
            torch.save({"samples": self.before}, old)
            torch.save({"video": self.after[0], "audio": self.after[1]}, new)
            manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
            command = ["--before", str(old), "--after", str(new), "--manifest", str(manifest), "--report", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analysis.main(command), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "pass")
            self.after[1].fill_(1)
            torch.save(self.after, new)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analysis.main(command), 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "fail")
            torch.save(SimpleNamespace(samples=self.after), new)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analysis.main(command), 1)
            self.assertIn("Input/read failure", json.loads(output.read_text(encoding="utf-8"))["hard_failures"][0])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "No existing local FFmpeg installation")
    def test_local_video_decoder_preserves_frame_count_without_downloads(self):
        frames = torch.zeros(4, 8, 12, 3, dtype=torch.uint8)
        frames[1, :, :, 0] = 255
        frames[2, :, :, 1] = 255
        frames[3, :, :, 2] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.mkv"
            subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", "12x8", "-r", "4", "-i", "pipe:0", "-c:v", "ffv1", str(path)],
                           input=bytes(frames.flatten().tolist()), capture_output=True, check=True)
            decoded, info = analysis.read_frames(path)
        self.assertTrue(torch.equal(decoded, frames))
        self.assertEqual(info["analysis_size"], [12, 8])


if __name__ == "__main__":
    unittest.main()
