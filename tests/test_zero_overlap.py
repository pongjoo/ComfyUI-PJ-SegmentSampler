import contextlib
import io
import unittest
from unittest.mock import patch

import torch

from test_segments import PJ_SegmentPlanner, PJ_SegmentSampler, RecordingGuider, callback
import comfy.samplers
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from pj_segments_test.segment_sampling import blend_sequence
from pj_segments_test.segment_timing import (delivery_chunks, frames_for_tokens, plan_segments,
                                           tokens_for_frames, tokens_for_frames_floor, zero_overlap)


class ZeroOverlapTests(unittest.TestCase):
    def plan(self, total=107, mode="按秒自动切分", seconds=6, count=3, overlap=0):
        audio = round(frames_for_tokens(total) * 40 / 24)
        rules = dict(split_mode=mode, overlap_seconds=overlap)
        chunks = plan_segments(total, audio, mode, seconds, overlap, count, "0-6,6-12,12-30")
        return chunks, delivery_chunks(chunks, rules), rules, audio

    def test_zero_delivery_covers_timeline_without_illegal_sampling_windows(self):
        for total in (2, 7, 12, 17, 37, 107, 212):
            for mode in ("按秒自动切分", "固定分段数"):
                for count in (1, 3, 50):
                    for seconds in (2, 6, 15):
                        with self.subTest(total=total, mode=mode, count=count, seconds=seconds):
                            sampled, delivered, _, audio = self.plan(total, mode, seconds, count)
                            self.assertEqual(delivered[0][::2], (0, 0))
                            self.assertEqual((delivered[-1][1], delivered[-1][3]), (total, audio))
                            self.assertEqual(sum(end - start for start, end, _, _ in delivered), total)
                            self.assertEqual(sum(end - start for _, _, start, end in delivered), audio)
                            for sample, output in zip(sampled, delivered):
                                self.assertEqual(sample[0] % 5, 0)
                                self.assertEqual((sample[1] - sample[0]) % 5, 2)
                                self.assertTrue(sample[0] == output[0] < output[1] <= sample[1])
                            for left, right in zip(delivered, delivered[1:]):
                                self.assertEqual(left[1], right[0])
                                self.assertEqual(left[3], right[2])

    def test_zero_is_distinct_from_nearest_positive_overlap_and_custom_ranges(self):
        sampled, delivered, rules, _ = self.plan()
        self.assertTrue(zero_overlap(rules))
        self.assertEqual([part[:2] for part in sampled], [(0, 42), (40, 82), (80, 107)])
        self.assertEqual([part[:2] for part in delivered], [(0, 40), (40, 80), (80, 107)])
        for mode, overlap in (("按秒自动切分", 0.5), ("固定分段数", 1.5), ("自定义切点", 0)):
            sampled, delivered, rules, _ = self.plan(mode=mode, overlap=overlap)
            self.assertFalse(zero_overlap(rules))
            self.assertEqual(sampled, delivered)
        self.assertFalse(zero_overlap(None))
        self.assertFalse(zero_overlap({}))

    def test_padding_is_not_written_to_video_or_audio_for_any_blend_mode(self):
        sampled, delivered, _, audio = self.plan()
        for is_audio, total, offset in ((False, 107, 0), (True, audio, 2)):
            axis = 3 if is_audio else 2
            shape = (1, 2, 2, total) if is_audio else (1, 2, total, 2, 2)
            original = torch.full(shape, -1., dtype=torch.bfloat16)
            expected = original.clone()
            chunks = []
            for index, (sample, output) in enumerate(zip(sampled, delivered)):
                start, end = sample[offset:offset + 2]
                left, right = output[offset:offset + 2]
                chunk_shape = list(shape)
                chunk_shape[axis] = end - start
                chunk = torch.full(chunk_shape, float(index + 1), dtype=original.dtype)
                if right < end:
                    chunk.narrow(axis, right - start, end - right).fill_(999)
                chunks.append(chunk)
                expected.narrow(axis, left, right - left).fill_(index + 1)
            for mode in ("余弦平滑(推荐)", "线性过渡", "硬切", "保留边界"):
                result = blend_sequence(chunks, [part[offset:offset + 2] for part in sampled], original,
                                        mode, is_audio=is_audio,
                                        delivery_intervals=[part[offset:offset + 2] for part in delivered])
                self.assertTrue(torch.equal(result, expected))
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(torch.all(original == -1))

    def test_invalid_delivery_windows_fail_instead_of_silent_crop(self):
        original = torch.zeros(1, 1, 7, 1, 1)
        with self.assertRaisesRegex(ValueError, "数量"):
            blend_sequence([original], [(0, 7)], original, "硬切", delivery_intervals=[])
        with self.assertRaisesRegex(ValueError, "位于"):
            blend_sequence([original], [(0, 7)], original, "硬切", delivery_intervals=[(0, 8)])

    def test_single_trimmed_window_does_not_retain_padding_storage(self):
        source = torch.ones(1, 1, 12, 2, 2)
        original = torch.zeros(1, 1, 7, 2, 2)
        result = blend_sequence([source], [(0, 12)], original, "硬切", delivery_intervals=[(0, 7)])
        self.assertEqual(result.untyped_storage().nbytes(), result.numel() * result.element_size())
        source.fill_(999)
        self.assertTrue(torch.all(result == 1))

    def test_planner_displays_delivery_zero_and_keeps_sampling_padding_explicit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            output = PJ_SegmentPlanner().plan(chunk_seconds=6, overlap_seconds=0)
        plan = output["result"][0]
        preview = output["ui"]["preview_data"][0]
        self.assertTrue(plan["zero_overlap"])
        self.assertTrue(preview["zero_overlap"])
        self.assertEqual(plan["chunks"][0][:2], (0, 42))
        self.assertEqual(plan["delivery_chunks"][0][:2], (0, 40))
        self.assertEqual([part["overlap_frames"] for part in preview["chunks"]], [0, 0, 0])
        self.assertEqual([part["padding_frames"] for part in preview["chunks"]], [5, 5, 0])
        self.assertEqual(preview["chunks"][0]["sample_end_frame"], 141)
        self.assertEqual(preview["chunks"][0]["end_frame"], 136)
        self.assertIn("右侧填充将裁掉", preview["timing_note"])
        for left, right in zip(preview["chunks"], preview["chunks"][1:]):
            self.assertEqual(left["end_frame"], right["start_frame"])

    def test_second_pass_does_not_inherit_or_deliver_padding(self):
        sampled, delivered, rules, audio = self.plan()

        class PaddingGuider(RecordingGuider):
            def sample(self, noise, latent, sampler, sigmas, **kwargs):
                super().sample(noise, latent, sampler, sigmas, **kwargs)
                index = len(self.calls) - 1
                parts = [torch.full_like(part, float(index + 1)) for part in latent.unbind()]
                for stream_index, axis in ((0, 2), (1, 3)):
                    end_index = stream_index * 2 + 1
                    padding = sampled[index][end_index] - delivered[index][end_index]
                    if padding:
                        parts[stream_index].narrow(axis, parts[stream_index].shape[axis] - padding, padding).fill_(999)
                return NestedTensor(parts)

        latent = {"samples": NestedTensor((torch.zeros(1, 24, 107, 2, 2), torch.zeros(1, 32, 2, audio)))}
        guider = PaddingGuider()
        node = PJ_SegmentSampler()
        with patch("latent_preview.prepare_callback", callback), contextlib.redirect_stdout(io.StringIO()):
            result, _ = node.execute(noise=Noise_RandomNoise(42), guider=guider,
                                     sampler=comfy.samplers.ksampler("euler"), sigmas=torch.tensor([1., 0.]),
                                     latent_image=latent, segment_plan=dict(rules, chunk_seconds=6),
                                     keep_original_audio=False, unload_model_after_sampling=False)
        for cache in (node._cache, node._results):
            if cache.directory is not None:
                self.addCleanup(cache.directory.cleanup)
        for index, (bounds, call) in enumerate(zip(delivered, guider.calls)):
            self.assertIsNone(call[3]["denoise_mask"])
            self.assertTrue(torch.all(call[1].unbind()[0] == 0))
            for stream_index, axis in ((0, 2), (1, 3)):
                start, end = bounds[stream_index * 2:stream_index * 2 + 2]
                output = result["samples"].unbind()[stream_index].narrow(axis, start, end - start)
                self.assertTrue(torch.all(output == index + 1))

    def test_planner_reference_total_floors_without_changing_condition_crop_helper(self):
        self.assertEqual(frames_for_tokens(tokens_for_frames_floor(322)), 311)
        self.assertEqual(frames_for_tokens(tokens_for_frames(322)), 328)
        for frames, fps, expected in ((322, 24, 311), (360, 24, 345), (450, 30, 345), (1, 24, 5)):
            with self.subTest(frames=frames, fps=fps):
                with patch("pj_segments_test.pj_segment_planner._generate_preview_mp4", return_value=None):
                    with contextlib.redirect_stdout(io.StringIO()):
                        output = PJ_SegmentPlanner().plan(images=torch.zeros(frames, 2, 2, 3), fps=fps)
                plan = output["result"][0]
                self.assertEqual(plan["total_frames"], expected)
                self.assertEqual(plan["length_policy"], "floor_h3")
                if frames < 5:
                    self.assertIn("最小5帧", output["ui"]["preview_data"][0]["timing_note"])
        with contextlib.redirect_stdout(io.StringIO()):
            reference = PJ_SegmentPlanner().plan()["result"][0]
            existing = PJ_SegmentPlanner().plan(latent={"samples": torch.zeros(1, 24, 107, 2, 2)})["result"][0]
        self.assertEqual(reference["total_frames"], 345)
        self.assertEqual(existing["total_frames"], 362)
        self.assertEqual(existing["length_policy"], "existing_latent")


if __name__ == "__main__":
    unittest.main()
