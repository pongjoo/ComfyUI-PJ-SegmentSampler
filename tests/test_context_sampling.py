import contextlib
import io
import unittest
from unittest.mock import patch

import test_segments as fixtures
import test_h3_context as context_fixtures

import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
import comfy.samplers
import comfy.model_sampling

from pj_segments_test.pj_first_pass_sampler import PJ_FirstPass_SegmentSampler
from pj_segments_test.pj_h3_context import PJ_H3SegmentContext
from pj_segments_test.segment_timing import frames_for_tokens, plan_relay_segments, plan_segments, delivery_chunks


class ContextSamplingTests(unittest.TestCase):
    def setUp(self):
        self.node = PJ_FirstPass_SegmentSampler()
        self.guider = fixtures.RecordingGuider()
        self.noise = Noise_RandomNoise(42)
        self.sampler = comfy.samplers.ksampler("euler")
        self.clip = context_fixtures.RecordingClip()
        self.vae = context_fixtures.RecordingVAE()
        self.context = PJ_H3SegmentContext().prepare(self.clip, self.vae, prompt="walk forward",
                                                     segment_prompts='{"2":"turn left"}')[0]
        self.plan = dict(split_mode="按秒自动切分", chunk_seconds=6., overlap_seconds=0.5, fps=24.)
        self.latent = self.empty_latent(252)
        self.callback = patch("latent_preview.prepare_callback", fixtures.callback)
        self.callback.start()

    def tearDown(self):
        self.callback.stop()
        for cache in (self.node._cache, self.node._revisions):
            if cache.directory is not None:
                cache.directory.cleanup()

    def empty_latent(self, tokens):
        audio = round(frames_for_tokens(tokens) * 40 / 24)
        return {"samples": NestedTensor((torch.zeros(1, 24, tokens, 2, 2), torch.zeros(1, 32, 2, audio)))}

    def run_node(self, **kwargs):
        args = dict(noise=self.noise, guider=self.guider, sampler=self.sampler,
                    sigmas=torch.tensor([1., 0.]), latent_image=self.latent,
                    segment_plan=self.plan, generation_mode="22帧接力", h3_context=self.context)
        args.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.node.execute(**args)

    def test_longer_than_thirty_seconds_rebuilds_each_window_and_passes_exact_overlap(self):
        result, _ = self.run_node()
        video, audio = result["samples"].unbind()
        self.assertGreater(frames_for_tokens(video.shape[2]) / 24, 30)
        chunks = plan_relay_segments(video.shape[2], audio.shape[-1], 6.)
        self.assertEqual(len(self.clip.calls), len(chunks))
        self.assertEqual(self.clip.calls[1][0], "walk forward\nturn left")
        self.assertEqual(self.vae.calls, [])
        self.assertNotIn("minimax_keyframes", self.guider.calls[0][2]["positive"][0])
        for index, (call, bounds) in enumerate(zip(self.guider.calls, chunks)):
            self.assertEqual(call[1].unbind()[0].shape[2], bounds[1] - bounds[0])
            self.assertLessEqual(frames_for_tokens(bounds[1] - bounds[0]), 141)
            if index:
                guides = call[2]["positive"][0]["minimax_keyframes"]
                combined = torch.cat([item["latent"] for item in guides], dim=2)
                previous = self.node._cache.load(index - 1, self.node._cache.entries[index - 1])[0].unbind()[0]
                self.assertEqual(combined.shape[2], 7)
                self.assertTrue(torch.equal(combined, previous[:, :, -7:]))
                self.assertEqual([item["resolved_frame_index"] for item in guides], [0, 1, 5, 9, 13, 17, 18])
        self.assertEqual(video.shape, self.latent["samples"].unbind()[0].shape)
        self.assertEqual(audio.shape, self.latent["samples"].unbind()[1].shape)

    def test_reference_video_conditions_match_each_absolute_window_without_double_crop(self):
        self.latent = self.empty_latent(107)
        frames = frames_for_tokens(107)
        reference = torch.arange(frames, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3) / 1000
        self.context = dict(self.context, reference_video=reference)
        result, _ = self.run_node()
        chunks = plan_relay_segments(107, result["samples"].unbind()[1].shape[-1], 6.)
        self.assertEqual(len(self.vae.calls), len(chunks))
        for encoded, bounds, call in zip(self.vae.calls, chunks, self.clip.calls):
            start = frames_for_tokens(bounds[0])
            length = frames_for_tokens(bounds[1] - bounds[0])
            self.assertTrue(torch.allclose(encoded, reference[start:start + length], atol=1 / 255 + 1e-6))
            semantic = call[1]["minimax_ref_items"][0]["data"]
            self.assertTrue(torch.allclose(semantic[0], reference[start], atol=1 / 255 + 1e-6))

    def test_context_change_invalidates_relay_cache_even_when_all_prior_segments_locked(self):
        self.run_node()
        self.guider.calls.clear()
        self.clip.calls.clear()
        self.run_node(relay_action="从指定段重做", relay_start_segment=100)
        self.assertEqual(self.guider.calls, [])
        self.assertEqual(self.clip.calls, [])
        changed = dict(self.context, prompt="walk backward")
        self.run_node(relay_action="从指定段重做", relay_start_segment=100, h3_context=changed)
        self.assertGreater(len(self.guider.calls), 1)
        self.assertEqual(self.clip.calls[0][0], "walk backward")

    def test_normal_segment_sampling_uses_real_short_overlap_and_cascades_dependencies(self):
        self.latent = self.empty_latent(107)
        self.run_node(generation_mode="分段编辑")
        self.assertEqual(len(self.guider.calls[1][2]["positive"][0]["minimax_keyframes"]), 2)
        count = len(self.guider.calls)
        self.guider.calls.clear()
        self.run_node(generation_mode="分段编辑", run_mode="仅重抽第2段")
        self.assertEqual(len(self.guider.calls), count - 1)

    def test_normal_motion_dependency_caches_are_invalidated_before_interruption(self):
        self.latent = self.empty_latent(107)
        self.run_node(generation_mode="分段编辑")
        with patch.object(self.guider, "sample", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.run_node(generation_mode="分段编辑", run_mode="仅重抽第2段")
        self.assertEqual(sorted(self.node._cache.entries), [0])

    def test_local_repair_rebuilds_conditions_without_backward_motion_guides(self):
        self.latent = self.empty_latent(107)
        class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass
        self.guider.model_patcher.get_model_object = lambda name: FlowSampling()
        self.run_node()
        self.clip.calls.clear()
        self.guider.calls.clear()
        result, _ = self.run_node(repair_action="仅重绘目标段", run_mode="仅重抽第2段")
        self.assertEqual(len(self.guider.calls), 1)
        self.assertEqual(self.clip.calls[0][0], "walk forward\nturn left")
        self.assertNotIn("minimax_keyframes", self.guider.calls[0][2]["positive"][0])
        self.assertEqual(result["pj_revision"]["version"], 2)
        self.run_node(repair_action="仅重绘目标段", run_mode="仅重抽第2段")
        self.assertEqual(len(self.guider.calls), 1)
        self.run_node(repair_action="仅重绘目标段", run_mode="仅重抽第2段",
                      h3_context=dict(self.context, prompt="new details"))
        self.assertEqual(len(self.guider.calls), 2)

    def test_without_context_preserves_original_conditioning_and_does_not_encode(self):
        self.latent = self.empty_latent(107)
        self.run_node(h3_context=None)
        self.assertEqual(self.clip.calls, [])
        self.assertEqual(self.vae.calls, [])
        for call in self.guider.calls:
            self.assertTrue(torch.equal(call[2]["positive"][0]["cross_attn"], self.guider.original_conds["positive"][0]["cross_attn"]))
            self.assertNotIn("minimax_keyframes", call[2]["positive"][0])

    def test_lazy_context_is_requested_only_when_sampling(self):
        self.assertIn("h3_context", self.node.check_lazy_status(h3_context=None))
        self.assertNotIn("h3_context", self.node.check_lazy_status(repair_action="载入回修基准", h3_context=None))
        self.assertNotIn("h3_context", self.node.check_lazy_status(repair_action="复用一采结果", h3_context=None))

    def test_zero_overlap_uses_delivery_ownership_without_handoff_or_motion_guides(self):
        self.latent = self.empty_latent(107)
        self.plan = dict(self.plan, overlap_seconds=0)
        result, first = self.run_node(generation_mode="分段编辑", boundary_constraints=True)
        chunks = plan_segments(107, self.latent["samples"].unbind()[1].shape[-1],
                               "按秒自动切分", 6., 0., 3, "", 24.)
        deliveries = delivery_chunks(chunks, self.plan)
        for index, (sample_bounds, delivered, call) in enumerate(zip(chunks, deliveries, self.guider.calls)):
            self.assertIsNone(call[3]["denoise_mask"])
            self.assertNotIn("minimax_keyframes", call[2]["positive"][0])
            cached = self.node._cache.load(index, self.node._cache.entries[index])[0].unbind()
            for stream_index, axis in ((0, 2), (1, 3)):
                start, end = delivered[stream_index * 2:stream_index * 2 + 2]
                sample_start = sample_bounds[stream_index * 2]
                actual = result["samples"].unbind()[stream_index].narrow(axis, start, end - start)
                expected = cached[stream_index].narrow(axis, start - sample_start, end - start)
                self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(first["samples"].unbind()[0].shape[2], chunks[0][1])
        self.assertEqual(result["samples"].unbind()[0].shape[2], 107)

    def test_zero_overlap_local_repair_does_not_modify_right_sampling_padding(self):
        self.latent = self.empty_latent(107)
        self.plan = dict(self.plan, overlap_seconds=0)
        class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass
        self.guider.model_patcher.get_model_object = lambda name: FlowSampling()
        baseline, _ = self.run_node(generation_mode="分段编辑")
        self.guider.calls.clear()
        repaired, _ = self.run_node(generation_mode="分段编辑", repair_action="仅重绘目标段", run_mode="仅重抽第2段")
        original = baseline["samples"].unbind()[0]
        actual = repaired["samples"].unbind()[0]
        self.assertEqual(repaired["pj_revision"]["changes"][-1]["video"], [[42, 78]])
        self.assertTrue(torch.equal(actual[:, :, :42], original[:, :, :42]))
        self.assertTrue(torch.equal(actual[:, :, 78:], original[:, :, 78:]))
        mask = self.guider.calls[0][3]["denoise_mask"].unbind()[0]
        self.assertTrue(torch.all(mask[:, :, -4:] == 0))
        self.assertEqual(len(self.guider.calls), 1)


if __name__ == "__main__":
    unittest.main()
