import contextlib
import io
from pathlib import Path
import tempfile
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
from pj_segments_test.pj_segment_sampler import PJ_SegmentSampler
from pj_segments_test.pj_h3_context import PJ_H3SegmentContext, prepare_segment_conditions, prepare_segment_guider
from pj_segments_test.segment_sampling import fingerprint
from pj_segments_test.segment_timing import plan_relay_segments


class ConditionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.guider = fixtures.RecordingGuider()
        self.clip = context_fixtures.RecordingClip()
        self.context = PJ_H3SegmentContext.prepare(self.clip, prompt="shoe", segment_prompts='{"2":"logo"}')[0]
        self.latent = {"samples": NestedTensor((torch.zeros(1, 24, 107, 2, 2), torch.zeros(1, 32, 2, 603)))}
        self.plan = dict(split_mode="按秒自动切分", chunk_seconds=6., overlap_seconds=.5, fps=24.)
        self.nodes = []
        self.events = []
        self.directories = []
        self.scope = contextlib.ExitStack()
        self.addCleanup(self.scope.close)
        self.scope.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.scope.enter_context(patch("latent_preview.prepare_callback", fixtures.callback))
        original_sample = fixtures.RecordingGuider.sample
        original_encode = self.clip.encode_from_tokens_scheduled
        original_directory = tempfile.TemporaryDirectory

        def sample(guider, *args, **kwargs):
            self.events.append("sample")
            return original_sample(guider, *args, **kwargs)

        def encode(*args, **kwargs):
            self.events.append("encode")
            return original_encode(*args, **kwargs)

        def directory(*args, **kwargs):
            result = original_directory(*args, **kwargs)
            if kwargs.get("prefix") == "pj_conditions_":
                self.directories.append(Path(result.name))
            return result

        self.scope.enter_context(patch.object(fixtures.RecordingGuider, "sample", sample))
        self.scope.enter_context(patch.object(self.clip, "encode_from_tokens_scheduled", encode))
        self.scope.enter_context(patch("tempfile.TemporaryDirectory", directory))
        self.scope.enter_context(patch("comfy.model_management.unload_all_models", side_effect=lambda: self.events.append("unload")))

    def tearDown(self):
        for path in self.directories:
            self.assertFalse(path.exists(), f"Leaked temporary conditioning: {path}")
        for node in self.nodes:
            for name in ("_cache", "_revisions", "_results"):
                store = getattr(node, name, None)
                if store is not None and store.directory is not None:
                    store.directory.cleanup()

    def node(self, cls=PJ_FirstPass_SegmentSampler):
        node = cls()
        self.nodes.append(node)
        return node

    def run_node(self, node, **kwargs):
        args = dict(noise=Noise_RandomNoise(42), guider=self.guider, sampler=comfy.samplers.ksampler("euler"),
                    sigmas=torch.tensor([1., 0.]), latent_image=self.latent,
                    segment_plan=self.plan, h3_context=self.context, unload_model_after_sampling=False)
        args.update(kwargs)
        return node.execute(**args)

    def assert_schedule(self, count):
        self.assertEqual(self.events, ["unload"] + ["encode"] * count + ["unload"] + ["sample"] * count)

    def test_relay_prepares_only_rebuilt_suffix_before_sampling(self):
        node = self.node()
        self.run_node(node, generation_mode="22帧接力")
        self.assert_schedule(3)
        first = node._cache.load(0, node._cache.entries[0])[0]
        self.events.clear()
        self.run_node(node, generation_mode="22帧接力", relay_action="从指定段重做", relay_start_segment=2)
        self.assert_schedule(2)
        self.assertTrue(torch.equal(first.unbind()[0], node._cache.load(0, node._cache.entries[0])[0].unbind()[0]))
        self.events.clear()
        self.run_node(node, generation_mode="22帧接力", relay_action="从指定段重做", relay_start_segment=100)
        self.assertEqual(self.events, [])

    def test_normal_first_pass_prepares_only_unlocked_independent_windows(self):
        node = self.node()
        context = dict(self.context, motion_context=False)
        self.run_node(node, generation_mode="分段编辑", h3_context=context)
        self.assert_schedule(3)
        self.events.clear()
        self.run_node(node, generation_mode="分段编辑", run_mode="仅重抽第2段", h3_context=context)
        self.assert_schedule(1)

    def test_second_pass_prepares_suffix_and_reuses_locked_prefix(self):
        node = self.node(PJ_SegmentSampler)
        self.run_node(node)
        self.assert_schedule(3)
        self.events.clear()
        self.run_node(node, refine_action="常规精修", run_mode="仅精修第2段")
        self.assert_schedule(2)

    def test_multiple_local_repairs_prepare_first_and_preserve_unselected_region(self):
        class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass
        self.guider.model_patcher.get_model_object = lambda name: FlowSampling()
        node = self.node()
        before, _ = self.run_node(node, generation_mode="22帧接力")
        self.events.clear()
        result, _ = self.run_node(node, generation_mode="22帧接力", repair_action="仅重绘目标段",
                                  run_mode="自定义指定分段(下方输入)", custom_rerun_segments="1,3")
        self.assert_schedule(2)
        old, new = before["samples"].unbind(), result["samples"].unbind()
        self.assertTrue(torch.equal(old[1], new[1]))
        self.assertTrue(torch.equal(old[0][:, :, 40:72], new[0][:, :, 40:72]))

    def test_prepared_conditions_match_direct_conditions_and_keep_dtype_and_policy(self):
        chunks = plan_relay_segments(107, 603, 6.)
        video = self.latent["samples"].unbind()[0]
        tail = torch.ones(1, 24, 7, 2, 2)
        self.context = dict(self.context, vae=context_fixtures.RecordingVAE(),
                            ref_videos={"ref_video_0": torch.linspace(0, 1, 362).view(-1, 1, 1, 1).expand(-1, 32, 32, 3)})
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype), patch.object(self.clip, "encode_from_tokens_scheduled",
                    return_value=[[torch.ones(1, 3, 4, dtype=dtype), {"minimax_token_tags": torch.arange(3)}]]):
                direct = prepare_segment_guider(self.guider, self.context, chunks[1], (1, 24, 42, 2, 2), tail, 1)
                with prepare_segment_conditions(self.context, chunks, video.shape, [1]) as prepared:
                    actual = prepared.guider(self.guider, 1, tail)
                    self.assertEqual(fingerprint(actual.original_conds), fingerprint(direct.original_conds))
                    self.assertEqual(actual.original_conds["positive"][0]["cross_attn"].dtype, dtype)
                    self.assertIs(actual.model_patcher, self.guider.model_patcher)

    def test_no_context_or_no_pending_segments_does_not_unload_models(self):
        for context, indices in ((None, [0]), (self.context, [])):
            with prepare_segment_conditions(context, [(0, 107, 0, 603)], (1, 24, 107, 2, 2), indices) as prepared:
                self.assertIsNone(prepared)
        self.assertEqual(self.events, [])
        self.assertEqual(self.directories, [])

    def test_encoding_failure_does_not_sample_and_removes_all_condition_files(self):
        node = self.node()
        self.run_node(node, generation_mode="22帧接力")
        version = node._revisions.metadata["version"]
        self.events.clear()
        with patch.object(self.clip, "encode_from_tokens_scheduled", side_effect=RuntimeError("encode failed")):
            with self.assertRaisesRegex(RuntimeError, "encode failed"):
                self.run_node(node, generation_mode="22帧接力", relay_action="从指定段重做", relay_start_segment=2)
        self.assertNotIn("sample", self.events)
        self.assertEqual(self.events, ["unload", "unload"])
        self.assertEqual(list(node._cache.entries), [0])
        self.assertEqual(node._revisions.metadata["version"], version)

    def test_sampling_failure_removes_prepared_files(self):
        with patch.object(fixtures.RecordingGuider, "sample", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.run_node(self.node(), generation_mode="22帧接力")
        self.assertEqual(self.events, ["unload", "encode", "encode", "encode", "unload"])


if __name__ == "__main__":
    unittest.main()
