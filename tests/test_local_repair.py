import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import test_segments as fixtures

import torch
import comfy.model_sampling
import comfy.samplers
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

from pj_segments_test.pj_first_pass_sampler import PJ_FirstPass_SegmentSampler
from pj_segments_test.segment_local_repair import repair_segments, repair_sigmas
from pj_segments_test.segment_revision import RevisionStore, changed_intervals
from pj_segments_test.segment_timing import plan_segments


class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


class LocalRepairTests(unittest.TestCase):
    def setUp(self):
        self.node = PJ_FirstPass_SegmentSampler()
        self.guider = fixtures.RecordingGuider()
        self.flow = FlowSampling()
        self.guider.model_patcher.get_model_object = lambda name: self.flow
        self.noise = Noise_RandomNoise(123)
        self.sampler = comfy.samplers.ksampler("euler")
        self.video = torch.arange(107, dtype=torch.float32).reshape(1, 1, 107, 1, 1).expand(1, 24, 107, 2, 2).clone()
        self.audio = torch.randn(1, 32, 2, 603)
        self.baseline = {"samples": NestedTensor((self.video, self.audio))}
        self.empty = {"samples": NestedTensor((torch.zeros_like(self.video), torch.zeros_like(self.audio)))}
        self.plan = dict(split_mode="按秒自动切分", chunk_seconds=6., overlap_seconds=0.5, fps=24.)
        self.chunks = plan_segments(107, 603, "按秒自动切分", 6., 0.5, 3, "", 24.)
        self.callback_patch = patch("latent_preview.prepare_callback", fixtures.callback)
        self.callback_patch.start()

    def tearDown(self):
        self.callback_patch.stop()
        for cache in (self.node._cache, self.node._revisions):
            if cache.directory is not None:
                cache.directory.cleanup()

    def run_node(self, **kwargs):
        args = dict(noise=self.noise, guider=self.guider, sampler=self.sampler,
                    sigmas=torch.tensor([1., 0.]), latent_image=self.empty,
                    segment_plan=self.plan, repair_action="仅重绘目标段", run_mode="仅重抽第2段",
                    repair_baseline=self.baseline)
        args.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.node.execute(**args)

    def helper(self, **kwargs):
        args = dict(noise=self.noise, guider=self.guider, sampler=self.sampler,
                    sigmas=torch.tensor([0.4, 0.]), baseline=self.baseline, chunks=self.chunks,
                    target_indices=[1], seed=123)
        args.update(kwargs)
        return repair_segments(**args)

    def test_old_result_initializes_only_selected_window_and_preserves_both_sides(self):
        result, first = self.run_node()
        self.assertEqual(len(self.guider.calls), 1)
        start, end = self.chunks[1][:2]
        actual = result["samples"].unbind()
        self.assertTrue(torch.equal(self.guider.calls[0][1].unbind()[0], self.video[:, :, start:end]))
        self.assertTrue(torch.equal(actual[0][:, :, :start + 2], self.video[:, :, :start + 2]))
        self.assertTrue(torch.equal(actual[0][:, :, end - 2:], self.video[:, :, end - 2:]))
        self.assertFalse(torch.equal(actual[0][:, :, start + 2:end - 2], self.video[:, :, start + 2:end - 2]))
        self.assertTrue(torch.equal(actual[1], self.audio))
        self.assertNotIn("noise_mask", result)
        self.assertEqual(first["samples"].unbind()[0].shape[2], self.chunks[0][1])
        self.assertNotIn("pj_revision", first)
        self.assertEqual(result["pj_revision"]["changes"][-1]["video"], [[start + 2, end - 2]])

    def test_scheduler_matches_standard_denoise_contract(self):
        actual = repair_sigmas(self.guider, "simple", 6, 0.4)
        expected = comfy.samplers.calculate_sigmas(self.flow, "simple", 15).cpu()[-7:]
        self.assertTrue(torch.equal(actual, expected))
        self.assertNotEqual(actual[0].item(), 0.4)

    def test_zero_denoise_skips_sampling_and_does_not_advance_version(self):
        initial, _ = self.run_node(repair_action="复用一采结果")
        result, _ = self.run_node(repair_denoise=0)
        self.assertEqual(self.guider.calls, [])
        self.assertEqual(initial["pj_revision"], result["pj_revision"])
        self.assertTrue(torch.equal(result["samples"].unbind()[0], self.video))

    def test_missing_baseline_fails_without_sampling(self):
        with self.assertRaisesRegex(ValueError, "没有已成功"):
            self.run_node(repair_baseline=None)
        self.assertEqual(self.guider.calls, [])

    def test_same_request_is_idempotent_and_new_request_uses_latest_result(self):
        result, _ = self.run_node()
        retry, _ = self.run_node()
        self.assertEqual(len(self.guider.calls), 1)
        self.assertEqual(retry["pj_revision"], result["pj_revision"])
        later, _ = self.run_node(repair_request_id=1)
        self.assertEqual(len(self.guider.calls), 2)
        start, end = self.chunks[1][:2]
        self.assertTrue(torch.equal(self.guider.calls[1][1].unbind()[0], result["samples"].unbind()[0][:, :, start:end]))
        self.assertEqual(later["pj_revision"]["version"], result["pj_revision"]["version"] + 1)

    def test_changes_accumulate_across_local_repairs(self):
        original, _ = self.run_node(repair_action="复用一采结果")
        left, _ = self.run_node(run_mode="仅重抽第1段")
        last = len(self.chunks)
        right, _ = self.run_node(run_mode=f"仅重抽第{last}段")
        expected = [[0, self.chunks[0][1] - 2], [self.chunks[-1][0] + 2, 107]]
        self.assertEqual(changed_intervals(right["pj_revision"], original["pj_revision"]), expected)
        self.assertEqual(changed_intervals(right["pj_revision"], left["pj_revision"]), expected[1:])
        self.assertEqual(changed_intervals(right["pj_revision"], right["pj_revision"]), [])
        self.assertEqual(changed_intervals(right["pj_revision"]), [[0, 107]])

    def test_failed_candidate_keeps_last_complete_result(self):
        successful, _ = self.run_node()
        with patch.object(self.guider, "sample", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_node(repair_request_id=1)
        current, _ = self.node._revisions.load()
        self.assertEqual(current["pj_revision"], successful["pj_revision"])
        self.assertTrue(torch.equal(current["samples"].unbind()[0], successful["samples"].unbind()[0]))

    def test_atomic_snapshot_failure_does_not_replace_old_version(self):
        successful, _ = self.run_node()
        with patch("pj_segments_test.segment_revision.os.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_node(repair_request_id=1)
        current, _ = self.node._revisions.load()
        self.assertEqual(current["pj_revision"], successful["pj_revision"])
        self.assertEqual(self.node._revisions.metadata, successful["pj_revision"])
        self.assertFalse(list(Path(self.node._revisions.directory.name).glob("*.pending")))

    def test_spatial_mask_has_priority_and_is_not_replaced_by_temporal_mask(self):
        mask = torch.ones(1, 1, 107, 2, 2)
        mask[..., 0] = 0
        masked = dict(self.baseline, noise_mask=mask)
        result, _ = self.run_node(repair_baseline=masked)
        actual = result["samples"].unbind()[0]
        self.assertTrue(torch.equal(actual[..., 0], self.video[..., 0]))
        self.assertTrue(torch.equal(result["noise_mask"], mask))
        self.assertTrue(torch.equal(mask, masked["noise_mask"]))

    def test_adjacent_targets_protect_only_outer_boundaries(self):
        result, _, changes = self.helper(target_indices=[0, 1])
        self.assertEqual(changes, [[0, self.chunks[1][1] - 2]])
        actual = result["samples"].unbind()[0]
        seam = self.chunks[1][0]
        self.assertFalse(torch.equal(actual[:, :, seam:seam + 2], self.video[:, :, seam:seam + 2]))
        self.assertTrue(torch.equal(actual[:, :, self.chunks[1][1] - 2:], self.video[:, :, self.chunks[1][1] - 2:]))

    def test_new_initial_latent_updates_only_requested_range_and_keeps_x0(self):
        source = {"samples": NestedTensor((torch.full_like(self.video, -9, dtype=torch.bfloat16), self.audio))}
        previous_x0 = {"samples": NestedTensor((self.video + 1000.123, self.audio))}
        start, end = self.chunks[1][:2]
        result, denoised, changes = self.helper(initial_latent=source, denoised_baseline=previous_x0,
                                                editable_ranges=[(start + 1, end - 1)])
        entered = self.guider.calls[0][1].unbind()[0]
        self.assertTrue(torch.all(entered[:, :, 1:-1] == -9))
        self.assertEqual(entered.dtype, torch.float32)
        self.assertTrue(torch.equal(entered[:, :, :1], self.video[:, :, start:start + 1]))
        output = result["samples"].unbind()[0]
        x0 = denoised["samples"].unbind()[0]
        self.assertTrue(torch.equal(x0[:, :, :start + 1], previous_x0["samples"].unbind()[0][:, :, :start + 1]))
        self.assertTrue(torch.equal(x0[:, :, start + 1:end - 1], output[:, :, start + 1:end - 1] * 2))
        self.assertEqual(changes, [[start + 1, end - 1]])

    def test_real_ersde_mask_adapter_preserves_both_boundaries(self):
        guider = fixtures.NativeMaskGuider()
        sampler = comfy.samplers.ksampler("er_sde")
        result, _, _ = self.helper(guider=guider, sampler=sampler)
        start, end = self.chunks[1][:2]
        output = result["samples"].unbind()
        self.assertTrue(torch.equal(output[0][:, :, :start + 2], self.video[:, :, :start + 2]))
        self.assertTrue(torch.equal(output[0][:, :, end - 2:], self.video[:, :, end - 2:]))
        self.assertTrue(torch.equal(output[1], self.audio))

    def test_successful_legacy_generation_becomes_repair_baseline(self):
        original, _ = self.run_node(repair_action="原采样方式", repair_baseline=None)
        self.assertIsNotNone(self.node._revisions.load())
        self.guider.calls.clear()
        result, _ = self.run_node(repair_baseline=None)
        self.assertEqual(len(self.guider.calls), 1)
        start, end = self.chunks[1][:2]
        self.assertTrue(torch.equal(self.guider.calls[0][1].unbind()[0], original["samples"].unbind()[0][:, :, start:end]))
        self.assertEqual(original["pj_revision"]["lineage"], result["pj_revision"]["lineage"])

    def test_nonfinite_candidate_is_rejected_without_replacing_saved_result(self):
        original, _ = self.run_node(repair_action="复用一采结果")
        sample = self.guider.sample
        def invalid(*args, **kwargs):
            return sample(*args, **kwargs) * float("nan")
        with patch.object(self.guider, "sample", side_effect=invalid):
            with self.assertRaisesRegex(ValueError, "非有限值"):
                self.run_node()
        self.assertEqual(self.node._revisions.load()[0]["pj_revision"], original["pj_revision"])

    def test_bounded_change_history_falls_back_to_full_refresh(self):
        store = RevisionStore()
        try:
            original, _ = store.commit(self.baseline, changes=[(0, 107)])
            for index in range(130):
                latest, _ = store.commit(self.baseline, changes=[(index % 100, index % 100 + 1)])
            self.assertEqual(len(latest["pj_revision"]["changes"]), 128)
            self.assertEqual(changed_intervals(latest["pj_revision"], original["pj_revision"]), [[0, 107]])
            self.assertEqual(len(list(Path(store.directory.name).iterdir())), 1)
        finally:
            if store.directory is not None:
                store.directory.cleanup()

    def test_lazy_inputs_request_only_the_selected_action_dependencies(self):
        pending = dict(noise=None, guider=None, sampler=None, sigmas=None,
                       latent_image=None, repair_baseline=None)
        self.assertEqual(self.node.check_lazy_status(**pending), ["noise", "guider", "sampler", "sigmas", "latent_image"])
        self.assertEqual(self.node.check_lazy_status(repair_action="复用一采结果", **pending), ["repair_baseline"])
        self.assertEqual(self.node.check_lazy_status(repair_action="载入回修基准", **pending), ["repair_baseline"])
        self.run_node(repair_action="载入回修基准")
        self.assertEqual(self.node.check_lazy_status(repair_action="复用一采结果", **pending), [])
        self.assertEqual(self.node.check_lazy_status(repair_action="仅重绘目标段", **pending), ["noise", "guider", "sampler"])
        self.assertEqual(self.node.check_lazy_status(repair_action="仅重绘目标段", repair_denoise=0, **pending), [])
        reused, _ = self.run_node(repair_action="复用一采结果", noise=None, guider=None, sampler=None,
                                  sigmas=None, latent_image=None, repair_baseline=None)
        self.assertTrue(torch.equal(reused["samples"].unbind()[0], self.video))

    def test_existing_result_ignores_new_external_input_until_explicit_load(self):
        edited, _ = self.run_node()
        changed_source = {"samples": NestedTensor((self.video + 123, self.audio))}
        reused, _ = self.run_node(repair_action="复用一采结果", repair_baseline=changed_source)
        self.assertEqual(reused["pj_revision"], edited["pj_revision"])
        loaded, _ = self.run_node(repair_action="载入回修基准", repair_baseline=changed_source)
        self.assertNotEqual(loaded["pj_revision"]["lineage"], edited["pj_revision"]["lineage"])
        self.assertTrue(torch.equal(loaded["samples"].unbind()[0], changed_source["samples"].unbind()[0]))

    def test_reuse_and_load_do_not_request_planner_or_recompute_secondary(self):
        pending = dict(segment_plan=None)
        self.assertEqual(self.node.check_lazy_status(**pending), ["segment_plan"])
        self.assertEqual(self.node.check_lazy_status(repair_action="仅重绘目标段", **pending), ["segment_plan"])
        self.assertEqual(self.node.check_lazy_status(repair_action="载入回修基准", **pending), [])
        self.assertEqual(self.node.check_lazy_status(repair_action="复用一采结果", **pending), [])
        loaded, imported_secondary = self.run_node(repair_action="载入回修基准", segment_plan=None)
        self.assertEqual(imported_secondary["samples"].unbind()[0].shape[2], 107)
        sampled, first = self.run_node(repair_request_id=1)
        with patch("pj_segments_test.pj_first_pass_sampler.resolve_plan", side_effect=AssertionError("planner requested")):
            reused, secondary = self.run_node(repair_action="复用一采结果", segment_plan=None,
                                              noise=None, guider=None, sampler=None, sigmas=None, latent_image=None)
        self.assertEqual(reused["pj_revision"], sampled["pj_revision"])
        self.assertTrue(torch.equal(secondary["samples"].unbind()[0], first["samples"].unbind()[0]))


if __name__ == "__main__":
    unittest.main()
