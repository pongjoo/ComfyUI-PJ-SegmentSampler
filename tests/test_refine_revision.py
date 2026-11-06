import contextlib
import io
import unittest
from unittest.mock import patch

from test_segments import RecordingGuider, callback
import torch
import comfy.samplers
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from pj_segments_test.pj_segment_sampler import PJ_SegmentSampler
from pj_segments_test.segment_revision import RevisionStore
from pj_segments_test.segment_timing import resolve_plan


class RefineRevisionTests(unittest.TestCase):
    def setUp(self):
        self.node = PJ_SegmentSampler()
        self.source = RevisionStore()
        self.guider = RecordingGuider()
        self.noise = Noise_RandomNoise(42)
        self.input = {"samples": NestedTensor((torch.zeros(1, 24, 107, 2, 2), torch.zeros(1, 32, 2, 603)))}
        self.plan = dict(split_mode="按秒自动切分", chunk_seconds=6., overlap_seconds=1.5, fps=24.)
        self.args = dict(noise=self.noise, guider=self.guider, sampler=comfy.samplers.ksampler("euler"),
                         sigmas=torch.tensor([0.4, 0.]), latent_image=self.input,
                         segment_plan=self.plan, unload_model_after_sampling=False)
        self.preview = patch("latent_preview.prepare_callback", callback)
        self.preview.start()

    def tearDown(self):
        self.preview.stop()
        for store in (self.source, self.node._cache, self.node._results):
            if store.directory is not None:
                store.directory.cleanup()

    def run_node(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.node.execute(**(self.args | kwargs))

    def test_local_refine_preserves_both_neighbours_and_denoised(self):
        baseline, x0 = self.run_node()
        self.guider.calls.clear()
        output, new_x0 = self.run_node(refine_action="仅重做目标段", run_mode="仅精修第2段", noise=Noise_RandomNoise(51))
        self.assertEqual(len(self.guider.calls), 1)
        chunks, _ = resolve_plan(self.input["samples"], self.plan, "按秒自动切分", 6, 1.5, 3, "", 24, False)
        start, end = chunks[1][0] + 2, chunks[1][1] - 2
        for before, after in ((baseline, output), (x0, new_x0)):
            a, b = before["samples"].unbind()[0], after["samples"].unbind()[0]
            self.assertTrue(torch.equal(a[:, :, :start], b[:, :, :start]))
            self.assertTrue(torch.equal(a[:, :, end:], b[:, :, end:]))
            self.assertFalse(torch.equal(a[:, :, start:end], b[:, :, start:end]))
        self.assertTrue(torch.equal(output["samples"].unbind()[1], baseline["samples"].unbind()[1]))

    def test_local_refine_requires_a_completed_result(self):
        with self.assertRaisesRegex(ValueError, "没有已完成"):
            self.run_node(refine_action="仅重做目标段")
        self.assertEqual(self.guider.calls, [])

    def test_repeated_request_and_reuse_do_not_sample(self):
        self.run_node()
        args = dict(refine_action="仅重做目标段", run_mode="仅精修第2段", refine_request_id=1)
        result = self.run_node(**args)
        self.guider.calls.clear()
        repeated = self.run_node(**args)
        reused = self.run_node(refine_action="复用高清结果")
        self.assertEqual(self.guider.calls, [])
        self.assertEqual(result[0]["pj_revision"], repeated[0]["pj_revision"])
        self.assertEqual(result[0]["pj_revision"], reused[0]["pj_revision"])

    def test_accumulated_source_changes_are_consumed_once(self):
        original, _ = self.source.commit(self.input, changes=[(0, 107)])
        baseline, _ = self.run_node(source_result=original, refine_action="更新一采变更")
        self.source.commit(self.input, changes=[(12, 17)])
        latest, _ = self.source.commit(self.input, changes=[(83, 88)])
        self.guider.calls.clear()
        result, _ = self.run_node(source_result=latest, refine_action="更新一采变更", noise=Noise_RandomNoise(52))
        self.assertGreater(len(self.guider.calls), 0)
        old_v = baseline["samples"].unbind()[0]
        new_v = result["samples"].unbind()[0]
        allowed = torch.zeros(107, dtype=torch.bool)
        allowed[12:17] = True
        allowed[83:88] = True
        self.assertTrue(torch.equal(old_v[:, :, ~allowed], new_v[:, :, ~allowed]))
        self.assertFalse(torch.equal(old_v[:, :, 12:17], new_v[:, :, 12:17]))
        self.assertFalse(torch.equal(old_v[:, :, 83:88], new_v[:, :, 83:88]))
        self.assertEqual(result["pj_refine_source"]["version"], 3)
        self.guider.calls.clear()
        self.run_node(source_result=latest, refine_action="更新一采变更")
        self.assertEqual(self.guider.calls, [])

    def test_failed_update_keeps_old_high_resolution_and_retries(self):
        first, _ = self.source.commit(self.input, changes=[(0, 107)])
        previous = self.run_node(source_result=first)
        latest, _ = self.source.commit(self.input, changes=[(10, 20)])
        with patch.object(self.guider, "sample", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.run_node(source_result=latest, refine_action="更新一采变更")
        stored = self.node._results.load()
        self.assertEqual(stored[0]["pj_revision"], previous[0]["pj_revision"])
        self.assertEqual(stored[0]["pj_refine_source"]["version"], 1)
        result = self.run_node(source_result=latest, refine_action="更新一采变更")
        self.assertEqual(result[0]["pj_refine_source"]["version"], 2)

    def test_external_upscaler_metadata_loss_uses_explicit_source(self):
        source, _ = self.source.commit(self.input, changes=[(0, 107)])
        self.assertNotIn("pj_revision", self.input)
        result = self.run_node(source_result=source)
        self.assertEqual(result[0]["pj_refine_source"]["lineage"], source["pj_revision"]["lineage"])
        with self.assertRaisesRegex(ValueError, "版本信息"):
            self.run_node(refine_action="更新一采变更")

    def test_stale_source_cannot_overwrite_newer_high_resolution(self):
        old, _ = self.source.commit(self.input, changes=[(0, 107)])
        self.run_node(source_result=old)
        latest, _ = self.source.commit(self.input, changes=[(12, 20)])
        current = self.run_node(source_result=latest, refine_action="更新一采变更")
        self.guider.calls.clear()
        with self.assertRaisesRegex(ValueError, "版本已过期"):
            self.run_node(source_result=old, refine_action="更新一采变更")
        self.assertEqual(self.guider.calls, [])
        self.assertEqual(current[0]["pj_revision"], self.node._results.load()[0]["pj_revision"])

    def test_synced_update_skips_lazy_upscale_and_sampler_inputs(self):
        source, _ = self.source.commit(self.input, changes=[(0, 107)])
        result = self.run_node(source_result=source, refine_action="更新一采变更")
        self.assertEqual(self.node.check_lazy_status(refine_action="更新一采变更", source_result=source), [])
        reused = self.run_node(source_result=source, refine_action="更新一采变更",
                               latent_image=None, noise=None, guider=None, sigmas=None, sampler=None)
        self.assertEqual(result[0]["pj_revision"], reused[0]["pj_revision"])

    def test_audio_only_source_change_preserves_video_and_syncs_new_audio(self):
        original, _ = self.source.commit(self.input, changes=[(0, 107)], audio_changes=[(0, 603)])
        baseline, _ = self.run_node(source_result=original)
        video, audio = self.input["samples"].unbind()
        changed = {"samples": NestedTensor((video, audio + 3))}
        latest, _ = self.source.commit(changed, audio_changes=[(0, 603)])
        self.guider.calls.clear()
        output, x0 = self.run_node(source_result=latest, latent_image=changed, refine_action="更新一采变更")
        self.assertEqual(self.guider.calls, [])
        self.assertTrue(torch.equal(output["samples"].unbind()[0], baseline["samples"].unbind()[0]))
        for value in (output, x0):
            self.assertTrue(torch.equal(value["samples"].unbind()[1], audio + 3))
            self.assertEqual(value["pj_refine_source"]["version"], 2)
        self.assertEqual(output["pj_revision"]["changes"][-1]["audio"], [[0, 603]])

    def test_zero_overlap_uses_valid_windows_without_inheriting_padding(self):
        plan = dict(self.plan, overlap_seconds=0.)
        output, _ = self.run_node(segment_plan=plan)
        self.assertGreater(len(self.guider.calls), 1)
        for _, latent, _, options in self.guider.calls:
            self.assertEqual(latent.unbind()[0].shape[2] % 5, 2)
            self.assertIsNone(options["denoise_mask"])
        self.assertEqual(output["samples"].unbind()[0].shape[2], 107)
        self.guider.calls.clear()
        revised, _ = self.run_node(segment_plan=plan, refine_action="仅重做目标段",
                                   run_mode="仅精修第2段", noise=Noise_RandomNoise(88))
        old, new = output["samples"].unbind()[0], revised["samples"].unbind()[0]
        self.assertTrue(torch.equal(old[:, :, :42], new[:, :, :42]))
        self.assertTrue(torch.equal(old[:, :, 78:], new[:, :, 78:]))


if __name__ == "__main__":
    unittest.main()
