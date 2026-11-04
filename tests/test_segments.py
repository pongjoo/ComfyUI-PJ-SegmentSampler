import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))
import comfy.cli_args
comfy.cli_args.args.cpu = True

import torch
import comfy.samplers
import comfy.model_base
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, Noise_EmptyNoise

spec = importlib.util.spec_from_file_location("pj_segments_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from pj_segments_test.segment_boundaries import constrain_boundaries
from pj_segments_test.segment_timing import frames_for_tokens, tokens_for_frames, plan_segments
from pj_segments_test.segment_sampling import blend_sequence, segment_guider, generate_noise, slice_mask, fingerprint
from pj_segments_test.pj_segment_planner import PJ_SegmentPlanner
from pj_segments_test.pj_first_pass_sampler import PJ_FirstPass_SegmentSampler
from pj_segments_test.pj_segment_sampler import PJ_SegmentSampler


def callback(_model, _steps, output):
    return lambda step, x0, x, total: output.update(x0=x0)


class RecordingGuider(comfy.samplers.CFGGuider):
    def __init__(self):
        model = SimpleNamespace(process_latent_out=lambda x: x * 2)
        patcher = SimpleNamespace(model_options={}, model=model, patches_uuid="test-model", is_dynamic=lambda: False)
        super().__init__(patcher)
        self.set_conds([[torch.ones(1, 2, 3), {}]], [])
        self.calls = []

    def sample(self, noise, latent, sampler, sigmas, **kwargs):
        self.calls.append((noise, latent, self.original_conds, kwargs))
        if kwargs["callback"]:
            kwargs["callback"](0, noise, noise, 1)
        return noise.cpu()


class NativeMaskGuider(RecordingGuider):
    """Real H3 mask/inpaint adapter with a noise-returning DiT substitute; no weights."""
    def sample(self, noise, latent, sampler, sigmas, **kwargs):
        super().sample(noise, latent, sampler, sigmas, **kwargs)
        mask = kwargs["denoise_mask"]
        if mask is None:
            return noise.cpu()
        nested = latent.is_nested
        streams = list(latent.unbind()) if nested else [latent]
        noises = list(noise.unbind()) if nested else [noise]
        masks = list(mask.unbind()) if mask.is_nested else [mask]
        masks = [comfy.sampler_helpers.prepare_mask(m, t.shape, t.device) for m, t in zip(masks, streams)]
        packed, shapes = comfy.utils.pack_latents(streams)
        packed_noise, _ = comfy.utils.pack_latents(noises)
        packed_mask, _ = comfy.utils.pack_latents(masks)
        model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
        torch.nn.Module.__init__(model)
        model.diffusion_model = SimpleNamespace(patch_size=(1, 2, 2))
        model.latent_shapes = shapes
        model.audio_scale = lambda: 1.0
        if nested:
            conditions = model._denoise_mask_values(packed_mask, shapes)
            assert "denoise_mask" in conditions or "audio_denoise_mask" in conditions
        class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass
        flow = FlowSampling()
        class Inner:
            inner_model = model
            model_patcher = SimpleNamespace(get_model_object=lambda name: flow)
            def __call__(self, x, sigma, **options):
                return packed_noise
        wrapper = comfy.samplers.KSamplerX0Inpaint(Inner(), sigmas)
        wrapper.noise = packed_noise
        wrapper.latent_image = packed
        result = sampler.sampler_function(wrapper, packed_noise, sigmas,
                                          extra_args={"denoise_mask": packed_mask, "seed": kwargs["seed"]}, disable=True)
        parts = comfy.utils.unpack_latents(result, shapes)
        return NestedTensor(parts) if nested else parts[0]


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.guider = RecordingGuider()
        self.latent = {"samples": NestedTensor((torch.zeros(1, 24, 107, 2, 2), torch.zeros(1, 32, 2, 603)))}
        self.noise = Noise_RandomNoise(42)
        self.sampler = comfy.samplers.ksampler("euler")
        self.plan = dict(split_mode="按秒自动切分", chunk_seconds=6., overlap_seconds=1.5, fps=24., align_51_frames=False)
        self.nodes = []
        self.patch = patch("latent_preview.prepare_callback", callback)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        for node in self.nodes:
            if node._cache.directory is not None:
                node._cache.directory.cleanup()

    def node(self, cls):
        node = cls()
        self.nodes.append(node)
        return node

    def run_node(self, node, **kwargs):
        args = dict(noise=self.noise, guider=self.guider, sampler=self.sampler,
                    sigmas=torch.tensor([1., 0.]), latent_image=self.latent,
                    segment_plan=self.plan, unload_model_after_sampling=False)
        args.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return node.execute(**args)

    def test_frame_contract_and_all_planning_modes(self):
        for frames, vt, at in ((124, 37, 207), (141, 42, 235), (362, 107, 603)):
            self.assertEqual(tokens_for_frames(frames), vt)
            self.assertEqual(frames_for_tokens(vt), frames)
            self.assertEqual(round(frames * 40 / 24), at)
        for total in (2, 7, 37, 107, 212):
            audio = round(frames_for_tokens(total) * 40 / 24)
            for mode in ("按秒自动切分", "固定分段数", "自定义切点"):
                for align in (False, True):
                    chunks = plan_segments(total, audio, mode, 6, 1.5, 3, "0-6, 4.5-10.5, 9-30", 24, align)
                    self.assertEqual(chunks[0][0], 0)
                    self.assertEqual(chunks[-1][1], total)
                    for vs, ve, a_s, ae in chunks:
                        self.assertEqual(vs % 5, 0)
                        self.assertEqual((ve - vs) % 5, 2)
                        self.assertTrue(0 <= a_s < ae <= audio)
                        self.assertEqual(a_s, round(frames_for_tokens(vs) * 40 / 24))
                        self.assertEqual(ae, round(frames_for_tokens(ve) * 40 / 24))

    def test_auto_selects_nearest_legal_duration_then_overlap(self):
        for seconds in (2, 3, 6, 6.5, 10, 30):
            for overlap in (0, 0.5, 1, 1.5, 4, 10):
                chunks = plan_segments(107, 603, "按秒自动切分", seconds, overlap, 3, "")
                first = chunks[0]
                lengths = range(7, 108, 5)
                closest = min(lengths, key=lambda t: (abs(frames_for_tokens(t) - seconds * 24), -t))
                self.assertEqual(first[1], closest)
                if len(chunks) > 1:
                    candidates = range(2, closest, 5)
                    nearest_overlap = min(candidates, key=lambda t: (abs(frames_for_tokens(t) - overlap * 24), -t))
                    self.assertEqual(first[1] - chunks[1][0], nearest_overlap)
                self.assertEqual(chunks[-1][1:], (107, chunks[-1][2], 603))
                self.assertTrue(all(left[1] >= right[0] for left, right in zip(chunks, chunks[1:])))

    def test_half_second_and_rounding_ties(self):
        chunks = plan_segments(107, 603, "按秒自动切分", 6, 0.5, 3, "")
        self.assertEqual([(c[0], c[1]) for c in chunks], [(0, 42), (40, 82), (80, 107)])
        self.assertEqual(frames_for_tokens(chunks[0][1]) - frames_for_tokens(chunks[1][0]), 5)
        # 149.5 frames is halfway between 141 and 158; 13.5 between 5 and 22.
        chunks = plan_segments(107, 603, "按秒自动切分", 149.5 / 24, 13.5 / 24, 3, "")
        self.assertEqual(chunks[0][1], 47)
        self.assertEqual(chunks[0][1] - chunks[1][0], 7)

    def test_fixed_count_preserves_distinct_ranges_and_nearest_overlap(self):
        for total in (2, 7, 37, 107):
            audio = round(frames_for_tokens(total) * 40 / 24)
            for count in (1, 3, 20, 50):
                chunks = plan_segments(total, audio, "固定分段数", 6, 0.5, count, "")
                self.assertEqual(len(chunks), min(count, (total - 2) // 5 + 1))
                self.assertEqual(len({c[0] for c in chunks}), len(chunks))
                self.assertEqual(chunks[-1][1], total)
                for a, b in zip(chunks, chunks[1:]):
                    self.assertEqual(frames_for_tokens(a[1]) - frames_for_tokens(b[0]), 5)

    def test_custom_boundaries_round_to_nearest_instead_of_outwards(self):
        chunks = plan_segments(107, 603, "自定义切点", 6, 0.5, 3, "0.4-1.2, 6.6-8.3")
        self.assertEqual([(c[0], c[1]) for c in chunks], [(5, 7), (45, 57)])

    def test_legacy_grid_toggle_is_migrated_and_actual_overlap_is_reported(self):
        with contextlib.redirect_stdout(io.StringIO()):
            new = PJ_SegmentPlanner().plan(chunk_seconds=6, overlap_seconds=0.5)
            old = PJ_SegmentPlanner().plan(chunk_seconds=6, overlap_seconds=0.5, align_51_frames=True)
        self.assertEqual(new["result"][0]["chunks"], old["result"][0]["chunks"])
        self.assertNotIn("align_51_frames", PJ_SegmentPlanner.INPUT_TYPES()["required"])
        data = new["ui"]["preview_data"][0]
        self.assertEqual(data["timing_policy"], "nearest_h3")
        self.assertEqual(data["chunks"][1]["overlap_frames"], 5)
        self.assertEqual(data["chunks"][1]["overlap_seconds"], 0.208)
        self.assertEqual([c["duration_sec"] for c in data["chunks"][:2]], [5.88, 5.88])
        # Both samplers must resolve saved rules identically to the new planner.
        for cls in (PJ_FirstPass_SegmentSampler, PJ_SegmentSampler):
            self.guider.calls.clear()
            self.run_node(self.node(cls), segment_plan=old["result"][0])
            self.assertEqual([call[1].unbind()[0].shape[2] for call in self.guider.calls], [42, 42, 27])

    def test_connected_planner_uses_real_audio_length(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.plan = PJ_SegmentPlanner().plan(latent=self.latent)["result"][0]
        self.assertEqual((self.plan["total_frames"], self.plan["total_a"]), (362, 603))
        for cls in (PJ_FirstPass_SegmentSampler, PJ_SegmentSampler):
            output = self.run_node(self.node(cls), keep_original_audio=False)[0]["samples"]
            self.assertEqual([tuple(x.shape) for x in output.unbind()], [tuple(x.shape) for x in self.latent["samples"].unbind()])

    def test_old_plan_audio_bounds_are_recomputed(self):
        self.plan.update(has_media=True, total_v=107, chunks=[(0, 107, 0, 606)])
        self.run_node(self.node(PJ_FirstPass_SegmentSampler))
        self.assertEqual(self.guider.calls[-1][1].unbind()[1].shape[-1], 93)

    def test_seed_offset_changes_noise_and_does_not_mutate_source(self):
        node = self.node(PJ_FirstPass_SegmentSampler)
        before = self.run_node(node)[0]["samples"]
        after = self.run_node(node, seg1_seed_offset=999)[0]["samples"]
        for a, b in zip(before.unbind(), after.unbind()):
            self.assertFalse(torch.equal(a, b))
        self.assertEqual(self.noise.seed, 42)
        empty = generate_noise(Noise_EmptyNoise(), self.latent, 99)
        self.assertEqual(sum(t.count_nonzero().item() for t in empty.unbind()), 0)

    def test_cache_isolation_invalidation_and_seed_lock(self):
        node = self.node(PJ_FirstPass_SegmentSampler)
        self.run_node(node)
        locks = {f"seg{i}_lock": True for i in range(1, 21)}
        self.guider.calls.clear()
        self.run_node(node, seg1_seed_offset=77, **locks)
        self.assertEqual(len(self.guider.calls), 0)
        self.run_node(self.node(PJ_FirstPass_SegmentSampler), **locks)
        self.assertEqual(len(self.guider.calls), 4)
        self.guider.calls.clear()
        self.latent["samples"].unbind()[0][:, :, :5] += 1
        self.run_node(node, **locks)
        self.assertEqual(len(self.guider.calls), 1)
        self.guider.calls.clear()
        self.guider.set_cfg(2.)
        self.run_node(node, **locks)
        self.assertEqual(len(self.guider.calls), 4)

    def test_partial_rerun_preserves_x0_positions_and_scaling(self):
        node = self.node(PJ_SegmentSampler)
        self.run_node(node)
        self.guider.calls.clear()
        output, denoised = self.run_node(node, run_mode="仅精修第2段")
        self.assertEqual(len(self.guider.calls), 1)
        self.assertEqual(denoised["samples"].unbind()[0].shape[2], 107)
        self.assertTrue(torch.allclose(denoised["samples"].unbind()[0], output["samples"].unbind()[0] * 2))
        self.assertTrue(torch.equal(output["samples"].unbind()[1], self.latent["samples"].unbind()[1]))

    def test_reference_and_keyframe_conditions_use_guider_contract(self):
        ref = {"kind": "video_audio", "latent": torch.zeros(1, 24, 107, 2, 2), "latent_t": 107,
               "audio_latent": torch.zeros(1, 32, 2, 603), "ref_audio_t": 603}
        keyframes = [{"resolved_frame_index": i, "latent": torch.ones(1, 24, 1, 2, 2)} for i in (0, 120, 361)]
        self.guider.set_conds([[torch.ones(1, 2, 3), {"minimax_refs": [ref], "minimax_keyframes": keyframes}]], [])
        clone = segment_guider(self.guider, (30, 72, 170, 405), True)
        condition = clone.original_conds["positive"][0]
        self.assertEqual(condition["minimax_refs"][0]["latent_t"], 42)
        self.assertEqual(condition["minimax_refs"][0]["ref_audio_t"], 235)
        self.assertEqual([x["resolved_frame_index"] for x in condition["minimax_keyframes"]], [18])
        self.assertEqual(self.guider.original_conds["positive"][0]["minimax_refs"][0]["latent_t"], 107)
        for cls in (PJ_FirstPass_SegmentSampler, PJ_SegmentSampler):
            self.guider.calls.clear()
            self.run_node(self.node(cls))
            self.assertEqual(self.guider.calls[1][2]["positive"][0]["minimax_refs"][0]["latent_t"], 42)

    def test_custom_gaps_and_hard_cut_do_not_average_or_zero_input(self):
        original = torch.full((1, 1, 20, 1, 1), 7.)
        chunks = [torch.zeros(1, 1, 7, 1, 1), torch.ones(1, 1, 7, 1, 1)]
        result = blend_sequence(chunks, [(0, 7), (5, 12)], original, "硬切")
        self.assertTrue(torch.equal(result[:, :, 5:12], chunks[1]))
        self.assertTrue(torch.equal(result[:, :, 12:], original[:, :, 12:]))
        result = blend_sequence([chunks[0]], [(5, 12)], original, "cosine")
        self.assertEqual(result.shape, original.shape)
        self.assertTrue(torch.equal(result[:, :, :5], original[:, :, :5]))

    def test_native_h3_clock_is_not_changed_by_playback_fps(self):
        self.plan["fps"] = 30
        node = self.node(PJ_FirstPass_SegmentSampler)
        self.run_node(node)
        self.assertEqual(self.guider.calls[0][1].unbind()[1].shape[-1], 235)

    def test_image_preview_uses_source_rate_but_h3_plan_uses_24(self):
        with patch("pj_segments_test.pj_segment_planner._generate_preview_mp4", return_value=None) as preview:
            with contextlib.redirect_stdout(io.StringIO()):
                result = PJ_SegmentPlanner().plan(images=torch.zeros(450, 2, 2, 3), fps=30)["result"][0]
        self.assertEqual(result["fps"], 24)
        self.assertEqual(result["total_frames"], 362)
        self.assertEqual(preview.call_args.args[2], 30)

    def test_full_length_and_unconnected_rule_preview(self):
        node = self.node(PJ_FirstPass_SegmentSampler)
        with contextlib.redirect_stdout(io.StringIO()):
            plan = PJ_SegmentPlanner().plan()["result"][0]
        output = self.run_node(node, segment_plan=plan, run_mode="整段全量采样(一采不分段)")[0]["samples"]
        self.assertEqual(len(self.guider.calls), 1)
        self.assertEqual(output.unbind()[0].shape[2], 107)

    def test_x0_audio_uses_denoised_stream_when_bypass_disabled(self):
        node = self.node(PJ_SegmentSampler)
        output, denoised = self.run_node(node, keep_original_audio=False)
        for a, b in zip(output["samples"].unbind(), denoised["samples"].unbind()):
            self.assertTrue(torch.allclose(a * 2, b))

    def test_masks_preserve_broadcast_and_temporal_slices(self):
        video_mask = torch.ones(1, 1, 1, 2, 2)
        audio_mask = torch.arange(603.).reshape(1, 1, 1, 603)
        masks = slice_mask(NestedTensor((video_mask, audio_mask)), (30, 72, 170, 405))
        self.assertEqual(masks.unbind()[0].shape[2], 1)
        self.assertTrue(torch.equal(masks.unbind()[1], audio_mask[..., 170:405]))
        self.latent["noise_mask"] = NestedTensor((video_mask, audio_mask))
        self.run_node(self.node(PJ_FirstPass_SegmentSampler))
        self.assertEqual(self.guider.calls[1][3]["denoise_mask"].unbind()[1].shape[-1], 235)

    def test_condition_and_schedule_changes_invalidate_locks(self):
        node = self.node(PJ_SegmentSampler)
        self.run_node(node)
        locks = dict(run_mode="智能缓存(仅重跑未锁定分段)", **{f"seg{i}_lock": True for i in range(1, 21)})
        self.guider.calls.clear()
        self.run_node(node, **locks)
        self.assertEqual(len(self.guider.calls), 0)
        self.guider.original_conds["positive"][0]["cross_attn"] += 1
        self.run_node(node, **locks)
        self.assertEqual(len(self.guider.calls), 4)
        self.guider.calls.clear()
        self.run_node(node, sigmas=torch.tensor([0.5, 0.]), **locks)
        self.assertEqual(len(self.guider.calls), 4)

    def test_cache_prunes_old_segments_and_stores_no_tensors_in_metadata(self):
        node = self.node(PJ_FirstPass_SegmentSampler)
        self.run_node(node)
        self.assertEqual(len(list(Path(node._cache.directory.name).glob("*.pt"))), 4)
        self.run_node(node, run_mode="整段全量采样(一采不分段)")
        self.assertEqual(list(node._cache.entries), [0])
        self.assertEqual(len(list(Path(node._cache.directory.name).glob("*.pt"))), 1)
        self.assertIsInstance(node._cache.entries[0], str)
        self.assertNotEqual(fingerprint([[1], 2]), fingerprint([[1, 2]]))

    def test_custom_invalid_ranges_report_errors(self):
        for cuts in ("bad", "6-0", "nan-10", "30-40"):
            with self.assertRaises(ValueError):
                plan_segments(107, 603, "自定义切点", 6, 1.5, 3, cuts)

    def test_crossing_guide_and_short_reference_never_produce_empty_video(self):
        video = torch.ones(1, 24, 37, 2, 2)
        self.guider.set_conds([[None, {"minimax_keyframes": [{"resolved_frame_index": 50, "latent": video}],
                                      "minimax_refs": [{"kind": "video", "latent": video, "latent_t": 37, "ref_audio_t": 0}]}]], [])
        clone = segment_guider(self.guider, (30, 72, 170, 405), True)
        guide = clone.original_conds["positive"][0]["minimax_keyframes"][0]
        self.assertEqual(guide["resolved_frame_index"], -1)
        self.assertEqual(guide["latent"].shape[2], 22)
        clone = segment_guider(self.guider, (60, 102, 340, 575), True)
        self.assertEqual(clone.original_conds["positive"][0]["minimax_refs"], [])

    def test_boundary_first_generation_uses_native_video_and_audio_masks(self):
        self.guider = NativeMaskGuider()
        self.plan["overlap_seconds"] = 0.5
        node = self.node(PJ_FirstPass_SegmentSampler)
        output = self.run_node(node, boundary_constraints=True)[0]["samples"]
        self.assertEqual(len(self.guider.calls), 3)
        first, second, third = self.guider.calls
        self.assertIsNone(first[3]["denoise_mask"])
        vm, am = second[3]["denoise_mask"].unbind()
        self.assertTrue(torch.all(vm[:, :, :2] == 0))
        self.assertTrue(torch.all(vm[:, :, 2:] == 1))
        self.assertTrue(torch.all(am[..., :8] == 0))
        self.assertTrue(torch.all(am[..., 8:] == 1))
        self.assertTrue(torch.equal(second[1].unbind()[0][:, :, :2], first[0].unbind()[0][:, :, 40:42]))
        self.assertTrue(torch.equal(third[1].unbind()[0][:, :, :2], output.unbind()[0][:, :, 80:82]))
        self.assertEqual([t.shape for t in output.unbind()], [t.shape for t in self.latent["samples"].unbind()])

    def test_local_redo_preserves_both_neighbours_without_cascade(self):
        self.guider = NativeMaskGuider()
        self.plan["overlap_seconds"] = 0.5
        node = self.node(PJ_FirstPass_SegmentSampler)
        before = self.run_node(node, boundary_constraints=True)[0]["samples"]
        self.guider.calls.clear()
        after = self.run_node(node, boundary_constraints=True, run_mode="仅重抽第2段", seg2_seed_offset=999)[0]["samples"]
        self.assertEqual(len(self.guider.calls), 1)
        for axis, start, end, a, b in zip((2, 3), (42, 235), (80, 453), before.unbind(), after.unbind()):
            self.assertTrue(torch.equal(a.narrow(axis, 0, start), b.narrow(axis, 0, start)))
            self.assertTrue(torch.equal(a.narrow(axis, end, a.shape[axis] - end), b.narrow(axis, end, b.shape[axis] - end)))
            self.assertFalse(torch.equal(a.narrow(axis, start, end - start), b.narrow(axis, start, end - start)))
        self.guider.calls.clear()
        repeated = self.run_node(node, boundary_constraints=True, **{f"seg{i}_lock": True for i in range(1, 21)})[0]["samples"]
        self.assertEqual(len(self.guider.calls), 0)
        for a, b in zip(after.unbind(), repeated.unbind()):
            self.assertTrue(torch.equal(a, b))

    def test_boundary_multi_segment_redo_and_triple_overlaps(self):
        self.guider = NativeMaskGuider()
        self.plan["overlap_seconds"] = 4
        node = self.node(PJ_FirstPass_SegmentSampler)
        before = self.run_node(node, boundary_constraints=True)[0]["samples"]
        self.guider.calls.clear()
        after = self.run_node(node, boundary_constraints=True, run_mode="自定义指定分段(下方输入)",
                              custom_rerun_segments="1,2", seg1_seed_offset=555, seg2_seed_offset=556)[0]["samples"]
        self.assertEqual(len(self.guider.calls), 2)
        intervals = plan_segments(107, 603, "按秒自动切分", 6, 4, 3, "")
        for axis, start, a, b in zip((2, 3), (intervals[2][0], intervals[2][2]), before.unbind(), after.unbind()):
            self.assertTrue(torch.equal(a.narrow(axis, start, a.shape[axis] - start), b.narrow(axis, start, b.shape[axis] - start)))

    def test_er_sde_boundary_sampling_keeps_retained_overlap_exact(self):
        self.guider = NativeMaskGuider()
        self.sampler = comfy.samplers.ksampler("er_sde")
        self.plan["overlap_seconds"] = 0.5
        node = self.node(PJ_FirstPass_SegmentSampler)
        sigmas = torch.tensor([1., 0.7, 0.3, 0.])
        before = self.run_node(node, boundary_constraints=True, sigmas=sigmas)[0]["samples"]
        self.guider.calls.clear()
        after = self.run_node(node, boundary_constraints=True, sigmas=sigmas,
                              run_mode="仅重抽第2段", seg2_seed_offset=456)[0]["samples"]
        self.assertEqual(len(self.guider.calls), 1)
        self.assertTrue(torch.equal(before.unbind()[0][:, :, :42], after.unbind()[0][:, :, :42]))
        self.assertTrue(torch.equal(before.unbind()[0][:, :, 80:], after.unbind()[0][:, :, 80:]))
        self.assertFalse(torch.equal(before.unbind()[0][:, :, 42:80], after.unbind()[0][:, :, 42:80]))

    def test_boundary_mode_changes_and_plan_changes_invalidate_cache(self):
        self.guider = NativeMaskGuider()
        node = self.node(PJ_FirstPass_SegmentSampler)
        locks = {f"seg{i}_lock": True for i in range(1, 21)}
        self.run_node(node)
        self.guider.calls.clear()
        self.run_node(node, boundary_constraints=True, **locks)
        self.assertEqual(len(self.guider.calls), 4)
        self.guider.calls.clear()
        self.plan["overlap_seconds"] = 0.5
        self.run_node(node, boundary_constraints=True, **locks)
        self.assertEqual(len(self.guider.calls), 3)
        self.guider.calls.clear()
        self.run_node(node, boundary_constraints=False, **locks)
        self.assertEqual(len(self.guider.calls), 3)

    def test_interrupted_boundary_rebuild_invalidates_all_cache_entries(self):
        self.guider = NativeMaskGuider()
        node = self.node(PJ_FirstPass_SegmentSampler)
        self.run_node(node, boundary_constraints=True)
        with patch.object(self.guider, "sample", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.run_node(node, boundary_constraints=True, run_mode="仅重抽第2段")
        self.assertFalse(node._cache.entries)
        self.guider.calls.clear()
        self.run_node(node, boundary_constraints=True, run_mode="仅重抽第2段")
        self.assertEqual(len(self.guider.calls), 4)

    def test_boundaries_preserve_input_masks_dtype_and_source_tensors(self):
        video = torch.zeros(1, 24, 42, 2, 2, dtype=torch.bfloat16)
        audio = torch.zeros(1, 32, 2, 235, dtype=torch.bfloat16)
        source = NestedTensor((torch.full_like(video, 3, dtype=torch.float32), torch.full_like(audio, 4, dtype=torch.float32)))
        mask = NestedTensor((torch.full((1, 1, 1, 2, 2), 0.25), torch.full((1, 1, 1, 1), 0.5)))
        result, masks = constrain_boundaries(NestedTensor((video, audio)), mask, (40, 82, 227, 462), [((0, 42, 0, 235), source)])
        self.assertEqual(result.unbind()[0].dtype, torch.float32)
        self.assertTrue(torch.all(result.unbind()[0][:, :, :2] == 3))
        self.assertTrue(torch.all(masks.unbind()[0][:, :, 2:] == 0.25))
        self.assertTrue(torch.all(masks.unbind()[1][..., 8:] == 0.5))
        self.assertEqual(video.count_nonzero(), 0)
        self.assertTrue(torch.all(mask.unbind()[0] == 0.25))

    def test_single_stream_and_nonoverlapping_boundaries(self):
        original = torch.zeros(1, 4, 7, 2, 2)
        source = torch.ones_like(original)
        unchanged, mask = constrain_boundaries(original, None, (10, 17, 0, 0), [((0, 7, 0, 0), source)])
        self.assertIs(unchanged, original)
        self.assertIsNone(mask)
        result, mask = constrain_boundaries(original, None, (5, 12, 0, 0), [((0, 7, 0, 0), source)])
        self.assertTrue(torch.all(result[:, :, :2] == 1))
        self.assertTrue(torch.all(mask[:, :, :2] == 0))
        self.assertTrue(torch.all(mask[:, :, 2:] == 1))

    def test_contained_and_duplicate_blends_do_not_erase_low_precision_tail(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            original = torch.zeros(1, 1, 107, 1, 1, dtype=dtype)
            for intervals in ([(0, 107), (5, 12)], [(0, 107), (0, 107)], [(0, 42), (30, 72), (60, 107)]):
                chunks = [torch.ones(1, 1, e - s, 1, 1, dtype=dtype) for s, e in intervals]
                result = blend_sequence(chunks, intervals, original, "cosine")
                self.assertTrue(torch.equal(result, torch.ones_like(original)))
                self.assertEqual(result.dtype, dtype)

    def test_dense_overlap_weights_remain_finite(self):
        original = torch.zeros(1, 1, 1, 1, 1)
        result = blend_sequence([torch.ones_like(original)] * 180, [(0, 1)] * 180, original, "cosine")
        self.assertTrue(torch.equal(result, torch.ones_like(original)))

    def test_blend_keeps_sample_precision_and_direct_copies_nonoverlap(self):
        original = torch.full((1, 1, 20, 1, 1), 7, dtype=torch.bfloat16)
        chunks = [torch.full((1, 1, 7, 1, 1), value) for value in (1.234567, 3.456789)]
        for mode in ("cosine", "linear", "硬切", "保留边界"):
            result = blend_sequence(chunks, [(0, 7), (5, 12)], original, mode)
            self.assertEqual(result.dtype, torch.float32)
            self.assertTrue(torch.equal(result[:, :, :5], chunks[0][:, :, :5]))
            self.assertTrue(torch.equal(result[:, :, 7:12], chunks[1][:, :, 2:]))
            self.assertTrue(torch.all(result[:, :, 12:] == 7))


if __name__ == "__main__":
    unittest.main()
