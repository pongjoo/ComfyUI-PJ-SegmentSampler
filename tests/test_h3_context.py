import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))
import comfy.cli_args
comfy.cli_args.args.cpu = True

import torch
import nodes
import comfy.samplers
from execution import PromptExecutor, validate_prompt
from comfy.ldm.minimax.model import FRAME_RESCALE, _video_t_grid
from comfy_extras.nodes_minimax_h3 import video_latent_t

package = ModuleType("pj_h3_context_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
from pj_h3_context_test.pj_h3_context import PJ_H3SegmentContext, motion_keyframes, prepare_segment_guider


class RecordingClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.full((1, 3, 4), len(tokens["prompt"]), dtype=torch.float32),
                 {"minimax_token_tags": torch.arange(3)}]]


class RecordingVAE:
    def __init__(self):
        self.calls = []

    def encode(self, frames):
        self.calls.append(frames.clone())
        length, height, width = frames.shape[:3]
        tokens = 1 if length == 1 else video_latent_t(length)
        return torch.full((1, 24, tokens, height // 16, width // 16), frames.mean().item())


class RecordingAudioVAE:
    audio_sample_rate = 24

    def __init__(self):
        self.calls = []

    def encode(self, waveform):
        self.calls.append(waveform.clone())
        return torch.zeros(1, 32, 2, round(waveform.shape[1] * 40 / 24))


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.clip = RecordingClip()
        self.vae = RecordingVAE()
        self.node = PJ_H3SegmentContext()
        patcher = SimpleNamespace(model_options={"transformer_options": {"test": 3}}, is_dynamic=lambda: False)
        self.guider = comfy.samplers.CFGGuider(patcher)
        self.guider.set_cfg(4.25)
        self.negative = torch.full((1, 1, 4), -1.)
        self.guider.set_conds(
            [[torch.zeros(1, 1, 4), {"strength": 0.8, "minimax_refs": [{"kind": "image", "latent": torch.zeros(1)}]}]],
            [[self.negative, {"strength": 0.3}]])

    def context(self, **kwargs):
        return self.node.prepare(self.clip, self.vae, "global prompt", **kwargs)[0]

    def test_target_outputs_floor_length_without_encoding_full_references(self):
        reference = torch.zeros(960, 32, 32, 3)
        image = torch.zeros(1, 32, 64, 3)
        context, latent, positive = self.node.execute(
            clip=self.clip, vae=self.vae, prompt="replace shoes", width=64, height=96, length=960,
            reference_video=reference, ref_images={"ref_image_0": image}).result
        video, audio = latent["samples"].unbind()
        self.assertEqual(tuple(video.shape), (1, 24, 282, 6, 4))
        self.assertEqual(tuple(audio.shape), (1, 32, 2, 1595))
        self.assertEqual(video.count_nonzero().item(), 0)
        self.assertEqual(audio.count_nonzero().item(), 0)
        self.assertIs(context["reference_video"], reference)
        self.assertEqual(context["prompt"], "replace shoes")
        self.assertEqual(self.vae.calls, [])
        self.assertEqual(self.clip.calls, [("", {"images": []})])
        self.assertNotIn("minimax_refs", positive[0][1])
        self.assertEqual(len(self.node.prepare(self.clip)), 1)

    def test_target_frame_floor_edges_and_legacy_defaults(self):
        for frames, tokens in ((5, 2), (21, 2), (22, 7), (322, 92), (328, 97)):
            with self.subTest(frames=frames):
                _, latent, _ = self.node.execute(clip=self.clip, width=32, height=32, length=frames).result
                self.assertEqual(latent["samples"].unbind()[0].shape[2], tokens)
        inputs = self.node.INPUT_TYPES()
        for name in ("width", "height", "length"):
            self.assertIn(name, inputs["optional"])
        self.assertEqual(list(self.node.RETURN_TYPES), ["PJ_H3_CONTEXT", "LATENT", "CONDITIONING"])

    def test_second_pass_conditions_use_actual_shape_not_target_dimensions(self):
        context, _, _ = self.node.execute(clip=self.clip, vae=self.vae, width=32, height=32, length=960,
                                         prompt="replace shoes", segment_prompts='{"2":"preserve logo"}',
                                         ref_images={"ref_image_0": torch.ones(1, 128, 96, 3)}).result
        rebuilt = prepare_segment_guider(self.guider, context, (35, 77, 198, 433),
                                         (1, 24, 42, 8, 6), segment_index=1)
        self.assertEqual(self.clip.calls[-1][0], "replace shoes\npreserve logo")
        self.assertEqual(tuple(self.vae.calls[-1].shape), (1, 128, 96, 3))
        ref = rebuilt.original_conds["positive"][0]["minimax_refs"][0]
        self.assertEqual((ref["latent_h"], ref["latent_w"]), (8, 6))

    def test_native_autogrow_schema_has_no_batch_socket(self):
        inputs = self.node.INPUT_TYPES()
        dynamic = inputs["optional"]["ref_images"]
        self.assertEqual(dynamic[0], "COMFY_AUTOGROW_V3")
        self.assertEqual(dynamic[1]["template"]["min"], 0)
        self.assertEqual(dynamic[1]["template"]["max"], 100)
        self.assertNotIn("reference_images", inputs["optional"])
        self.assertEqual(inputs["required"]["ref_image_size"][1]["options"], ["match", "max"])

    def test_independent_images_keep_dimensions_and_numeric_port_order(self):
        refs = {"ref_image_10": torch.full((1, 96, 32, 3), .9),
                "ref_image_2": torch.full((1, 32, 64, 3), .5),
                "ref_image_0": torch.full((1, 64, 32, 3), .1)}
        context = self.context(ref_images=refs, ref_image_size="max")
        prepare_segment_guider(self.guider, context, (0, 12, 0, 65), (1, 24, 12, 2, 2))
        self.assertEqual([tuple(x.shape) for x in self.vae.calls], [(1, 64, 32, 3), (1, 32, 64, 3), (1, 96, 32, 3)])
        for actual, name in zip(self.vae.calls, ("ref_image_0", "ref_image_2", "ref_image_10")):
            self.assertTrue(torch.allclose(actual, refs[name], atol=1 / 255 + 1e-6))

    def test_match_and_max_reach_native_reference_sizing(self):
        image = torch.rand(1, 96, 64, 3)
        for mode, shape in (("match", (1, 32, 32, 3)), ("max", (1, 96, 64, 3))):
            with self.subTest(mode=mode):
                self.vae.calls.clear()
                context = self.context(ref_images={"ref_image_0": image}, ref_image_size=mode)
                prepare_segment_guider(self.guider, context, (0, 12, 0, 65), (1, 24, 12, 2, 2))
                self.assertEqual(tuple(self.vae.calls[0].shape), shape)

    def test_native_executor_expands_dynamic_image_inputs(self):
        clip = self.clip
        images = (torch.zeros(1, 32, 64, 3), torch.ones(1, 64, 32, 3))
        captured = []
        targets = []

        class Source:
            INPUT_TYPES = classmethod(lambda cls: {"required": {}})
            RETURN_TYPES = ("CLIP", "IMAGE", "IMAGE", "INT", "INT", "INT")
            FUNCTION = "run"

            def run(self):
                return clip, *images, 64, 96, 322

        class Sink:
            INPUT_TYPES = classmethod(lambda cls: {"required": {"context": ("PJ_H3_CONTEXT",)},
                "optional": {"latent": ("LATENT",), "positive": ("CONDITIONING",)}})
            RETURN_TYPES = ()
            FUNCTION = "run"
            OUTPUT_NODE = True

            def run(self, context, latent=None, positive=None):
                captured.append(context)
                targets.append((latent, positive))
                return ()

        server = SimpleNamespace(client_id=None, send_sync=lambda *args, **kwargs: None)
        executor = PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})
        prompt = {
            "source": {"class_type": "PJImageTestSource", "inputs": {}},
            "context": {"class_type": "PJ_H3SegmentContext", "inputs": {
                "clip": ["source", 0], "prompt": "products", "segment_prompts": "{}",
                "reference_mode": "原视频编辑（按时间裁剪）", "motion_context": True, "ref_image_size": "max"}},
            "sink": {"class_type": "PJImageTestSink", "inputs": {"context": ["context", 0]}},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"PJImageTestSource": Source, "PJImageTestSink": Sink,
                                                   "PJ_H3SegmentContext": PJ_H3SegmentContext}):
            for count in (0, 1, 12):
                if count:
                    prompt["context"]["inputs"].update(width=["source", 3], height=["source", 4], length=["source", 5])
                    prompt["sink"]["inputs"].update(latent=["context", 1], positive=["context", 2])
                prompt["context"]["inputs"].update({f"ref_images.ref_image_{i}": ["source", 1 + i % 2] for i in range(count)})
                validated = asyncio.run(validate_prompt("autogrow-test", prompt, None))
                self.assertTrue(validated[0], validated)
                executor.execute(prompt, f"autogrow-{count}", execute_outputs=validated[2])
                self.assertTrue(executor.success, executor.status_messages)
                actual = captured[-1]["reference_images"]
                self.assertEqual(len(actual or []), count)
                self.assertEqual(captured[-1]["ref_image_size"], "max")
                self.assertIs(captured[-1]["clip"], clip)
                if count:
                    latent, positive = targets[-1]
                    self.assertEqual(tuple(latent["samples"].unbind()[0].shape), (1, 24, 92, 6, 4))
                    self.assertNotIn("minimax_refs", positive[0][1])
                for index, image in enumerate(actual or []):
                    self.assertTrue(torch.equal(image, images[index % 2][0]))

    def test_rebuild_uses_native_text_node_preserves_guider_policy_and_negative(self):
        context = self.context(segment_prompts='{"2":"turn left"}')
        clone = prepare_segment_guider(self.guider, context, (10, 22, 57, 122), (1, 24, 12, 2, 2), segment_index=1)
        self.assertEqual(self.clip.calls[0][0], "global prompt\nturn left")
        self.assertIsNot(clone, self.guider)
        self.assertEqual(clone.cfg, 4.25)
        self.assertIs(clone.model_patcher, self.guider.model_patcher)
        self.assertIs(clone.model_options, self.guider.model_options)
        self.assertIs(clone.original_conds["negative"][0]["cross_attn"], self.negative)
        self.assertEqual(clone.original_conds["negative"][0]["strength"], 0.3)
        positive = clone.original_conds["positive"][0]
        self.assertEqual(positive["strength"], 0.8)
        self.assertNotIn("minimax_refs", positive)
        self.assertIn("minimax_refs", self.guider.original_conds["positive"][0])
        self.assertTrue(torch.equal(positive["minimax_token_tags"], torch.arange(3)))

    def test_source_video_crops_text_semantics_and_vae_to_same_absolute_window(self):
        video = torch.arange(90, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3) / 100
        context = self.context(reference_video=video)
        clone = prepare_segment_guider(self.guider, context, (10, 22, 57, 122), (1, 24, 12, 2, 2))
        self.assertTrue(torch.allclose(self.vae.calls[0], video[34:73], atol=1 / 255 + 1e-6))
        semantic_video = self.clip.calls[0][1]["minimax_ref_items"][0]
        self.assertEqual(semantic_video["timestamps"], [0., .5, 1., 1.5])
        self.assertTrue(torch.allclose(semantic_video["data"], video[[34, 46, 58, 70]], atol=1 / 255 + 1e-6))
        self.assertEqual(clone.original_conds["positive"][0]["minimax_refs"][0]["latent_t"], 12)

    def test_independent_material_keeps_video_duration_and_all_images(self):
        video = torch.rand(90, 32, 32, 3)
        images = torch.rand(3, 32, 32, 3)
        context = self.context(reference_video=video, reference_images=images,
                               reference_mode="素材参考（保持完整）")
        clone = prepare_segment_guider(self.guider, context, (10, 22, 57, 122), (1, 24, 12, 2, 2))
        items = self.clip.calls[0][1]["minimax_ref_items"]
        self.assertEqual([item["type"] for item in items], ["image", "image", "image", "video"])
        self.assertTrue(torch.allclose(self.vae.calls[-1], video, atol=1 / 255 + 1e-6))
        self.assertEqual(clone.original_conds["positive"][0]["minimax_refs"][-1]["latent_t"], 27)

    def test_motion_context_matches_native_time_grid_copies_tail_without_decode(self):
        tail = torch.randn(1, 24, 7, 2, 2, dtype=torch.bfloat16)
        clone = prepare_segment_guider(self.guider, self.context(), (10, 22, 57, 122),
                                       (1, 24, 12, 2, 2), previous_tail=tail)
        guides = clone.original_conds["positive"][0]["minimax_keyframes"]
        native_grid = _video_t_grid(7, 0) / FRAME_RESCALE
        self.assertTrue(torch.allclose(torch.tensor([item["resolved_frame_index"] for item in guides], dtype=torch.float64), native_grid))
        self.assertEqual([item["resolved_frame_index"] for item in guides], [0, 1, 5, 9, 13, 17, 18])
        combined = torch.cat([item["latent"] for item in guides], dim=2)
        self.assertTrue(torch.equal(combined, tail))
        tail.fill_(0)
        self.assertTrue(torch.equal(combined, torch.cat([item["latent"] for item in guides], dim=2)))
        self.assertEqual(combined.dtype, torch.bfloat16)
        self.assertEqual(self.vae.calls, [])

    def test_existing_keyframes_rebased_and_motion_can_be_disabled(self):
        anchor = torch.rand(1, 24, 1, 2, 2)
        self.guider.set_conds([[torch.zeros(1, 1, 4), {"minimax_keyframes": [{"resolved_frame_index": 68, "latent": anchor}]}]], [])
        clone = prepare_segment_guider(self.guider, self.context(motion_context=False), (10, 22, 57, 122),
                                       (1, 24, 12, 2, 2), previous_tail=torch.zeros(1, 24, 7, 2, 2))
        self.assertEqual(len(clone.original_conds["positive"][0]["minimax_keyframes"]), 1)
        self.assertEqual(clone.original_conds["positive"][0]["minimax_keyframes"][0]["resolved_frame_index"], 34)
        self.assertEqual(self.guider.original_conds["positive"][0]["minimax_keyframes"][0]["resolved_frame_index"], 68)

    def test_paired_audio_tracks_source_window_standalone_audio_stays_complete(self):
        audio_vae = RecordingAudioVAE()
        soundtrack = {"waveform": torch.arange(90, dtype=torch.float32).view(1, 1, -1), "sample_rate": 24}
        standalone = {"waveform": torch.ones(1, 1, 48), "sample_rate": 24}
        context = self.context(reference_video=torch.rand(90, 32, 32, 3),
                               reference_video_audio=soundtrack, reference_audio=standalone, audio_vae=audio_vae)
        clone = prepare_segment_guider(self.guider, context, (10, 22, 57, 122), (1, 24, 12, 2, 2))
        self.assertTrue(torch.equal(audio_vae.calls[0], soundtrack["waveform"][..., 34:73].movedim(1, -1)))
        self.assertTrue(torch.equal(audio_vae.calls[1], standalone["waveform"].movedim(1, -1)))
        refs = clone.original_conds["positive"][0]["minimax_refs"]
        self.assertEqual([item["kind"] for item in refs], ["video_audio", "audio"])
        self.assertEqual([item["ref_audio_t"] for item in refs], [65, 80])

    def test_text_only_context_does_not_require_vae(self):
        context = self.node.prepare(self.clip, prompt="text only")[0]
        clone = prepare_segment_guider(self.guider, context, (0, 12, 0, 65), (1, 24, 12, 2, 2))
        self.assertTrue(torch.isfinite(clone.original_conds["positive"][0]["cross_attn"]).all())

    def test_independent_reference_floor_trims_paired_audio_to_same_duration(self):
        audio_vae = RecordingAudioVAE()
        audio = {"waveform": torch.arange(322, dtype=torch.float32).view(1, 1, -1), "sample_rate": 24}
        context = self.context(reference_video=torch.rand(322, 32, 32, 3), reference_video_audio=audio,
                               audio_vae=audio_vae, reference_mode="素材参考（保持完整）")
        clone = prepare_segment_guider(self.guider, context, (0, 12, 0, 65), (1, 24, 12, 2, 2))
        self.assertEqual(self.vae.calls[0].shape[0], 311)
        self.assertTrue(torch.equal(audio_vae.calls[0], audio["waveform"][..., :311].movedim(1, -1)))
        reference = clone.original_conds["positive"][0]["minimax_refs"][0]
        self.assertEqual(reference["latent_t"], 92)
        self.assertEqual(reference["ref_audio_t"], round(311 * 40 / 24))

    def test_upstream_keyframes_must_match_refine_resolution(self):
        self.guider.set_conds([[torch.zeros(1, 1, 4), {"minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 2, 2)}]}]], [])
        with self.assertRaisesRegex(ValueError, "分辨率.*不匹配"):
            prepare_segment_guider(self.guider, self.context(), (0, 12, 0, 65), (1, 24, 12, 4, 4))

    def test_reference_exhaustion_and_invalid_phase_fail_before_encoding(self):
        context = self.context(reference_video=torch.zeros(60, 32, 32, 3))
        with self.assertRaisesRegex(ValueError, "参考视频不足"):
            prepare_segment_guider(self.guider, context, (10, 22, 57, 122), (1, 24, 12, 2, 2))
        with self.assertRaisesRegex(ValueError, "17n"):
            prepare_segment_guider(self.guider, self.context(), (11, 23, 57, 122), (1, 24, 12, 2, 2))
        with self.assertRaisesRegex(ValueError, "相位对齐"):
            motion_keyframes(torch.zeros(1, 24, 5, 2, 2), (1, 24, 12, 2, 2))
        self.assertEqual(self.clip.calls, [])

    def test_prompt_schema_rejects_ambiguous_values_and_audio_pairing(self):
        for prompts in ("[]", '{"0":"x"}', '{"2":3}', '{bad'):
            with self.assertRaises(ValueError):
                self.context(segment_prompts=prompts)
        with self.assertRaisesRegex(ValueError, "配套视频音轨"):
            self.context(reference_video_audio={"sample_rate": 24, "waveform": torch.zeros(1, 1, 90)})


if __name__ == "__main__":
    unittest.main()
