import contextlib
import io
import json
import unittest
from unittest.mock import patch

import test_segments as fixtures
import test_h3_context as context_fixtures

import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
import comfy.samplers

from pj_segments_test.pj_h3_context import PJ_H3SegmentContext, _encode_segment_conditions
from pj_segments_test.segment_script import parse_segment_prompts, segment_prompt
from pj_segments_test.segment_timing import plan_relay_segments, plan_segments, frames_for_tokens, delivery_chunks
from pj_segments_test.segment_scenes import split_scene_windows, scene_deliveries
from pj_segments_test.segment_audio import encode_soundtrack
from pj_segments_test.segment_local_repair import repair_segments
from pj_segments_test.segment_sampling import segment_guider


class ScriptTests(unittest.TestCase):
    def test_old_strings_and_structured_entries_preserve_text(self):
        source = {'01': 'shoe <d>[Chinese] 好</d>', '8': {'prompt': 'turn', 'transition': 'cut', 'dialogue': []}}
        parsed = parse_segment_prompts(json.dumps(source))
        self.assertEqual(parsed, {1: source['01'], 8: source['8']})
        self.assertEqual(source['8']['dialogue'], [])

    def test_invalid_script_is_explicit_not_silently_discarded(self):
        for text in ('[]', '{', '{"0":"x"}', '{"1":"x","01":"y"}',
                     '{"2":{"promt":"typo"}}', '{"1":{"dialogue":[{"text":"hello"}]}}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_segment_prompts(text)

    def test_dialogue_uses_only_current_entry_and_final_timeline_clock(self):
        context = {'prompt': 'global\noverall_soundscape:\nroom', 'segment_prompts': {
            1: {'dialogue': [{'speaker': '(S1)', 'text': '[Chinese] 第一段', 'start': 1, 'end': 2}]},
            2: {'prompt': 'turn', 'transition': 'cut', 'dialogue': [
                {'speaker': '(S2)', 'text': '[Chinese] 第二段', 'start': 15, 'end': 16}]}}}
        result = segment_prompt(context, 1, (100, 207, 567, 1170))
        self.assertIn('[0.83s–1.83s] (S2) says <d>[Chinese] 第二段</d>', result)
        self.assertNotIn('第一段', result)
        self.assertNotIn('transition', result)
        self.assertLess(result.index('第二段'), result.index('overall_soundscape'))
        context['segment_prompts'][2]['dialogue'] = []
        self.assertIn('no speech', segment_prompt(context, 1, (100, 207, 567, 1170)))


class PurposeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.nodes = []
        self.clip = context_fixtures.RecordingClip()
        self.vae = context_fixtures.RecordingVAE()
        self.guider = fixtures.RecordingGuider()
        self.noise = Noise_RandomNoise(42)
        self.sampler = comfy.samplers.ksampler('euler')
        self.plan = dict(split_mode='按秒自动切分', chunk_seconds=6., overlap_seconds=.92, fps=24.)
        self.callback = patch('latent_preview.prepare_callback', fixtures.callback)
        self.callback.start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.callback.stop()
        for node in self.nodes:
            for name in ('_cache', '_revisions', '_results'):
                cache = getattr(node, name, None)
                if cache is not None and cache.directory is not None:
                    cache.directory.cleanup()

    def latent(self, tokens=107):
        return {'samples': NestedTensor((torch.zeros(1, 24, tokens, 2, 2),
                    torch.zeros(1, 32, 2, round(frames_for_tokens(tokens) * 40 / 24))))}

    def context(self, **kwargs):
        return PJ_H3SegmentContext.prepare(self.clip, self.vae, prompt='global', **kwargs)[0]

    def run_first(self, **kwargs):
        node = fixtures.PJ_FirstPass_SegmentSampler()
        self.nodes.append(node)
        result = node.execute(self.noise, self.guider, self.sampler, torch.tensor([1., 0.]),
                              kwargs.pop('latent_image', self.latent()), segment_plan=self.plan,
                              generation_mode=kwargs.pop('generation_mode', '22帧接力'), **kwargs)
        return node, result[0]

    def test_mixed_cuts_have_legal_windows_and_no_compensation_tail(self):
        chunks = plan_relay_segments(317, 1793, 15., cut_segments={2})
        self.assertEqual(len(chunks), 3)
        self.assertEqual([c[:2] for c in chunks], [(0, 107), (105, 212), (205, 312)])
        for start, end, *_ in chunks:
            self.assertEqual(start % 5, 0)
            self.assertEqual((end - start) % 5, 2)
        delivered = scene_deliveries(chunks, chunks, [105])
        self.assertEqual(delivered[0][1], 105)
        self.assertEqual(frames_for_tokens(chunks[-1][1]), 1059)

    def test_cut_omits_video_inheritance_and_motion_guide_but_audio_can_continue(self):
        context = self.context(segment_prompts='{"2":{"transition":"cut"}}')
        node, result = self.run_first(h3_context=context)
        cut = result['pj_scene_cuts'][0]
        call = self.guider.calls[1]
        vm, am = call[3]['denoise_mask'].unbind()
        self.assertTrue((vm == 1).all())
        self.assertTrue((am[..., :8] == 0).all())
        self.assertNotIn('minimax_keyframes', call[2]['positive'][0])
        first = node._cache.load(0, node._cache.entries[0])[0].unbind()[0]
        self.assertTrue(torch.equal(result['samples'].unbind()[0][:, :, :cut], first[:, :, :cut]))
        second = node._cache.load(1, node._cache.entries[1])[0].unbind()[0]
        self.assertTrue(torch.equal(result['samples'].unbind()[0][:, :, cut:cut + 2], second[:, :, :2]))

    def test_audio_reset_can_keep_continuous_video(self):
        self.run_first(h3_context=self.context(segment_prompts='{"2":{"audio_transition":"reset"}}'))
        vm, am = self.guider.calls[1][3]['denoise_mask'].unbind()
        self.assertTrue((vm[:, :, :7] == 0).all())
        self.assertTrue((am == 1).all())
        self.assertIn('minimax_keyframes', self.guider.calls[1][2]['positive'][0])

    def test_editing_cut_can_continue_audio_without_continuing_video(self):
        self.run_first(generation_mode='分段编辑', boundary_constraints=True,
                       h3_context=self.context(segment_prompts='{"2":{"transition":"cut"}}'))
        vm, am = self.guider.calls[1][3]['denoise_mask'].unbind()
        self.assertTrue((vm == 1).all())
        self.assertTrue((am[..., :8] == 0).all())

    def test_explicit_soundtrack_overrides_old_audio_mask_in_full_and_segmented_modes(self):
        for mode in ('整段采样', '分段编辑'):
            with self.subTest(mode=mode):
                latent = self.latent()
                video, audio = latent['samples'].unbind()
                audio.fill_(9)
                latent['noise_mask'] = NestedTensor((torch.ones(1, 1, video.shape[2], 1, 1), torch.zeros(1, 1, 1, audio.shape[-1])))
                avae = context_fixtures.RecordingAudioVAE()
                _, output = self.run_first(generation_mode=mode, latent_image=latent, h3_context=self.context(
                    audio_mode='保留指定音轨', audio_vae=avae, soundtrack={'waveform': torch.ones(1, 2, 50), 'sample_rate': 24}))
                self.assertEqual(output['samples'].unbind()[1].count_nonzero(), 0)
                self.assertTrue((audio == 9).all())

    def test_next_scene_keyframe_is_excluded_from_preencoded_first_window(self):
        self.guider.set_conds([[torch.ones(1, 2, 3), {'minimax_keyframes': [
            {'resolved_frame_index': 136, 'latent': torch.full((1, 24, 1, 2, 2), 9.)}]}]], [])
        self.run_first(h3_context=self.context(segment_prompts='{"2":{"transition":"cut"}}'))
        self.assertFalse(self.guider.calls[0][2]['positive'][0].get('minimax_keyframes'))

    def test_legacy_reference_padding_excludes_next_scene_but_retains_audio(self):
        video = torch.ones(1, 24, 42, 2, 2)
        video[:, :, 40:] = 9
        audio = torch.arange(235.).view(1, 1, 1, 235).expand(1, 32, 2, 235)
        self.guider.set_conds([[torch.ones(1, 2, 3), {
            'minimax_refs': [{'kind': 'video_audio', 'latent': video, 'audio_latent': audio}],
            'minimax_keyframes': [{'resolved_frame_index': 0, 'latent': video, 'audio_latent': audio}]}]], [])
        prepared = segment_guider(self.guider, (0, 42, 0, 235), True, scene_cuts=[40])
        condition = prepared.original_conds['positive'][0]
        self.assertTrue((condition['minimax_refs'][0]['latent'] == 1).all())
        self.assertTrue((condition['minimax_keyframes'][0]['latent'] == 1).all())
        self.assertTrue(torch.equal(condition['minimax_refs'][0]['audio_latent'], audio))
        self.assertTrue(torch.equal(condition['minimax_keyframes'][0]['audio_latent'], audio))
        self.assertTrue((video[:, :, 40:] == 9).all())

    def test_new_audio_omits_audio_references_without_mutating_input(self):
        audio = {'waveform': torch.ones(1, 2, 100), 'sample_rate': 24}
        avae = context_fixtures.RecordingAudioVAE()
        context = self.context(audio_vae=avae, reference_audio=audio, audio_mode='生成新声音')
        _encode_segment_conditions(context, (0, 17, 0, 93), (1, 24, 17, 2, 2), 0)
        self.assertEqual(avae.calls, [])
        self.assertIs(context['ref_audios']['ref_audio_0'], audio)

    def test_preserved_soundtrack_is_zero_masked_in_each_window(self):
        audio = {'waveform': torch.ones(1, 2, 50), 'sample_rate': 24}
        avae = context_fixtures.RecordingAudioVAE()
        encode = avae.encode
        avae.encode = lambda waveform: encode(waveform) + 3
        self.guider = fixtures.NativeMaskGuider()
        _, result = self.run_first(h3_context=self.context(audio_mode='保留指定音轨', soundtrack=audio, audio_vae=avae))
        self.assertEqual(len(avae.calls), 1)
        for call in self.guider.calls:
            self.assertEqual(call[3]['denoise_mask'].unbind()[1].count_nonzero(), 0)
        self.assertTrue((result['samples'].unbind()[1] == 3).all())
        self.assertTrue(torch.equal(audio['waveform'], torch.ones(1, 2, 50)))
        with self.assertRaisesRegex(ValueError, '音频 VAE'):
            encode_soundtrack(None, audio, 56)

    def test_refine_respects_source_scene_cuts_even_when_upscaler_drops_metadata(self):
        latent = self.latent()
        video = latent['samples'].unbind()[0]
        video[:, :, :50] = 3
        video[:, :, 50:] = 9
        source = dict(latent, pj_scene_cuts=[50])
        node = fixtures.PJ_SegmentSampler()
        self.nodes.append(node)
        out, x0 = node.execute(self.noise, self.guider, self.sampler, torch.tensor([1., 0.]), latent,
                              segment_plan=self.plan, source_result=source, unload_model_after_sampling=False)
        self.assertEqual(out['pj_scene_cuts'], [50])
        self.assertEqual(x0['pj_scene_cuts'], [50])
        self.assertEqual(out['samples'].unbind()[0].shape, video.shape)
        calls = self.guider.calls
        for _, samples, _, _ in calls:
            window = samples.unbind()[0]
            # Carried same-scene samples can change; opposite-scene source pixels cannot enter.
            self.assertFalse((window == 3).any() and (window == 9).any())
        first_new = next(call for call in calls if (call[1].unbind()[0] == 9).all())
        self.assertIsNone(first_new[3]['denoise_mask'])

    def test_repair_does_not_overwrite_next_scene_with_padding(self):
        baseline = self.latent()
        baseline['pj_scene_cuts'] = [50]
        baseline['samples'].unbind()[0][:, :, 50:] = 9
        chunks = split_scene_windows(plan_segments(107, 603, '按秒自动切分', 6., .92, 3, '', 24.), [50], 107, 603)
        deliveries = scene_deliveries(chunks, delivery_chunks(chunks, self.plan), [50])
        target = next(i for i, bounds in enumerate(chunks) if bounds[0] < 50 < bounds[1])
        out, x0, _ = repair_segments(self.noise, self.guider, self.sampler, torch.tensor([1., 0.]),
                                     baseline, chunks, [target], 12, delivery_bounds=deliveries)
        for value in (out, x0):
            self.assertTrue((value['samples'].unbind()[0][:, :, 50:] == 9).all())
        self.assertFalse((self.guider.calls[0][1].unbind()[0] == 9).any())


if __name__ == '__main__':
    unittest.main()
