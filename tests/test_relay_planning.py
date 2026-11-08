import unittest

import test_segments as fixtures
import test_purpose_modes as purpose
from pj_segments_test.segment_timing import frames_for_tokens, plan_relay_segments, plan_segments, tokens_for_frames_floor


class RelayPlanningTests(unittest.TestCase):
    def test_fixed_and_custom_match_three_fifteen_second_windows(self):
        expected = [(0, 107, 0, 603), (100, 207, 567, 1170), (200, 307, 1133, 1737)]
        for mode in ('固定分段数', '自定义切点'):
            for hidden_seconds in (4, 6, 15, 30):
                with self.subTest(mode=mode, seconds=hidden_seconds):
                    chunks = plan_relay_segments(317, 1793, hidden_seconds, split_mode=mode,
                                                num_segments=3, custom_cuts='0-15,15-30,30-45')
                    self.assertEqual(chunks, expected)
        self.assertEqual(plan_relay_segments(317, 1793, 15), expected)

    def test_contiguous_short_ranges_include_eighth_second(self):
        total = tokens_for_frames_floor(9 * 24)
        chunks = plan_relay_segments(total, 348, 15, split_mode='custom', custom_cuts='0-7,7-9')
        self.assertEqual([c[:2] for c in chunks], [(0, 52), (45, 62)])
        self.assertEqual(frames_for_tokens(chunks[0][1]) - frames_for_tokens(chunks[1][0]), 22)

    def test_gap_overlap_order_and_incomplete_budget_are_rejected(self):
        for ranges in ('0-7,8-9', '0-7,6-9', '1-7,7-9', '7-9,0-7', '0-7',
                       '0-10', '0-nan', '', '0-7,7-7'):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                plan_relay_segments(62, 348, 6, split_mode='custom', custom_cuts=ranges)

    def test_nonuniform_budgets_and_cuts_stay_on_native_grid(self):
        for cuts in ((), (2,), (2, 3)):
            chunks = plan_relay_segments(317, 1793, 6, split_mode='custom',
                                        custom_cuts='0-10,10-25,25-45', cut_segments=cuts)
            self.assertEqual(len(chunks), 3)
            for index, (start, end, a_start, a_end) in enumerate(chunks):
                self.assertEqual(start % 5, 0)
                self.assertEqual((end - start) % 5, 2)
                self.assertLess(start, end)
                self.assertTrue(0 <= a_start < a_end <= 1793)
                if index:
                    self.assertEqual(chunks[index - 1][1] - start, 2 if index + 1 in cuts else 7)

    def test_short_segments_fail_instead_of_returning_empty_new_content(self):
        for options in (dict(split_mode='fixed_segments', num_segments=30),
                        dict(split_mode='custom', custom_cuts='0-8.9,8.9-9')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, '合并过短区间'):
                plan_relay_segments(62, 348, 6, **options)

    def test_ordinary_custom_ranges_keep_partial_editing_behavior(self):
        chunks = plan_segments(62, 348, 'custom', 6, .92, 3, '0-7,8-9')
        self.assertEqual(len(chunks), 2)
        result = fixtures.PJ_SegmentPlanner().plan(split_mode='自定义切点', custom_cuts='0-7,8-9')
        preview = result['ui']['preview_data'][0]
        self.assertEqual(preview['n_chunks'], 2)
        self.assertIn('应从7秒开始', preview['relay_preview']['error'])

    def test_imported_result_repair_covers_existing_end_without_shortening(self):
        for mode in ('fixed_segments', 'custom'):
            chunks = plan_relay_segments(307, 1737, 6, complete_timeline=True, split_mode=mode,
                                        custom_cuts='0-15,15-30,30-45')
            self.assertEqual(chunks[-1][1:], (307, chunks[-1][2], 1737))
            self.assertEqual(chunks[0][0], 0)


class RelayNodeTests(unittest.TestCase):
    setUp = purpose.PurposeSamplingTests.setUp
    tearDown = purpose.PurposeSamplingTests.tearDown
    latent = purpose.PurposeSamplingTests.latent
    run_first = purpose.PurposeSamplingTests.run_first

    def test_sampler_and_planner_share_exact_windows(self):
        for mode in ('固定分段数', '自定义切点'):
            self.plan.update(split_mode=mode, num_segments=3, custom_cuts='0-15,15-30,30-45')
            self.guider.calls.clear()
            latent = self.latent(317)
            _, output = self.run_first(latent_image=latent)
            self.assertEqual(len(self.guider.calls), 3)
            video, audio = output['samples'].unbind()
            self.assertEqual((video.shape[2], audio.shape[-1]), (307, 1737))
            preview = fixtures.PJ_SegmentPlanner().plan(latent=latent, **self.plan)['ui']['preview_data'][0]
            relay = preview['relay_preview']
            self.assertEqual(relay['n_chunks'], 3)
            self.assertEqual(relay['total_frames'], 1042)
            self.assertEqual([(c['start_frame'], c['end_frame']) for c in relay['chunks']],
                             [(0, 362), (340, 702), (680, 1042)])
            for call in self.guider.calls[1:]:
                video_mask = call[3]['denoise_mask'].unbind()[0]
                self.assertTrue((video_mask[:, :, :7] == 0).all())
                self.assertTrue((video_mask[:, :, 7:] == 1).all())

    def test_invalid_custom_ranges_fail_before_sampling(self):
        self.plan.update(split_mode='自定义切点', custom_cuts='0-7,8-9')
        with self.assertRaisesRegex(ValueError, '应从7秒开始'):
            self.run_first(latent_image=self.latent(62))
        self.assertEqual(self.guider.calls, [])


if __name__ == '__main__':
    unittest.main()
