"""Regression cases found by independent review of local revision updates."""

import unittest

import test_local_repair
import test_refine_revision
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise


class RevisionReviewTests(unittest.TestCase):
    def test_skipped_video_update_keeps_complete_snapshot_and_can_retry(self):
        for sigmas in (torch.tensor([]), torch.tensor([0.4])):
            with self.subTest(sigma_count=len(sigmas)):
                fixture = test_refine_revision.RefineRevisionTests()
                fixture.setUp()
                try:
                    first, _ = fixture.source.commit(fixture.input, changes=[(0, 107)], audio_changes=[(0, 603)])
                    previous = fixture.run_node(source_result=first)
                    video, audio = fixture.input["samples"].unbind()
                    incoming = {"samples": NestedTensor((video + 1, audio + 3))}
                    latest, _ = fixture.source.commit(incoming, changes=[(0, 107)], audio_changes=[(0, 603)], reset=True)
                    fixture.guider.calls.clear()
                    skipped = fixture.run_node(source_result=latest, latent_image=incoming,
                                               refine_action="更新一采变更", sigmas=sigmas)
                    self.assertEqual(fixture.guider.calls, [])
                    for before, returned, stored in zip(previous, skipped, fixture.node._results.load()):
                        self.assertEqual(returned["pj_revision"], before["pj_revision"])
                        self.assertEqual(stored["pj_revision"], before["pj_revision"])
                        self.assertEqual(returned["pj_refine_source"], before["pj_refine_source"])
                        for old, new, saved in zip(before["samples"].unbind(), returned["samples"].unbind(), stored["samples"].unbind()):
                            self.assertTrue(torch.equal(new, old))
                            self.assertTrue(torch.equal(saved, old))
                    self.assertEqual(fixture.node._source_stamp, previous[0]["pj_refine_source"])
                    retried, _ = fixture.run_node(source_result=latest, latent_image=incoming,
                                                 refine_action="更新一采变更", noise=Noise_RandomNoise(52))
                    self.assertGreater(len(fixture.guider.calls), 0)
                    self.assertFalse(torch.equal(retried["samples"].unbind()[0], previous[0]["samples"].unbind()[0]))
                    self.assertTrue(torch.equal(retried["samples"].unbind()[1], audio + 3))
                    self.assertEqual(retried["pj_refine_source"]["lineage"], latest["pj_revision"]["lineage"])
                finally:
                    fixture.tearDown()

    def test_audio_only_update_does_not_require_sampling_sigmas(self):
        fixture = test_refine_revision.RefineRevisionTests()
        fixture.setUp()
        try:
            first, _ = fixture.source.commit(fixture.input, changes=[(0, 107)], audio_changes=[(0, 603)])
            previous, _ = fixture.run_node(source_result=first)
            video, audio = fixture.input["samples"].unbind()
            incoming = {"samples": NestedTensor((video, audio + 3))}
            latest, _ = fixture.source.commit(incoming, audio_changes=[(0, 603)])
            fixture.guider.calls.clear()
            result = fixture.run_node(source_result=latest, latent_image=incoming,
                                      refine_action="更新一采变更", sigmas=torch.tensor([]))
            self.assertEqual(fixture.guider.calls, [])
            for value in result:
                self.assertTrue(torch.equal(value["samples"].unbind()[1], audio + 3))
                self.assertEqual(value["pj_refine_source"]["version"], latest["pj_revision"]["version"])
            self.assertTrue(torch.equal(result[0]["samples"].unbind()[0], previous["samples"].unbind()[0]))
        finally:
            fixture.tearDown()

    def test_old_completed_first_pass_request_cannot_sample_again_after_new_request(self):
        fixture = test_local_repair.LocalRepairTests()
        fixture.setUp()
        try:
            fixture.run_node(repair_request_id=10)
            latest, _ = fixture.run_node(repair_request_id=11)
            count = len(fixture.guider.calls)
            try:
                fixture.run_node(repair_request_id=10)
            except ValueError:
                pass  # Rejecting a stale request is also a valid outcome.
            self.assertEqual(len(fixture.guider.calls), count)
            self.assertEqual(fixture.node._revisions.load()[0]["pj_revision"], latest["pj_revision"])
        finally:
            fixture.tearDown()

    def test_source_audio_update_is_not_marked_synced_while_old_audio_is_retained(self):
        fixture = test_refine_revision.RefineRevisionTests()
        fixture.setUp()
        try:
            first, _ = fixture.source.commit(fixture.input, changes=[(0, 107)], audio_changes=[(0, 603)])
            fixture.run_node(source_result=first)
            video, audio = fixture.input["samples"].unbind()
            incoming = {"samples": NestedTensor((video + 1, audio + 1))}
            latest, _ = fixture.source.commit(incoming, changes=[(12, 20)], audio_changes=[(0, 603)])
            result, _ = fixture.run_node(source_result=latest, latent_image=incoming,
                                         refine_action="更新一采变更", keep_original_audio=True)
            self.assertEqual(result["pj_refine_source"]["version"], latest["pj_revision"]["version"])
            self.assertTrue(torch.equal(result["samples"].unbind()[1], incoming["samples"].unbind()[1]))
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
