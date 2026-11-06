"""Regression cases found by independent review of local revision updates."""

import unittest

import test_local_repair
import test_refine_revision
import torch
from comfy.nested_tensor import NestedTensor


class RevisionReviewTests(unittest.TestCase):
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
