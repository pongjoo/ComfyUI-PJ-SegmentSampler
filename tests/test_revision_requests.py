import unittest
from unittest.mock import patch

import test_segments
import torch
from pj_segments_test.segment_revision import RevisionStore


class RequestRevisionTests(unittest.TestCase):
    def setUp(self):
        self.store = RevisionStore()
        self.latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}

    def tearDown(self):
        if self.store.directory is not None:
            self.store.directory.cleanup()

    def test_current_request_reuses_but_prior_completed_request_is_stale(self):
        self.assertFalse(self.store.check_request("A"))
        self.store.commit(self.latent, request_key="A")
        self.assertTrue(self.store.check_request("A"))
        self.assertFalse(self.store.check_request("B"))
        self.store.commit(self.latent, request_key="B")
        self.assertTrue(self.store.check_request("B"))
        with self.assertRaisesRegex(ValueError, "请求已过期"):
            self.store.check_request("A")

    def test_failed_atomic_commit_does_not_consume_request_or_clear_old_history(self):
        self.store.commit(self.latent, request_key="A")
        with patch("pj_segments_test.segment_revision.os.replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.store.commit(self.latent, request_key="B", reset=True)
        self.assertTrue(self.store.check_request("A"))
        self.assertFalse(self.store.check_request("B"))
        self.store.commit(self.latent, request_key="B")
        with self.assertRaisesRegex(ValueError, "请求已过期"):
            self.store.check_request("A")

    def test_reset_creates_a_new_request_scope_and_recent_history_is_bounded(self):
        for index in range(130):
            self.store.commit(self.latent, request_key=str(index))
        self.assertFalse(self.store.check_request("0"))
        with self.assertRaisesRegex(ValueError, "请求已过期"):
            self.store.check_request("2")
        self.assertTrue(self.store.check_request("129"))
        self.store.commit(self.latent, reset=True)
        self.assertFalse(self.store.check_request("129"))
        self.assertFalse(self.store.check_request("2"))

    def test_nonfinite_primary_or_secondary_never_replaces_committed_snapshot(self):
        before, _ = self.store.commit(self.latent, request_key="A")
        for number in (float("nan"), float("inf"), -float("inf")):
            invalid = {"samples": torch.full_like(self.latent["samples"], number)}
            for primary, secondary in ((invalid, None), (self.latent, invalid)):
                with self.assertRaisesRegex(ValueError, "非有限值"):
                    self.store.commit(primary, secondary, request_key="B", reset=True)
                self.assertEqual(self.store.load()[0]["pj_revision"], before["pj_revision"])
                self.assertTrue(self.store.check_request("A"))
                self.assertFalse(self.store.check_request("B"))


if __name__ == "__main__":
    unittest.main()
