"""Stateful PJ outputs must stay current under every retaining executor cache."""

from contextlib import contextmanager
import copy
from functools import partial
import unittest
from unittest.mock import patch

import torch

import test_stage_execution as stage
from execution import CacheType, PromptExecutor


@contextmanager
def executor_case(cache_type):
    case = stage.ActualPJStageExecutionTests("test_load_and_reuse_actual_first_skip_unused_randomized_sources")
    case.setUp()
    case.executor = PromptExecutor(stage.Server(), cache_type=cache_type,
                                   cache_args={"lru": 100, "ram": 0, "ram_inactive": 0})
    try:
        yield case
    finally:
        case.tearDown()
        case.doCleanups()


def output(case, node):
    return case.executor.caches.outputs.get_local(node).outputs[0][0]


class ExecutionCacheTests(unittest.TestCase):
    def test_loaded_baseline_message_reports_actual_length_without_a_sampled_plan(self):
        with executor_case(CacheType.RAM_PRESSURE) as case:
            prompt = case.native_planner_prompt()
            prompt["baseline"]["inputs"]["tokens"] = 252
            with patch.object(case.executor, "execute", partial(case.executor.execute, extra_data={"client_id": "summary-test"})), patch.object(case.executor.server, "send_sync") as sent:
                case.execute(prompt)
            messages = [call.args[1] for call in sent.call_args_list
                        if call.args[0] == "executed" and call.args[1]["node"] == "first"]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["output"]["pj_sampling_summary"], [{
                "n_chunks": None, "mode": "full", "plan": None, "total_video_tokens": 252}])
            self.assertEqual(case.executor.caches.outputs.get_local("plan").outputs[1][0], 3)
            self.assertEqual(stage.SamplingInputs.calls, 0)

    def test_each_sampler_message_reports_its_actual_plan_instead_of_preview_count(self):
        with executor_case(CacheType.RAM_PRESSURE) as case:
            prompt = case.native_planner_prompt()
            prompt["first"]["inputs"].update(repair_action="原采样方式", generation_mode="分段编辑")
            prompt["original"]["inputs"]["tokens"] = 252
            prompt["second_plan"] = copy.deepcopy(prompt["plan"])
            prompt["second_plan"]["inputs"]["chunk_seconds"] = 8.
            prompt["second"]["inputs"]["segment_plan"] = ["second_plan", 0]
            prompt["stage"]["inputs"]["switch"] = True
            with patch.object(case.executor, "execute", partial(case.executor.execute, extra_data={"client_id": "summary-test"})), patch.object(case.executor.server, "send_sync") as sent:
                case.execute(prompt)
            summaries = {call.args[1]["node"]: call.args[1]["output"]["pj_sampling_summary"][0]
                         for call in sent.call_args_list
                         if call.args[0] == "executed" and call.args[1]["node"] in ("first", "second")}
            self.assertEqual(set(summaries), {"first", "second"})
            for name, count, seconds in (("first", 7, 6.), ("second", 5, 8.)):
                self.assertEqual(summaries[name]["n_chunks"], count)
                self.assertEqual(summaries[name]["total_video_tokens"], 252)
                self.assertEqual(summaries[name]["mode"], "segments")
                self.assertEqual(summaries[name]["plan"]["chunk_seconds"], seconds)
            self.assertEqual(case.executor.caches.outputs.get_local("plan").outputs[1][0], 3)
            self.assertEqual(case.executor.caches.outputs.get_local("second_plan").outputs[1][0], 2)

    def test_empty_sigmas_keep_previous_summary_until_source_update_is_sampled(self):
        with executor_case(CacheType.RAM_PRESSURE) as case:
            prompt = case.actual_prompt("载入回修基准")
            prompt["baseline"]["inputs"]["tokens"] = 107
            prompt["stage"]["inputs"]["switch"] = True
            case.execute(prompt)
            previous = output(case, "second")
            summary = case.executor.caches.outputs.get_local("second").ui["output"]["pj_sampling_summary"]
            prompt["baseline"]["inputs"]["seed"] = 6
            prompt["resources"]["inputs"]["seed"] = 7
            prompt["plan"] = {"class_type": "PJTestPlan", "inputs": {}}
            prompt["second"]["inputs"].update(refine_action="更新一采变更", segment_plan=["plan", 0])
            sampling_inputs = stage.SamplingInputs.execute

            def empty_sigmas(node, seed):
                return (*sampling_inputs(node, seed)[:3], torch.tensor([]))

            with patch.object(stage.SamplingInputs, "execute", empty_sigmas):
                case.execute(prompt)
            latest_source = output(case, "first")["pj_revision"]
            self.assertNotEqual(latest_source["lineage"], previous["pj_refine_source"]["lineage"])
            self.assertEqual(stage.SamplingInputs.guiders[-1].calls, [])
            self.assertEqual(output(case, "second")["pj_revision"], previous["pj_revision"])
            self.assertEqual(case.executor.caches.outputs.get_local("second").ui["output"]["pj_sampling_summary"], summary)
            prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
            prompt["resources"]["inputs"]["seed"] = 8
            case.execute(prompt)
            self.assertGreater(len(stage.SamplingInputs.guiders[-1].calls), 0)
            self.assertEqual(output(case, "second")["pj_refine_source"]["lineage"], latest_source["lineage"])
            updated_summary = case.executor.caches.outputs.get_local("second").ui["output"]["pj_sampling_summary"][0]
            self.assertEqual(updated_summary["n_chunks"], 3)
            self.assertEqual(updated_summary["mode"], "segments")

    def test_first_reuse_returns_current_revision_after_repair(self):
        for mode in (CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE):
            with self.subTest(mode=mode), executor_case(mode) as case:
                prompt = case.actual_prompt("载入回修基准")
                case.execute(prompt)
                reuse = copy.deepcopy(prompt)
                reuse["first"]["inputs"]["repair_action"] = "复用一采结果"
                case.execute(reuse)
                repair = copy.deepcopy(prompt)
                repair["first"]["inputs"].update(repair_action="仅重绘目标段", repair_steps=1)
                case.execute(repair)
                current = output(case, "first")["pj_revision"]
                calls = len(stage.SamplingInputs.guiders[-1].calls)
                case.execute(reuse)
                self.assertEqual(output(case, "first")["pj_revision"], current)
                self.assertEqual(len(stage.SamplingInputs.guiders[-1].calls), calls)

    def test_second_reuse_returns_current_revision_after_repair(self):
        for mode in (CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE):
            with self.subTest(mode=mode), executor_case(mode) as case:
                prompt = case.actual_prompt("载入回修基准")
                prompt["stage"]["inputs"]["switch"] = True
                case.execute(prompt)
                reuse = copy.deepcopy(prompt)
                reuse["first"]["inputs"]["repair_action"] = "复用一采结果"
                reuse["second"]["inputs"]["refine_action"] = "复用高清结果"
                case.execute(reuse)
                repair = copy.deepcopy(reuse)
                repair["second"]["inputs"]["refine_action"] = "仅重做目标段"
                case.execute(repair)
                current = output(case, "second")["pj_revision"]
                calls = len(stage.SamplingInputs.guiders[-1].calls)
                case.execute(reuse)
                self.assertEqual(output(case, "second")["pj_revision"], current)
                self.assertEqual(len(stage.SamplingInputs.guiders[-1].calls), calls)

    def test_identical_nodes_each_establish_their_own_revision(self):
        with executor_case(CacheType.RAM_PRESSURE) as case:
            prompt = case.actual_prompt("载入回修基准")
            prompt["other"] = copy.deepcopy(prompt["first"])
            prompt["other_save"] = {"class_type": "PJTestCapture", "inputs": {"latent": ["other", 0]}}
            case.execute(prompt)
            first = case.executor.caches.objects.get_local("first")
            other = case.executor.caches.objects.get_local("other")
            self.assertIsNotNone(other)
            self.assertTrue(other._revisions.available())
            self.assertNotEqual(first._revisions.metadata["lineage"], other._revisions.metadata["lineage"])
            other._revisions.directory.cleanup()

    def test_identical_normal_request_does_not_resample_or_reset_revisions(self):
        with executor_case(CacheType.RAM_PRESSURE) as case:
            prompt = case.actual_prompt("原采样方式")
            prompt["stage"]["inputs"]["switch"] = True
            case.execute(prompt)
            versions = [output(case, key)["pj_revision"] for key in ("first", "second")]
            calls = len(stage.SamplingInputs.guiders[-1].calls)
            case.execute(prompt)
            self.assertEqual([output(case, key)["pj_revision"] for key in ("first", "second")], versions)
            self.assertEqual(len(stage.SamplingInputs.guiders[-1].calls), calls)

    def test_returning_to_normal_sampling_after_repair_rebuilds_current_result(self):
        for mode in (CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE):
            for name in ("first", "second"):
                with self.subTest(mode=mode, node=name), executor_case(mode) as case:
                    normal = case.actual_prompt("原采样方式" if name == "first" else "载入回修基准")
                    normal["stage"]["inputs"]["switch"] = name == "second"
                    case.execute(normal)
                    if name == "second":
                        normal["first"]["inputs"]["repair_action"] = "复用一采结果"
                        case.execute(normal)
                    original = output(case, name)["samples"].clone()
                    repair = copy.deepcopy(normal)
                    action = "repair_action" if name == "first" else "refine_action"
                    repair[name]["inputs"][action] = "仅重绘目标段" if name == "first" else "仅重做目标段"
                    repair["resources"]["inputs"]["seed"] = 7
                    case.execute(repair)
                    repaired = output(case, name)
                    self.assertFalse(torch.equal(repaired["samples"], original))
                    calls = sum(len(guider.calls) for guider in stage.SamplingInputs.guiders)
                    case.execute(normal)
                    restored = output(case, name)
                    self.assertTrue(torch.equal(restored["samples"], original))
                    self.assertNotEqual(restored["pj_revision"]["lineage"], repaired["pj_revision"]["lineage"])
                    self.assertGreater(sum(len(guider.calls) for guider in stage.SamplingInputs.guiders), calls)
                    calls = sum(len(guider.calls) for guider in stage.SamplingInputs.guiders)
                    case.execute(normal)
                    self.assertEqual(output(case, name)["pj_revision"], restored["pj_revision"])
                    self.assertEqual(sum(len(guider.calls) for guider in stage.SamplingInputs.guiders), calls)

    def test_failed_plan_change_does_not_replace_current_result_summary(self):
        for name in ("first", "second"):
            with self.subTest(node=name), executor_case(CacheType.RAM_PRESSURE) as case:
                prompt = case.actual_prompt("原采样方式")
                prompt["original"]["inputs"]["tokens"] = 107
                prompt["stage"]["inputs"]["switch"] = name == "second"
                case.execute(prompt)
                version = output(case, name)["pj_revision"]
                summary = case.executor.caches.outputs.get_local(name).ui["output"]["pj_sampling_summary"]
                changed = copy.deepcopy(prompt)
                changed["plan"] = {"class_type": "PJTestPlan", "inputs": {}}
                changed[name]["inputs"]["segment_plan"] = ["plan", 0]
                if name == "first":
                    changed[name]["inputs"]["generation_mode"] = "分段编辑"
                else:
                    changed["first"]["inputs"]["repair_action"] = "复用一采结果"
                with patch.object(stage.InputAwareGuider, "sample", side_effect=RuntimeError("deliberate failure")):
                    with self.assertLogs(level="ERROR"), self.assertRaises(AssertionError):
                        case.execute(changed)
                self.assertIn("deliberate failure", str(case.executor.status_messages))
                reused = copy.deepcopy(prompt)
                action = "repair_action" if name == "first" else "refine_action"
                reused[name]["inputs"][action] = "复用一采结果" if name == "first" else "复用高清结果"
                case.execute(reused)
                self.assertEqual(output(case, name)["pj_revision"], version)
                self.assertEqual(case.executor.caches.outputs.get_local(name).ui["output"]["pj_sampling_summary"], summary)

    def test_unload_failure_after_commit_keeps_the_new_result_summary(self):
        for name in ("first", "second"):
            with self.subTest(node=name), executor_case(CacheType.RAM_PRESSURE) as case:
                prompt = case.actual_prompt("原采样方式")
                prompt["original"]["inputs"]["tokens"] = 107
                prompt["stage"]["inputs"]["switch"] = name == "second"
                case.execute(prompt)
                previous = output(case, name)["pj_revision"]
                changed = copy.deepcopy(prompt)
                changed["plan"] = {"class_type": "PJTestPlan", "inputs": {}}
                changed[name]["inputs"].update(segment_plan=["plan", 0], unload_model_after_sampling=True)
                if name == "first":
                    changed[name]["inputs"].update(repair_action="仅重绘目标段", generation_mode="分段编辑")
                else:
                    changed["first"]["inputs"]["repair_action"] = "复用一采结果"
                    changed[name]["inputs"]["refine_action"] = "仅重做目标段"
                with patch("comfy.model_management.unload_all_models", side_effect=RuntimeError("unload failure")):
                    with self.assertLogs(level="ERROR"), self.assertRaises(AssertionError):
                        case.execute(changed)
                self.assertIn("unload failure", str(case.executor.status_messages))
                node = case.executor.caches.objects.get_local(name)
                committed = (node._revisions if name == "first" else node._results).load()[0]
                self.assertEqual(committed["pj_revision"]["version"], previous["version"] + 1)
                reused = copy.deepcopy(prompt)
                action = "repair_action" if name == "first" else "refine_action"
                reused[name]["inputs"][action] = "复用一采结果" if name == "first" else "复用高清结果"
                case.execute(reused)
                self.assertEqual(output(case, name)["pj_revision"], committed["pj_revision"])
                self.assertTrue(torch.equal(output(case, name)["samples"], committed["samples"]))
                summary = case.executor.caches.outputs.get_local(name).ui["output"]["pj_sampling_summary"][0]
                self.assertEqual(summary["n_chunks"], 3)
                self.assertEqual(summary["mode"], "segments")
                self.assertEqual(summary["total_video_tokens"], 107)

    def test_first_repair_failure_reports_the_imported_baseline_without_a_sampled_plan(self):
        with executor_case(CacheType.RAM_PRESSURE) as case:
            prompt = case.native_planner_prompt()
            prompt["baseline"]["inputs"]["tokens"] = 107
            prompt["first"]["inputs"].update(repair_action="仅重绘目标段", generation_mode="分段编辑")
            with patch.object(stage.InputAwareGuider, "sample", side_effect=RuntimeError("first repair failed")):
                with self.assertLogs(level="ERROR"), self.assertRaises(AssertionError):
                    case.execute(prompt)
            self.assertIn("first repair failed", str(case.executor.status_messages))
            node = case.executor.caches.objects.get_local("first")
            imported = node._revisions.load()[0]
            self.assertEqual(imported["pj_revision"]["version"], 1)
            self.assertTrue(torch.all(imported["samples"] == 5))
            prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
            case.execute(prompt)
            self.assertEqual(output(case, "first")["pj_revision"], imported["pj_revision"])
            self.assertEqual(stage.SamplingInputs.guiders[-1].calls, [])
            summary = case.executor.caches.outputs.get_local("first").ui["output"]["pj_sampling_summary"]
            self.assertEqual(summary, [{"n_chunks": None, "mode": "full", "plan": None, "total_video_tokens": 107}])


if __name__ == "__main__":
    unittest.main()
