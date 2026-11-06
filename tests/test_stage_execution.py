"""Run the native lazy switch through ComfyUI's executor; no H3 weights required."""

import asyncio
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))
import comfy.cli_args
comfy.cli_args.args.cpu = True

import torch
import nodes
from comfy_extras.nodes_logic import SwitchNode
from execution import PromptExecutor, validate_prompt
from test_segments import RecordingGuider, callback, PJ_FirstPass_SegmentSampler, PJ_SegmentSampler, PJ_SegmentPlanner
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
import comfy.samplers
import comfy.model_sampling
from pj_segments_test.segment_revision import changed_intervals


class FirstPass:
    calls = 0
    samples = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT",), "repair_action": (["原采样方式", "复用一采结果"],)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    def __init__(self):
        self.result = None

    def execute(self, seed, repair_action):
        type(self).calls += 1
        if repair_action == "复用一采结果":
            if self.result is None:
                raise ValueError("No committed first-pass result")
        else:
            type(self).samples += 1
            self.result = {"samples": torch.full((1, 1, 1, 1, 1), float(seed)), "revision": type(self).samples}
        return (self.result,)


class Upscale:
    calls = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    def execute(self, latent):
        type(self).calls += 1
        # Deliberately drop revision metadata, as external upscalers may do.
        return ({"samples": latent["samples"] * 2},)


class SecondPass:
    calls = 0
    source_revisions = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "source_result": ("LATENT",), "seed": ("INT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    def execute(self, latent, source_result, seed):
        type(self).calls += 1
        type(self).source_revisions.append(source_result["revision"])
        if not torch.equal(latent["samples"], source_result["samples"] * 2):
            raise ValueError("Upscale and source result refer to different first-pass revisions")
        return ({"samples": latent["samples"] + seed},)


class Capture:
    values = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, latent):
        type(self).values.append(latent["samples"].clone())
        return ()


class Server:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


class LatentSource:
    calls = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"name": ("STRING",), "seed": ("INT",)},
                "optional": {"tokens": ("INT", {"default": 17})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    def execute(self, name, seed, tokens=17):
        type(self).calls[name] = type(self).calls.get(name, 0) + 1
        return ({"samples": torch.full((1, 24, tokens, 2, 2), float(seed))},)


class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


class InputAwareGuider(RecordingGuider):
    def sample(self, noise, latent, sampler, sigmas, **kwargs):
        super().sample(noise, latent, sampler, sigmas, **kwargs)
        return latent + noise * 0.1


class SegmentPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("PJ_SEGMENT_PLAN",)
    FUNCTION = "execute"

    def execute(self):
        return ({"split_mode": "按秒自动切分", "chunk_seconds": 6., "overlap_seconds": 0., "fps": 24.},)


class SamplingInputs:
    calls = 0
    guiders = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT",)}}

    RETURN_TYPES = ("NOISE", "GUIDER", "SAMPLER", "SIGMAS")
    FUNCTION = "execute"

    def execute(self, seed):
        type(self).calls += 1
        guider = InputAwareGuider()
        flow = FlowSampling()
        guider.model_patcher.get_model_object = lambda name: flow
        type(self).guiders.append(guider)
        return Noise_RandomNoise(seed), guider, comfy.samplers.ksampler("euler"), torch.tensor([1., 0.])


class ExecutorTestCase(unittest.TestCase):
    def setUp(self):
        FirstPass.calls = FirstPass.samples = Upscale.calls = SecondPass.calls = 0
        SecondPass.source_revisions = []
        Capture.values = []
        classes = {"PJTestFirst": FirstPass, "PJTestUpscale": Upscale, "PJTestSecond": SecondPass,
                   "PJTestCapture": Capture, "ComfySwitchNode": SwitchNode}
        self.registration = patch.dict(nodes.NODE_CLASS_MAPPINGS, classes)
        self.registration.start()
        self.addCleanup(self.registration.stop)
        self.executor = PromptExecutor(Server(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})
        self.prompt = {
            "first": {"class_type": "PJTestFirst", "inputs": {"seed": 7, "repair_action": "原采样方式"}},
            "upscale": {"class_type": "PJTestUpscale", "inputs": {"latent": ["first", 0]}},
            "second": {"class_type": "PJTestSecond", "inputs": {
                "latent": ["upscale", 0], "source_result": ["first", 0], "seed": 2}},
            "stage": {"class_type": "ComfySwitchNode", "inputs": {
                "switch": False, "on_false": ["first", 0], "on_true": ["second", 0]}},
            "save": {"class_type": "PJTestCapture", "inputs": {"latent": ["stage", 0]}},
        }

    def execute(self, prompt=None, outputs=None):
        prompt = copy.deepcopy(self.prompt if prompt is None else prompt)
        validated = asyncio.run(validate_prompt("stage-test", prompt, outputs))
        self.assertTrue(validated[0], validated)
        self.executor.execute(prompt, "stage-test", execute_outputs=validated[2])
        self.assertTrue(self.executor.success, self.executor.status_messages)


class StageExecutionTests(ExecutorTestCase):
    def test_first_only_does_not_execute_upscale_or_second(self):
        self.execute()
        self.assertEqual((FirstPass.samples, Upscale.calls, SecondPass.calls), (1, 0, 0))
        self.assertEqual(float(Capture.values[-1].item()), 7)

    def test_high_resolution_path_executes_both_stages_and_preserves_source_link(self):
        self.prompt["stage"]["inputs"]["switch"] = True
        self.execute()
        self.assertEqual((FirstPass.samples, Upscale.calls, SecondPass.calls), (1, 1, 1))
        self.assertEqual(SecondPass.source_revisions, [1])
        self.assertEqual(float(Capture.values[-1].item()), 16)

    def test_switch_to_second_reuses_first_even_when_its_seed_changes(self):
        self.execute()
        self.prompt["stage"]["inputs"]["switch"] = True
        self.prompt["first"]["inputs"].update(repair_action="复用一采结果", seed=999)
        self.execute()
        self.assertEqual(FirstPass.calls, 2)
        self.assertEqual((FirstPass.samples, Upscale.calls, SecondPass.calls), (1, 1, 1))
        self.assertEqual(float(Capture.values[-1].item()), 16)
        self.prompt["second"]["inputs"]["seed"] = 3
        self.execute()
        self.assertEqual((FirstPass.samples, Upscale.calls, SecondPass.calls), (1, 1, 2))
        self.assertEqual(float(Capture.values[-1].item()), 17)

    def test_first_repair_does_not_trigger_connected_high_resolution_branch(self):
        self.prompt["stage"]["inputs"]["switch"] = True
        self.execute()
        self.prompt["stage"]["inputs"]["switch"] = False
        self.prompt["first"]["inputs"]["seed"] = 11
        self.execute()
        self.assertEqual((FirstPass.samples, Upscale.calls, SecondPass.calls), (2, 1, 1))
        self.assertEqual(float(Capture.values[-1].item()), 11)

    def test_independent_output_is_a_counterexample_to_lazy_branch_skipping(self):
        self.prompt["extra_preview"] = {"class_type": "PJTestCapture", "inputs": {"latent": ["second", 0]}}
        self.execute()
        self.assertEqual((FirstPass.samples, Upscale.calls, SecondPass.calls), (1, 1, 1))
        self.assertEqual(sorted(float(x.item()) for x in Capture.values), [7, 16])


class ActualPJStageExecutionTests(ExecutorTestCase):
    def setUp(self):
        super().setUp()
        LatentSource.calls = {}
        SamplingInputs.calls = 0
        SamplingInputs.guiders = []
        registry = patch.dict(nodes.NODE_CLASS_MAPPINGS, {
            "PJ_FirstPass_SegmentSampler": PJ_FirstPass_SegmentSampler,
            "PJ_SegmentSampler": PJ_SegmentSampler,
            "PJ_SegmentPlanner": PJ_SegmentPlanner,
            "PJTestSource": LatentSource, "PJTestInputs": SamplingInputs, "PJTestPlan": SegmentPlan})
        registry.start()
        self.addCleanup(registry.stop)
        preview = patch("latent_preview.prepare_callback", callback)
        preview.start()
        self.addCleanup(preview.stop)

    def tearDown(self):
        for name, store_name in (("first", "_revisions"), ("second", "_results")):
            node = self.executor.caches.objects.get_local(name)
            if node is not None:
                for cache in (node._cache, getattr(node, store_name)):
                    if cache.directory is not None:
                        cache.directory.cleanup()

    def actual_prompt(self, action):
        sampling = {"noise": ["resources", 0], "guider": ["resources", 1],
                    "sampler": ["resources", 2], "sigmas": ["resources", 3],
                    "unload_model_after_sampling": False}
        return {
            "resources": {"class_type": "PJTestInputs", "inputs": {"seed": 3}},
            "original": {"class_type": "PJTestSource", "inputs": {"name": "original", "seed": 0}},
            "baseline": {"class_type": "PJTestSource", "inputs": {"name": "baseline", "seed": 5}},
            "first": {"class_type": "PJ_FirstPass_SegmentSampler", "inputs": dict(
                sampling, latent_image=["original", 0], repair_baseline=["baseline", 0],
                repair_action=action, generation_mode="整段采样")},
            "upscale": {"class_type": "PJTestUpscale", "inputs": {"latent": ["first", 0]}},
            "second": {"class_type": "PJ_SegmentSampler", "inputs": dict(
                sampling, latent_image=["upscale", 0], source_result=["first", 0], refine_action="常规精修")},
            "stage": {"class_type": "ComfySwitchNode", "inputs": {
                "switch": False, "on_false": ["first", 0], "on_true": ["second", 0]}},
            "save": {"class_type": "PJTestCapture", "inputs": {"latent": ["stage", 0]}},
        }

    def native_planner_prompt(self, media=False):
        prompt = self.actual_prompt("载入回修基准")
        planner_inputs = {"fps": 24., "split_mode": "按秒自动切分", "chunk_seconds": 6.,
                          "overlap_seconds": 0., "num_segments": 3, "custom_cuts": "0-6,6-12"}
        if media:
            planner_inputs["latent"] = ["original", 0]
        prompt["plan"] = {"class_type": "PJ_SegmentPlanner", "inputs": planner_inputs}
        prompt["first"]["inputs"]["segment_plan"] = ["plan", 0]
        prompt["second"]["inputs"]["segment_plan"] = ["plan", 0]
        return prompt

    def test_load_and_reuse_actual_first_skip_unused_randomized_sources(self):
        prompt = self.actual_prompt("载入回修基准")
        self.execute(prompt)
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        self.assertEqual(SamplingInputs.calls, 0)
        self.assertEqual(Upscale.calls, 0)
        self.assertTrue(torch.all(Capture.values[-1] == 5))
        prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
        for name in ("original", "baseline", "resources"):
            prompt[name]["inputs"]["seed"] = 999
        self.execute(prompt)
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        self.assertEqual(SamplingInputs.calls, 0)
        self.assertTrue(torch.all(Capture.values[-1] == 5))

    def test_only_second_with_actual_pj_reuses_loaded_first(self):
        prompt = self.actual_prompt("载入回修基准")
        self.execute(prompt)
        prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
        prompt["stage"]["inputs"]["switch"] = True
        prompt["original"]["inputs"]["seed"] = 999
        prompt["baseline"]["inputs"]["seed"] = 999
        self.execute(prompt)
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        self.assertEqual(Upscale.calls, 1)
        self.assertEqual(SamplingInputs.calls, 1)
        self.assertEqual(sum(len(guider.calls) for guider in SamplingInputs.guiders), 1)

    def test_actual_first_generation_then_reuse_does_not_regenerate_input(self):
        prompt = self.actual_prompt("原采样方式")
        self.execute(prompt)
        generated = Capture.values[-1].clone()
        self.assertEqual(LatentSource.calls, {"original": 1})
        self.assertEqual(sum(len(guider.calls) for guider in SamplingInputs.guiders), 1)
        prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
        for name in ("original", "baseline", "resources"):
            prompt[name]["inputs"]["seed"] = 999
        self.execute(prompt)
        self.assertEqual(LatentSource.calls, {"original": 1})
        self.assertEqual(SamplingInputs.calls, 1)
        self.assertEqual(Upscale.calls, 0)
        self.assertTrue(torch.equal(generated, Capture.values[-1]))

    def test_actual_second_reuse_skips_all_upstream_generation_and_upscale(self):
        prompt = self.actual_prompt("载入回修基准")
        prompt["stage"]["inputs"]["switch"] = True
        self.execute(prompt)
        refined = Capture.values[-1].clone()
        prompt["second"]["inputs"]["refine_action"] = "复用高清结果"
        prompt["first"]["inputs"]["repair_action"] = "原采样方式"
        for name in ("original", "baseline", "resources"):
            prompt[name]["inputs"]["seed"] = 999
        self.execute(prompt)
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        self.assertEqual(SamplingInputs.calls, 1)
        self.assertEqual(Upscale.calls, 1)
        self.assertEqual(sum(len(guider.calls) for guider in SamplingInputs.guiders), 1)
        self.assertTrue(torch.equal(refined, Capture.values[-1]))

    def test_two_first_only_repairs_then_cumulative_hd_update_and_linked_repair(self):
        prompt = self.actual_prompt("载入回修基准")
        prompt["baseline"]["inputs"]["tokens"] = 107
        prompt["original"]["inputs"]["tokens"] = 107
        prompt["plan"] = {"class_type": "PJTestPlan", "inputs": {}}
        prompt["first"]["inputs"].update(segment_plan=["plan", 0], generation_mode="分段编辑")
        prompt["second"]["inputs"]["segment_plan"] = ["plan", 0]
        prompt["hd_resources"] = {"class_type": "PJTestInputs", "inputs": {"seed": 13}}
        for name, index in (("noise", 0), ("guider", 1), ("sampler", 2), ("sigmas", 3)):
            prompt["second"]["inputs"][name] = ["hd_resources", index]
        prompt["stage"]["inputs"]["switch"] = True
        self.execute(prompt)
        original_hd = Capture.values[-1].clone()
        first = self.executor.caches.objects.get_local("first")
        second = self.executor.caches.objects.get_local("second")
        consumed = dict(second._source_stamp)
        hd_guider = SamplingInputs.guiders[0]
        self.assertEqual(len(hd_guider.calls), 3)
        self.assertEqual(Upscale.calls, 1)

        prompt["stage"]["inputs"]["switch"] = False
        prompt["first"]["inputs"].update(repair_action="仅重绘目标段", run_mode="仅重抽第1段", repair_request_id=1)
        prompt["original"]["inputs"]["seed"] = 101
        prompt["baseline"]["inputs"]["seed"] = 101
        self.execute(prompt)
        self.assertEqual(first._revisions.metadata["version"], 2)
        self.assertEqual(changed_intervals(first._revisions.metadata, consumed), [[0, 38]])
        prompt["first"]["inputs"].update(run_mode="仅重抽第2段", repair_request_id=2)
        prompt["resources"]["inputs"]["seed"] = 7
        self.execute(prompt)
        self.assertEqual(first._revisions.metadata["version"], 3)
        self.assertEqual(changed_intervals(first._revisions.metadata, consumed), [[0, 38], [42, 78]])
        self.assertEqual((Upscale.calls, len(hd_guider.calls)), (1, 3))
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        self.assertTrue(torch.equal(second._results.load()[0]["samples"], original_hd))

        prompt["stage"]["inputs"]["switch"] = True
        prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
        prompt["second"]["inputs"]["refine_action"] = "更新一采变更"
        prompt["resources"]["inputs"]["seed"] = 999
        resources_before = SamplingInputs.calls
        self.execute(prompt)
        updated_hd = Capture.values[-1].clone()
        self.assertEqual(SamplingInputs.calls, resources_before)
        self.assertEqual((Upscale.calls, len(hd_guider.calls)), (2, 5))
        self.assertEqual(second._source_stamp["version"], 3)
        self.assertEqual(changed_intervals(first._revisions.metadata, second._source_stamp), [])
        self.assertEqual(first._revisions.metadata["version"], 3)
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        for start, end in ((38, 42), (78, 107)):
            self.assertTrue(torch.equal(updated_hd[:, :, start:end], original_hd[:, :, start:end]))
        for start, end in ((0, 38), (42, 78)):
            self.assertFalse(torch.equal(updated_hd[:, :, start:end], original_hd[:, :, start:end]))

        prompt["first"]["inputs"].update(repair_action="仅重绘目标段", run_mode="仅重抽第3段", repair_request_id=3)
        self.execute(prompt)
        linked_hd = Capture.values[-1]
        self.assertEqual(first._revisions.metadata["version"], 4)
        self.assertEqual(second._source_stamp["version"], 4)
        self.assertEqual(changed_intervals(first._revisions.metadata, second._source_stamp), [])
        self.assertEqual((Upscale.calls, len(hd_guider.calls)), (3, 6))
        self.assertTrue(torch.equal(linked_hd[:, :, :82], updated_hd[:, :, :82]))
        self.assertFalse(torch.equal(linked_hd[:, :, 82:], updated_hd[:, :, 82:]))
        self.assertEqual(LatentSource.calls, {"baseline": 1})

        prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
        prompt["resources"]["inputs"]["seed"] = 1000
        prompt["hd_resources"]["inputs"]["seed"] = 1000
        resources_before = SamplingInputs.calls
        self.execute(prompt)
        self.assertEqual(SamplingInputs.calls, resources_before)
        self.assertEqual((Upscale.calls, len(hd_guider.calls)), (3, 6))
        self.assertTrue(torch.equal(Capture.values[-1], linked_hd))

    def test_native_rule_only_planner_does_not_force_randomized_sources(self):
        prompt = self.native_planner_prompt()
        self.execute(prompt)
        prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
        prompt["original"]["inputs"]["seed"] = 999
        prompt["baseline"]["inputs"]["seed"] = 999
        self.execute(prompt)
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        self.assertEqual((SamplingInputs.calls, Upscale.calls), (0, 0))

    def test_native_media_planner_is_an_independent_output_counterexample(self):
        prompt = self.native_planner_prompt(media=True)
        validated = asyncio.run(validate_prompt("planner-output-test", copy.deepcopy(prompt), None))
        self.assertEqual(set(validated[2]), {"plan", "save"})
        self.execute(prompt)
        self.assertEqual(LatentSource.calls, {"baseline": 1, "original": 1})
        prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
        prompt["original"]["inputs"]["seed"] = 999
        prompt["baseline"]["inputs"]["seed"] = 999
        self.execute(prompt)
        # Planner previews are independent outputs: normal queueing still needs their media.
        self.assertEqual(LatentSource.calls, {"baseline": 1, "original": 2})
        self.assertEqual((SamplingInputs.calls, Upscale.calls), (0, 0))
        self.assertTrue(torch.all(Capture.values[-1] == 5))

    def test_partial_output_skips_media_planner_when_real_samplers_reuse_results(self):
        prompt = self.native_planner_prompt(media=True)
        self.execute(prompt, outputs=["save"])
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        prompt["first"]["inputs"]["repair_action"] = "复用一采结果"
        prompt["original"]["inputs"]["seed"] = 999
        prompt["baseline"]["inputs"]["seed"] = 999
        self.execute(prompt, outputs=["save"])
        self.assertEqual(LatentSource.calls, {"baseline": 1})
        self.assertEqual(SamplingInputs.calls, 0)
        prompt["stage"]["inputs"]["switch"] = True
        self.execute(prompt, outputs=["save"])
        self.assertEqual(LatentSource.calls, {"baseline": 1, "original": 1})
        refined = Capture.values[-1].clone()
        prompt["second"]["inputs"]["refine_action"] = "复用高清结果"
        prompt["original"]["inputs"]["seed"] = 1000
        self.execute(prompt, outputs=["save"])
        self.assertEqual(LatentSource.calls, {"baseline": 1, "original": 1})
        self.assertEqual((SamplingInputs.calls, Upscale.calls), (1, 1))
        self.assertTrue(torch.equal(Capture.values[-1], refined))


class StageWorkflowTests(unittest.TestCase):
    def test_example_links_source_revision_and_has_one_shared_video_output(self):
        path = ROOT / "workflows" / "局部回修_执行阶段示例.json"
        workflow = json.loads(path.read_text(encoding="utf-8"))
        by_id = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        self.assertEqual(len(by_id), len(workflow["nodes"]))
        self.assertEqual(len(links), len(workflow["links"]))
        for link_id, source, source_slot, target, target_slot, kind in links.values():
            self.assertIn(link_id, by_id[source]["outputs"][source_slot]["links"])
            self.assertEqual(by_id[target]["inputs"][target_slot]["link"], link_id)
            self.assertTrue({kind, "*"}.intersection(by_id[source]["outputs"][source_slot]["type"].split(",")))
            self.assertTrue({kind, "*"}.intersection(by_id[target]["inputs"][target_slot]["type"].split(",")))
        switch = next(node for node in by_id.values() if node["type"] == "ComfySwitchNode")
        inputs = {value["name"]: links[value["link"]] for value in switch["inputs"] if value["link"] is not None}
        self.assertEqual(by_id[inputs["on_false"][1]]["type"], "PJ_FirstPass_SegmentSampler")
        self.assertEqual(by_id[inputs["on_true"][1]]["type"], "PJ_SegmentSampler")
        second = by_id[inputs["on_true"][1]]
        revision_link = next(value["link"] for value in second["inputs"] if value["name"] == "source_result")
        self.assertEqual(links[revision_link][1], inputs["on_false"][1])
        self.assertEqual(sum(node["type"] == "VHS_VideoCombine" for node in by_id.values()), 1)
        planner = next(node for node in by_id.values() if node["type"] == "PJ_SegmentPlanner")
        self.assertTrue(all(value["link"] is None for value in planner["inputs"]))
        split_id = links[switch["outputs"][0]["links"][0]][3]
        self.assertEqual(by_id[split_id]["type"], "LTXVSeparateAVLatent")
        for output in by_id[split_id]["outputs"]:
            self.assertEqual(len(output["links"]), 1)
            decoder_id = links[output["links"][0]][3]
            self.assertIn(by_id[decoder_id]["type"], ("VAEDecode", "VAEDecodeAudio"))


if __name__ == "__main__":
    unittest.main()
