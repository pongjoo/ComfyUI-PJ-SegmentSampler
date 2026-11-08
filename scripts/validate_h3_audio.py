"""Validate H3 relay audio using local weights and an optional fixed first window.

The JSON config supplies model/clip/vae/audio_vae, optional lora/lora_strength,
width/height/frames, steps/scheduler/sampler/seed, prompt/segment_prompts,
reference_images (file paths), and optional prefix_file ([video, audio] tensors).
Runs in the native PromptExecutor without restarting or queuing the local server.
ASR/listening is a separate check; valid tensors do not establish correct speech.
"""

import argparse
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_h3_repair import check_queue, write_json, ValidationServer

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parent.parent


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AudioValidation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(self):
        started = time.perf_counter()
        clip = nodes.CLIPLoader().load_clip(config["clip"], type="minimax")[0]
        vae = nodes.VAELoader().load_vae(config["vae"])[0]
        refs = []
        for filename in config.get("reference_images", []):
            with Image.open(filename) as source:
                refs.append(torch.from_numpy(np.array(ImageOps.exif_transpose(source).convert("RGB"), copy=True)).float().div_(255))
        context, = PJ_H3SegmentContext.prepare(clip, vae, prompt=config["prompt"],
                                             segment_prompts=json.dumps(config.get("segment_prompts", {}), ensure_ascii=False),
                                             reference_mode="素材参考（保持完整）", reference_images=refs or None)
        positive, empty = MiniMaxH3ImageToVideo.execute(clip, vae, "", config["width"], config["height"], config["frames"]).result
        model = nodes.UNETLoader().load_unet(config["model"], "default")[0]
        if config.get("lora"):
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, config["lora"], config.get("lora_strength", 1.))[0]
        if config.get("sage_patch"):
            with patch("server.PromptServer.instance", ValidationServer(), create=True):
                kj = load_module("pj_audio_validation_sage", ROOT.parent / "ComfyUI-KJNodes/nodes/ltxv_nodes.py")
            model = kj.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
        guider = BasicGuider.execute(model, positive).result[0]
        sigmas = BasicScheduler.execute(model, config["scheduler"], config["steps"], 1.).result[0]
        sampler = comfy.samplers.ksampler(config["sampler"])
        video, audio = empty["samples"].unbind()
        chunks = plan_relay_segments(video.shape[2], audio.shape[-1], config.get("chunk_seconds", 6.))
        prefix_parts = torch.load(config["prefix_file"], map_location="cpu", weights_only=True) if config.get("prefix_file") else None
        expected_shapes = ((video.shape[0], video.shape[1], chunks[0][1], *video.shape[-2:]), (*audio.shape[:-1], chunks[0][3]))
        if prefix_parts is not None and (len(prefix_parts) != 2 or any(tuple(part.shape) != shape for part, shape in zip(prefix_parts, expected_shapes))):
            raise ValueError("prefix_file must match the first planned AV window")
        prefix = comfy.nested_tensor.NestedTensor(prefix_parts) if prefix_parts is not None else None

        class PrefixCache(SegmentCache):
            def load(self, index, signature):
                if prefix is not None and index == 0 and index not in self.entries:
                    self.save(index, signature, prefix)
                return super().load(index, signature)

        check_queue(options.server_url)
        cache = PrefixCache()
        case_started = time.perf_counter()
        try:
            result, _ = relay.sample_relay(Noise_RandomNoise(config["seed"]), guider, sampler, sigmas, empty, chunks,
                                           cache, config["seed"], list(range(len(chunks))),
                                           2 if prefix is not None else 1, True, h3_context=context)
            parts = result["samples"].unbind()
            if any(part.shape != original.shape or not torch.isfinite(part).all() for part, original in zip(parts, (video, audio))):
                raise ValueError("Audio relay produced invalid AV tensors")
            if prefix_parts is not None and (not torch.equal(parts[0][:, :, :chunks[0][1]], prefix_parts[0])
                                             or not torch.equal(parts[1][..., :chunks[0][3]], prefix_parts[1])):
                raise ValueError("Audio relay changed the fixed AV prefix")
            torch.save(list(parts), output / "relay.pt")
            for index in range(len(chunks)):
                torch.save(list(cache.load(index, cache.entries[index])[0].unbind()), output / f"relay_window_{index + 1}.pt")
            result_info = {"case": "relay", "sampling_seconds": time.perf_counter() - case_started,
                           "video_shape": list(parts[0].shape), "audio_shape": list(parts[1].shape)}
        finally:
            if cache.directory is not None:
                cache.directory.cleanup()

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        audio_vae = nodes.VAELoader().load_vae(config["audio_vae"])[0]
        check_queue(options.server_url)
        decoded = VAEDecodeAudio.execute(audio_vae, result).result[0]
        waveform = decoded["waveform"][0].movedim(0, -1).cpu().numpy()
        soundfile.write(output / "relay.wav", waveform, decoded["sample_rate"], subtype="FLOAT")
        result_info["sample_rate"] = decoded["sample_rate"]
        result_info["audio_seconds"] = len(waveform) / decoded["sample_rate"]
        write_json(output / "analysis.json", {"status": "pass", "cases": [result_info], "windows": chunks,
                                             "elapsed_seconds": time.perf_counter() - started,
                                             "unconfirmed": "Speech accuracy and voice continuity require separate ASR/listening comparisons."})
        return (str(output / "analysis.json"),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plugin-root", default=str(ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:7188")
    parser.add_argument("--preflight", action="store_true")
    options = parser.parse_args()
    config = json.loads(Path(options.config).read_text(encoding="utf-8-sig"))
    output = Path(options.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    check_queue(options.server_url)
    for filename in filter(None, [config.get("prefix_file"), *config.get("reference_images", [])]):
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    sys.path.insert(0, str(COMFY_ROOT))
    import folder_paths
    import utils.extra_config
    paths = COMFY_ROOT / "extra_model_paths.yaml"
    if paths.exists():
        utils.extra_config.load_extra_path_config(str(paths))
    for key, category in (("model", "diffusion_models"), ("clip", "text_encoders"), ("vae", "vae"), ("audio_vae", "vae"), ("lora", "loras")):
        if config.get(key):
            folder_paths.get_full_path_or_raise(category, config[key])
    if options.preflight:
        print("Local inputs and weights found; queue empty.")
        sys.exit(0)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    import torch
    import soundfile
    import comfy.model_management
    import comfy.nested_tensor
    import comfy.samplers
    import execution
    import nodes
    from comfy_extras.nodes_audio import VAEDecodeAudio
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, Noise_RandomNoise
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
    load_module("pj_audio_validation", Path(options.plugin_root) / "__init__.py", package=True)
    import pj_audio_validation.segment_relay as relay
    from pj_audio_validation.pj_h3_context import PJ_H3SegmentContext
    from pj_audio_validation.segment_sampling import SegmentCache
    from pj_audio_validation.segment_timing import plan_relay_segments
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    nodes.NODE_CLASS_MAPPINGS["PJAudioValidation"] = AudioValidation
    executor = execution.PromptExecutor(ValidationServer(), cache_type=execution.CacheType.NONE, cache_args={"ram": 0, "ram_inactive": 0})
    executor.execute({"test": {"class_type": "PJAudioValidation", "inputs": {}}}, "pj-audio-validation", extra_data={"preview_method": "none"}, execute_outputs=["test"])
    if not executor.success:
        write_json(output / "error.json", {"status_messages": executor.status_messages})
        sys.exit(1)
