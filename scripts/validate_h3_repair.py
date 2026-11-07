"""Run a small real H3 baseline/local-repair comparison in an isolated process.

Uses existing local weights and ComfyUI's own PromptExecutor. It never submits a
workflow to, restarts, or changes the running ComfyUI server. --preflight only
checks paths and the server queue; --executor-smoke-test runs a CPU-only harness.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parent.parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plugin-root", default=str(ROOT), help="Plugin checkout to validate; permits before/after comparisons with the same harness.")
    parser.add_argument("--model", default="海螺/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors")
    parser.add_argument("--clip", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    parser.add_argument("--vae", default="minimax_h3_video_vae_int8_convrot.safetensors")
    parser.add_argument("--lora", default="hailuo/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", help="Empty string disables LoRA.")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frames", type=int, default=73)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--scheduler", default="beta")
    parser.add_argument("--sampler", default="er_sde")
    parser.add_argument("--seed", type=int, default=481516)
    parser.add_argument("--repair-seed", type=int, default=481517)
    parser.add_argument("--denoise", type=float, default=0.4)
    parser.add_argument("--denoises", help="Comma-separated candidate strengths, all using the same untouched baseline.")
    parser.add_argument("--seeds", help="Comma-separated candidate seeds.")
    parser.add_argument("--baseline-file", help="Reuse a previous before.pt containing ordinary [video, audio] tensors.")
    parser.add_argument("--boundary-tokens", type=int, default=2)
    parser.add_argument("--target-segment", type=int, default=2)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--overlap-seconds", type=float, default=0.,
                        help="Requested overlap for local-repair and refinement plans.")
    parser.add_argument("--prompt", default="A locked-off tripod shot of a single red toy car moving slowly from left to right on a plain wooden table. The beige wall and soft indoor lighting remain completely steady. One continuous shot, no cuts, no camera movement.")
    parser.add_argument("--prompt-file", help="Read the prompt verbatim from a local UTF-8 text file.")
    parser.add_argument("--reference-video", help="Local source video, resampled to 24 FPS and cropped to --frames.")
    parser.add_argument("--reference-video-start", type=float, default=0., help="Source-video start time in seconds.")
    parser.add_argument("--reference-image", action="append", default=[], help="Local product reference; repeat for multiple images of the same dimensions.")
    parser.add_argument("--mask-file", help="Optional torch-saved [frames,height,width] pixel mask for an imported --baseline-file.")
    parser.add_argument("--extra-model-paths", action="append", default=[])
    parser.add_argument("--server-url", default="http://127.0.0.1:7188")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--executor-smoke-test", action="store_true")
    parser.add_argument("--relay-smoke", action="store_true", help="Validate native H3 motion-context relay windows instead of local repaint.")
    parser.add_argument("--relay-chunk-seconds", type=float, default=39 / 24,
                        help="Requested relay window duration; backend resolves the H3 frame grid.")
    parser.add_argument("--trace-conditions", action="store_true", help="Save relay conditions and noise for numerical before/after checks.")
    parser.add_argument("--refine-smoke", action="store_true", help="Validate full latent upscale, second pass, then a linked local update. Requires the installed Easy 3D upscaler node.")
    parser.add_argument("--upscaler-model", default="minimax_h3_latent_upscaler_3d_bf16.safetensors")
    parser.add_argument("--refine-width", type=int, default=640)
    parser.add_argument("--refine-height", type=int, default=384)
    parser.add_argument("--refine-steps", type=int, default=3)
    return parser.parse_args()


def check_queue(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("--server-url must identify the local ComfyUI server.")
    with urllib.request.urlopen(url.rstrip("/") + "/queue", timeout=5) as response:
        queue = json.load(response)
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("The local ComfyUI queue is not empty; wait before starting independent GPU validation.")
    return {"queue_running": [], "queue_pending": []}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reference_inputs(options):
    """Read local media only; use the same 24 FPS window for text and video conditions."""
    video, images = None, []
    if options.reference_video:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("--reference-video requires the existing FFmpeg executable on PATH")
        command = [ffmpeg, "-v", "error", "-nostdin", "-ss", str(options.reference_video_start),
                   "-i", str(Path(options.reference_video).resolve()), "-map", "0:v:0",
                   "-vf", f"fps=24,scale={options.width}:{options.height}:flags=lanczos",
                   "-frames:v", str(options.frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        decoded = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        expected = options.frames * options.height * options.width * 3
        if len(decoded) != expected:
            raise ValueError(f"Reference video does not cover {options.frames} frames at 24 FPS from the requested start")
        array = np.frombuffer(decoded, dtype=np.uint8).reshape(options.frames, options.height, options.width, 3).copy()
        video = torch.from_numpy(array).float().div_(255)
    for filename in options.reference_image:
        with Image.open(filename) as image:
            array = np.array(ImageOps.exif_transpose(image).convert("RGB"), copy=True)
        images.append(torch.from_numpy(array).float().div_(255))
    if images and any(image.shape != images[0].shape for image in images):
        raise ValueError("PJ_H3_CONTEXT reference images must share dimensions; prepare a matching image batch first")
    return video, torch.stack(images) if images else None


def reference_report():
    has_references = bool(options.reference_video or options.reference_image)
    return {"video": str(Path(options.reference_video).resolve()) if options.reference_video else None,
            "video_start_seconds": options.reference_video_start, "video_frames": options.frames if options.reference_video else 0,
            "video_fps": 24, "images": [str(Path(path).resolve()) for path in options.reference_image],
            "pixel_mask": str(Path(options.mask_file).resolve()) if options.mask_file else None,
            "per_window_h3_context": has_references,
            "limitations": (["No SAM/spatial mask was supplied; this is reference-media and linked-refinement validation, not the complete SAM editing workflow."]
                            if has_references and not options.mask_file else [])}


class ValidationServer:
    """The in-process executor's event sink; no sockets or server are created."""
    client_id = None
    last_node_id = None
    sockets_metadata = {}

    def send_sync(self, event, data, sid=None):
        if event in ("execution_error", "execution_interrupted"):
            logging.error("%s: %s", event, data)

    def queue_updated(self):
        pass


class ValidationRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "validation"

    def run(self):
        if options.executor_smoke_test:
            # Verify the actual executor's inference setup without loading any weights.
            device = comfy.model_management.get_torch_device()
            if not torch.is_inference_mode_enabled() or device.type != "cpu":
                raise RuntimeError("CPU executor smoke test used an unexpected execution context")
            write_json(output / "analysis.json", {"status": "pass", "executor_inference_context": True,
                                                  "execution_device": str(device),
                                                  "cuda_initialized": torch.cuda.is_initialized(), "real_h3_sampling": False})
            return (str(output / "analysis.json"),)
        return (run_comparison(),)


def run_comparison():
    started = time.perf_counter()
    check_queue(options.server_url)
    logging.info("Loading native H3 CLIP, VAE, model and optional LoRA")
    clip = nodes.CLIPLoader().load_clip(options.clip, type="minimax")[0]
    vae = nodes.VAELoader().load_vae(options.vae)[0]
    reference_video, reference_images = reference_inputs(options)
    h3_context = None
    if reference_video is not None or reference_images is not None:
        h3_context, = PJ_H3SegmentContext().prepare(
            clip, vae=vae, prompt=options.prompt, reference_video=reference_video,
            reference_images=reference_images, motion_context=True)
    prepared = MiniMaxH3ImageToVideo.execute(clip, vae, options.prompt,
                                            options.width, options.height, options.frames)
    positive, empty = prepared.result
    model = nodes.UNETLoader().load_unet(options.model, "default")[0]
    if options.lora:
        model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, options.lora, options.lora_strength)[0]
    guider, = BasicGuider.execute(model, positive).result
    sampler = comfy.samplers.ksampler(options.sampler)
    sigmas, = BasicScheduler.execute(model, options.scheduler, options.steps, 1.0).result
    check_queue(options.server_url)
    if options.relay_smoke:
        return run_relay_comparison(clip, vae, guider, sampler, sigmas, empty, started, h3_context)
    if options.baseline_file:
        logging.info("Reusing fixed baseline %s", options.baseline_file)
        saved = torch.load(options.baseline_file, weights_only=True, map_location="cpu")
        expected = empty["samples"].unbind()
        if not isinstance(saved, (tuple, list)) or len(saved) != 2 or any(
                not isinstance(part, torch.Tensor) or part.shape != target.shape for part, target in zip(saved, expected)):
            raise ValueError("--baseline-file must contain [video, audio] matching the requested dimensions and duration")
        baseline = {"samples": comfy.nested_tensor.NestedTensor(saved)}
    else:
        logging.info("Generating baseline with native SamplerCustomAdvanced")
        full_guider = guider
        if h3_context is not None:
            video, audio = empty["samples"].unbind()
            full_guider = prepare_segment_guider(guider, h3_context, (0, video.shape[2], 0, audio.shape[-1]), video.shape)
        baseline, _ = SamplerCustomAdvanced.execute(Noise_RandomNoise(options.seed), full_guider, sampler, sigmas, empty).result
        baseline["samples"] = baseline["samples"].cpu()
    if options.mask_file:
        pixel_mask = torch.load(options.mask_file, weights_only=True, map_location="cpu")
        if not isinstance(pixel_mask, torch.Tensor):
            raise ValueError("--mask-file must contain a pixel-mask tensor")
        baseline, = PJ_H3EditPrepare().prepare(baseline, {"samples": baseline["samples"].unbind()[0]}, pixel_mask)
    before = list(baseline["samples"].unbind())
    torch.save(before, output / "before.pt")
    if options.refine_smoke:
        return run_refine_comparison(vae, guider, sampler, sigmas, empty, baseline, started, h3_context)
    plan = dict(split_mode="固定分段数", num_segments=options.num_segments, overlap_seconds=options.overlap_seconds,
                chunk_seconds=6., custom_cuts="", fps=24., align_51_frames=False)
    chunks, _ = resolve_plan(baseline["samples"], plan, "固定分段数", 6., 0.,
                             options.num_segments, "", 24., False)
    if not 1 <= options.target_segment <= len(chunks):
        raise ValueError("--target-segment is outside the resolved plan")
    actual_frames = frames_for_tokens(before[0].shape[2])
    deliveries = delivery_chunks(chunks, plan)
    target = deliveries[options.target_segment - 1]
    left = target[0] + (options.boundary_tokens if target[0] else 0)
    right = target[1] - (options.boundary_tokens if target[1] < before[0].shape[2] else 0)
    protected = [[target[0], left]] if left > target[0] else []
    if right < target[1]:
        protected.append([right, target[1]])
    denoises = list(dict.fromkeys(float(value) for value in options.denoises.split(","))) if options.denoises else [options.denoise]
    seeds = list(dict.fromkeys(int(value) for value in options.seeds.split(","))) if options.seeds else [options.repair_seed]
    if any(not 0 <= value <= 1 for value in denoises):
        raise ValueError("All candidate denoise strengths must be in [0, 1]")
    candidates = []
    matrix = len(denoises) * len(seeds) > 1
    for denoise in denoises:
        for seed in seeds:
            check_queue(options.server_url)
            candidate = output / f"denoise_{denoise:g}_seed_{seed}" if matrix else output
            candidate.mkdir(exist_ok=True)
            logging.info("Repainting segment %d: denoise %.3f, seed %d", options.target_segment, denoise, seed)
            # Every candidate owns a fresh revision store and starts from the same baseline.
            repair = PJ_FirstPass_SegmentSampler()
            after_latent, _ = repair.execute(
                Noise_RandomNoise(seed), guider, sampler, sigmas, empty,
                segment_plan=plan, run_mode=f"仅重抽第{options.target_segment}段",
                repair_action="仅重绘目标段", repair_denoise=denoise,
                repair_scheduler=options.scheduler, repair_steps=options.steps,
                repair_boundary_tokens=options.boundary_tokens, repair_baseline=baseline,
                repair_request_id=1, unload_model_after_sampling=False, h3_context=h3_context,
                **{f"seg{options.target_segment}_seed_offset": 0})
            torch.save([item.cpu() for item in after_latent["samples"].unbind()], candidate / "after.pt")
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "reference_inputs": reference_report(),
                "baseline": {"source": options.baseline_file or "native SamplerCustomAdvanced", "model": options.model,
                             "clip": options.clip, "vae": options.vae, "lora": options.lora,
                             "seed": options.seed, "repair_seed": seed, "repair_seed_offset": 0,
                             "scheduler": options.scheduler, "steps": options.steps,
                             "sampler": options.sampler, "denoise": denoise, "prompt": options.prompt},
                "streams": {"video": {"time_dim": 2, "editable_intervals": [[left, right]] if denoise > 0 else [],
                                        "protected_intervals": protected},
                            "audio": {"time_dim": 3, "editable_intervals": []}},
                "timeline": {"total_frames": actual_frames, "delivered_intervals": [[0, actual_frames]]},
                "sampling_windows": [list(item) for item in chunks],
                "delivery_windows": [list(item) for item in deliveries], "target_segment": options.target_segment,
                "repair_sigmas": repair_sigmas(guider, options.scheduler, options.steps, denoise).tolist(),
                "revision": after_latent.get("pj_revision"),
            }
            write_json(candidate / "manifest.json", manifest)
            candidates.append((candidate, manifest))
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    check_queue(options.server_url)
    logging.info("Decoding complete before/after timelines with the same video VAE")
    before_frames = nodes.VAEDecode().decode(vae, baseline)[0].cpu()
    torch.save(before_frames, output / "before_frames.pt")
    results = []
    for candidate, manifest in candidates:
        check_queue(options.server_url)
        after = torch.load(candidate / "after.pt", weights_only=True, map_location="cpu")
        after_frames = nodes.VAEDecode().decode(vae, {"samples": comfy.nested_tensor.NestedTensor(after)})[0].cpu()
        torch.save(after_frames, candidate / "after_frames.pt")
        report = analyze_outputs(before, after, manifest, before_frames, after_frames,
                                 edit_masks=baseline.get("noise_mask"))
        report["real_h3_sampling"] = True
        report["repair_sampling_executed"] = manifest["baseline"]["denoise"] > 0
        if manifest["baseline"]["denoise"] > 0 and not report.get("streams", {}).get("video", {}).get("changed_steps"):
            report["hard_failures"].append("Nonzero-denoise repair did not change any video latent step")
            report["status"] = "fail"
        write_json(candidate / "analysis.json", report)
        results.append({"directory": str(candidate), "status": report["status"], "hard_failures": report["hard_failures"],
                        "denoise": manifest["baseline"]["denoise"], "seed": manifest["baseline"]["repair_seed"],
                        "changed_video_steps": report.get("streams", {}).get("video", {}).get("changed_steps", [])})
    summary = {"status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
               "candidates": results, "real_h3_sampling": True,
               "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "reference_inputs": reference_report(),
               "unconfirmed": ["Automated data checks do not certify perceptual seam quality."]}
    write_json(output / "summary.json", summary)
    if matrix:
        write_json(output / "analysis.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return str(output / "analysis.json")


def run_relay_comparison(clip, vae, guider, sampler, sigmas, empty, started, context=None):
    if context is None:
        context, = PJ_H3SegmentContext().prepare(clip, vae=vae, prompt=options.prompt, motion_context=True)
    plan = dict(split_mode="按秒自动切分", num_segments=3, overlap_seconds=22 / 24,
                chunk_seconds=options.relay_chunk_seconds, custom_cuts="", fps=24., align_51_frames=False)
    node = PJ_FirstPass_SegmentSampler()
    logging.info("Sampling native H3 motion-context relay with %.3f-second requested windows and 22-frame context",
                 options.relay_chunk_seconds)
    phases = []
    original_sample = type(guider).sample
    original_encode = clip.encode_from_tokens_scheduled

    def timed(kind, function, *args, **kwargs):
        begin = time.perf_counter()
        value = function(*args, **kwargs)
        phases.append({"kind": kind, "seconds": time.perf_counter() - begin})
        return value

    def sample(instance, *args, **kwargs):
        if options.trace_conditions:
            index = sum(item["kind"] == "sample" for item in phases)
            saved = {name: [{key: value for key, value in condition.items() if key != "uuid"}
                            for condition in conditions] for name, conditions in instance.original_conds.items()}
            torch.save(saved, output / f"conditions-{index}.pt")
            torch.save(list(args[0].unbind()), output / f"noise-{index}.pt")
        return timed("sample", original_sample, instance, *args, **kwargs)

    def encode(*args, **kwargs):
        return timed("encode", original_encode, *args, **kwargs)

    sampling_started = time.perf_counter()
    with patch.object(type(guider), "sample", sample), patch.object(clip, "encode_from_tokens_scheduled", encode):
        result, first = node.execute(Noise_RandomNoise(options.seed), guider, sampler, sigmas, empty,
                                     segment_plan=plan, generation_mode="22帧接力", h3_context=context,
                                     unload_model_after_sampling=False)
    sampling_seconds = time.perf_counter() - sampling_started
    video, audio = result["samples"].unbind()
    chunks = plan_relay_segments(video.shape[2], audio.shape[-1], options.relay_chunk_seconds)
    failures, overlap_checks = [], []
    previous, written = None, [0, 0]
    # Diagnostic read-only cache inspection verifies actual sampled windows, not only the plan.
    for index, bounds in enumerate(chunks):
        saved = node._cache.load(index, node._cache.entries[index])
        window = saved[0].unbind()
        if previous is not None:
            previous_bounds, previous_video = previous
            overlap = previous_bounds[1] - bounds[0]
            same = torch.equal(previous_video[:, :, -overlap:], window[0][:, :, :overlap])
            overlap_checks.append({"segment": index + 1, "video_tokens": overlap, "equal": same})
            if not same:
                failures.append(f"Segment {index + 1} did not inherit its preceding overlap")
        for stream, (part, full, axis) in enumerate(zip(window, (video, audio), (2, 3))):
            start, end = bounds[stream * 2:stream * 2 + 2]
            length = end - written[stream]
            if not torch.equal(full.narrow(axis, written[stream], length),
                               part.narrow(axis, written[stream] - start, length)):
                failures.append(f"Segment {index + 1} stream {stream} differs from its delivered window")
            written[stream] = end
        previous = bounds, window[0]
    target_video, target_audio = empty["samples"].unbind()
    if video.shape != target_video.shape or audio.shape != target_audio.shape:
        failures.append("Relay changed the requested AV timeline shape")
    if not all(torch.isfinite(part).all().item() for part in (video, audio)):
        failures.append("Relay contains non-finite values")
    if not torch.equal(video[:, :, :chunks[0][1]], first["samples"].unbind()[0]):
        failures.append("First segment ownership was not preserved")
    torch.save([video.cpu(), audio.cpu()], output / "relay.pt")
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    check_queue(options.server_url)
    frames = nodes.VAEDecode().decode(vae, result)[0].cpu()
    torch.save(frames, output / "relay_frames.pt")
    actual_frames = frames_for_tokens(video.shape[2])
    quality = frame_metrics(frames, actual_frames)
    report = {"status": "fail" if failures else "pass", "hard_failures": failures,
              "real_h3_sampling": True, "motion_context": True, "window_count": len(chunks),
              "requested_chunk_seconds": options.relay_chunk_seconds,
              "windows": [list(bounds) for bounds in chunks], "overlaps": overlap_checks,
              "video_shape": list(video.shape), "audio_shape": list(audio.shape),
              "decoded_frames": frames.shape[0], "quality_clues": quality,
              "elapsed_seconds": time.perf_counter() - started,
              "sampling_seconds": sampling_seconds, "phases": phases,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "reference_inputs": reference_report(),
              "unconfirmed": ["This smoke test checks the native conditioning path and timeline integrity, not perceptual seam quality."]}
    write_json(output / "analysis.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "quality_clues"}, ensure_ascii=False))
    return str(output / "analysis.json")


def run_refine_comparison(vae, guider, sampler, sigmas, empty, baseline, started, h3_context=None):
    plan = dict(split_mode="固定分段数", num_segments=options.num_segments, overlap_seconds=options.overlap_seconds,
                chunk_seconds=6., custom_cuts="", fps=24., align_51_frames=False)
    first = PJ_FirstPass_SegmentSampler()
    original, _ = first.execute(None, None, None, None, None,
                                repair_action="载入回修基准", repair_baseline=baseline)
    refine_sigmas = repair_sigmas(guider, options.scheduler, options.refine_steps, options.denoise)
    second = PJ_SegmentSampler()

    def upscale(latent):
        check_queue(options.server_url)
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        # Exercise the installed node's public API; PJ itself has no Easy dependency.
        value, = MiniMaxH3EasyLatentUpscaler3D.execute(
            latent, options.upscaler_model,
            {"mode": "target dimensions", "width": options.refine_width, "height": options.refine_height},
            align=32, enable_chunking=True, device="cuda", precision="bf16").result
        return value

    logging.info("Upscaling the entire baseline latent through the installed public 3D upscaler")
    old_input = upscale(original)
    check_queue(options.server_url)
    logging.info("Generating the complete high-resolution second-pass baseline")
    old_hd, old_x0 = second.execute(
        Noise_RandomNoise(options.seed), guider, sampler, refine_sigmas, old_input,
        segment_plan=plan, source_result=original, keep_original_audio=True,
        unload_model_after_sampling=False, h3_context=h3_context)
    logging.info("Repainting the low-resolution target and retaining its source revision")
    check_queue(options.server_url)
    latest, _ = first.execute(
        Noise_RandomNoise(options.repair_seed), guider, sampler, sigmas, empty,
        segment_plan=plan, run_mode=f"仅重抽第{options.target_segment}段",
        repair_action="仅重绘目标段", repair_denoise=options.denoise,
        repair_scheduler=options.scheduler, repair_steps=options.steps,
        repair_boundary_tokens=options.boundary_tokens, repair_request_id=1,
        unload_model_after_sampling=False, h3_context=h3_context,
        **{f"seg{options.target_segment}_seed_offset": 0})
    logging.info("Upscaling the complete revised latent before updating only changed HDR intervals")
    new_input = upscale(latest)
    check_queue(options.server_url)
    new_hd, new_x0 = second.execute(
        Noise_RandomNoise(options.seed), guider, sampler, refine_sigmas, new_input,
        segment_plan=plan, source_result=latest, refine_action="更新一采变更",
        repair_boundary_tokens=options.boundary_tokens, keep_original_audio=True,
        unload_model_after_sampling=False, h3_context=h3_context)
    check_queue(options.server_url)
    repeated, _ = second.execute(None, None, None, None, None,
                                 source_result=latest, refine_action="更新一采变更")
    before, after = [list(value["samples"].unbind()) for value in (old_hd, new_hd)]
    intervals = changed_intervals(latest["pj_revision"], old_hd["pj_refine_source"])
    chunks, _ = resolve_plan(baseline["samples"], plan, "固定分段数", 6., 0.,
                             options.num_segments, "", 24., False)
    actual_frames = frames_for_tokens(before[0].shape[2])
    manifest = {
        "reference_inputs": reference_report(),
        "baseline": {"source": options.baseline_file or "native SamplerCustomAdvanced",
                     "model": options.model, "clip": options.clip, "vae": options.vae,
                     "lora": options.lora, "lora_strength": options.lora_strength,
                     "seed": options.seed, "repair_seed": options.repair_seed,
                     "sampler": options.sampler, "scheduler": options.scheduler,
                     "repair_steps": options.steps, "refine_steps": options.refine_steps,
                     "denoise": options.denoise, "boundary_tokens": options.boundary_tokens,
                     "overlap_seconds": options.overlap_seconds,
                     "prompt": options.prompt},
        "streams": {"video": {"time_dim": 2, "editable_intervals": intervals},
                    "audio": {"time_dim": 3, "editable_intervals": []}},
        "timeline": {"total_frames": actual_frames, "delivered_intervals": [[0, actual_frames]]},
        "sampling_windows": [list(item) for item in chunks],
        "delivery_windows": [list(item) for item in delivery_chunks(chunks, plan)],
        "target_segment": options.target_segment,
        "source_revision": latest["pj_revision"], "consumed_source": new_hd["pj_refine_source"],
        "upscaler": {"node": "MiniMaxH3EasyLatentUpscaler3D", "model": options.upscaler_model,
                     "width": options.refine_width, "height": options.refine_height},
    }
    write_json(output / "manifest.json", manifest)
    for name, value in (("before_hd", old_hd), ("after_hd", new_hd),
                        ("before_upscaled", old_input), ("after_upscaled", new_input),
                        ("after_low", latest)):
        torch.save([part.cpu() for part in value["samples"].unbind()], output / f"{name}.pt")
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    logging.info("Decoding complete before/after high-resolution timelines")
    check_queue(options.server_url)
    before_frames = nodes.VAEDecode().decode(vae, old_hd)[0].cpu()
    check_queue(options.server_url)
    after_frames = nodes.VAEDecode().decode(vae, new_hd)[0].cpu()
    torch.save(before_frames, output / "before_hd_frames.pt")
    torch.save(after_frames, output / "after_hd_frames.pt")
    report = analyze_outputs(before, after, manifest, before_frames, after_frames)
    x0_report = analyze_outputs(list(old_x0["samples"].unbind()), list(new_x0["samples"].unbind()), manifest)
    report["hard_failures"].extend(f"Denoised: {failure}" for failure in x0_report["hard_failures"])
    if repeated["pj_revision"] != new_hd["pj_revision"]:
        report["hard_failures"].append("Repeated source update did not reuse the completed HDR revision")
    expected_stamp = {key: latest["pj_revision"][key] for key in ("lineage", "version")}
    if new_hd["pj_refine_source"] != expected_stamp:
        report["hard_failures"].append("HDR source stamp did not consume the latest first-pass revision")
    if options.denoise > 0 and not report.get("streams", {}).get("video", {}).get("changed_steps"):
        report["hard_failures"].append("The linked HDR update did not change the selected latent region")
    report.update(status="fail" if report["hard_failures"] else "pass", real_h3_sampling=True,
                  reference_inputs=reference_report(),
                  denoised_validation=x0_report, source_intervals=intervals,
                  repeat_reuses_completed_revision=repeated["pj_revision"] == new_hd["pj_revision"],
                  elapsed_seconds=time.perf_counter() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(output / "analysis.json", report)
    print(json.dumps({key: report[key] for key in ("status", "hard_failures", "source_intervals",
                      "repeat_reuses_completed_revision", "elapsed_seconds", "peak_cuda_allocated_bytes")}, ensure_ascii=False))
    return str(output / "analysis.json")


if __name__ == "__main__":
    options = arguments()
    plugin_root = Path(options.plugin_root).resolve()
    if not (plugin_root / "__init__.py").is_file():
        raise FileNotFoundError(plugin_root / "__init__.py")
    local_inputs = [options.baseline_file, options.prompt_file, options.reference_video,
                    options.mask_file, *options.reference_image]
    for filename in filter(None, local_inputs):
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    if options.prompt_file:
        options.prompt = Path(options.prompt_file).read_text(encoding="utf-8-sig")
    if options.reference_video_start < 0:
        raise ValueError("--reference-video-start must be nonnegative")
    if options.reference_video and shutil.which("ffmpeg") is None:
        raise FileNotFoundError("--reference-video requires the existing FFmpeg executable on PATH")
    if options.mask_file and (not options.baseline_file or options.relay_smoke):
        raise ValueError("--mask-file requires an imported --baseline-file and a local-repair/refinement run")
    if options.width % 32 or options.height % 32 or min(options.width, options.height) < 32:
        raise ValueError("Width and height must be positive multiples of 32")
    if options.frames < 5 or options.frames % 17 != 5:
        raise ValueError("Frames must follow the H3 17n+5 grid")
    if options.steps < 1 or not 0 <= options.denoise <= 1 or options.boundary_tokens < 0:
        raise ValueError("Invalid sampling steps, denoise or protected boundary size")
    if options.relay_smoke and options.refine_smoke:
        raise ValueError("Choose one smoke-test mode per run")
    if options.refine_smoke and (options.refine_width % 32 or options.refine_height % 32 or options.refine_steps < 1):
        raise ValueError("Refinement dimensions must be multiples of 32 and refinement steps positive")
    check_queue(options.server_url)
    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("before.pt", "after.pt", "analysis.json")):
        raise ValueError("Choose a fresh output directory; existing validation artifacts will not be overwritten")
    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import comfy.cli_args
    comfy.cli_args.args.cpu = options.executor_smoke_test
    import folder_paths
    import utils.extra_config
    config = COMFY_ROOT / "extra_model_paths.yaml"
    if config.is_file():
        utils.extra_config.load_extra_path_config(str(config))
    for config in options.extra_model_paths:
        utils.extra_config.load_extra_path_config(config)
    resolved = {}
    if not options.executor_smoke_test:
        for name, category in (("model", "diffusion_models"), ("clip", "text_encoders"), ("vae", "vae"), ("lora", "loras")):
            value = getattr(options, name)
            if value:
                resolved[name] = folder_paths.get_full_path_or_raise(category, value)
        if options.refine_smoke:
            resolved["upscaler_model"] = folder_paths.get_full_path_or_raise("latent_upscale_models", options.upscaler_model)
    write_json(output / "preflight.json", {"weights": resolved, "queue_empty_at_start": True,
                                           "reference_inputs": reference_report(),
                                           "baseline_file": options.baseline_file,
                                           "comfy_root": str(COMFY_ROOT), "plugin_root": str(plugin_root),
                                           "real_sampling_requested": not options.preflight and not options.executor_smoke_test})
    if options.preflight:
        print(json.dumps({"weights_found": list(resolved), "output_dir": str(output)}, ensure_ascii=False))
        sys.exit(0)

    # Bootstrap the existing execution environment only after the read-only preflight.
    import torch
    import comfy.model_management
    import comfy.nested_tensor
    import comfy.samplers
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, Noise_RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
    from analyze_segment_outputs import analyze_outputs, frame_metrics

    spec = importlib.util.spec_from_file_location("pj_h3_validation", plugin_root / "__init__.py", submodule_search_locations=[str(plugin_root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from pj_h3_validation.pj_first_pass_sampler import PJ_FirstPass_SegmentSampler
    from pj_h3_validation.pj_segment_sampler import PJ_SegmentSampler
    from pj_h3_validation.pj_h3_context import PJ_H3SegmentContext, prepare_segment_guider
    from pj_h3_validation.pj_h3_edit_prepare import PJ_H3EditPrepare
    from pj_h3_validation.segment_revision import changed_intervals
    from pj_h3_validation.segment_local_repair import repair_sigmas
    from pj_h3_validation.segment_timing import frames_for_tokens, resolve_plan, plan_relay_segments, delivery_chunks
    if options.refine_smoke:
        upscaler_path = ROOT.parent / "ComfyUI-MiniMaxH3-Easy" / "h3_latent_upscaler.py"
        if not upscaler_path.is_file():
            raise FileNotFoundError("--refine-smoke requires the installed MiniMaxH3EasyLatentUpscaler3D node")
        upscaler_spec = importlib.util.spec_from_file_location("pj_validation_public_upscaler", upscaler_path)
        upscaler_module = importlib.util.module_from_spec(upscaler_spec)
        sys.modules[upscaler_spec.name] = upscaler_module
        upscaler_spec.loader.exec_module(upscaler_module)
        MiniMaxH3EasyLatentUpscaler3D = upscaler_module.MiniMaxH3EasyLatentUpscaler3D

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    nodes.NODE_CLASS_MAPPINGS["PJValidationRun"] = ValidationRun
    executor = execution.PromptExecutor(ValidationServer(), cache_type=execution.CacheType.NONE,
                                        cache_args={"ram": 0, "ram_inactive": 0})
    executor.execute({"validation": {"class_type": "PJValidationRun", "inputs": {}}},
                     "pj-isolated-validation", extra_data={"preview_method": "none"}, execute_outputs=["validation"])
    if not executor.success:
        write_json(output / "execution_error.json", {"status_messages": executor.status_messages})
        sys.exit(1)
    report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    sys.exit(0 if report["status"] == "pass" else 1)
