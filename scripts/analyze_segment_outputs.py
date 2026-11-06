"""Local output checks; latent intervals are tensor steps, timeline intervals are frames.

Example manifest:
{"baseline": {"seed": 42, "denoise": 0.4},
 "streams": {"video": {"time_dim": 2, "editable_intervals": [[2, 6]],
                       "protected_intervals": [[2, 3], [5, 6]]}},
 "timeline": {"total_frames": 22, "delivered_intervals": [[0, 10], [10, 22]]}}

Export AV samples as ordinary [video_tensor, audio_tensor], not NestedTensor.
Decoded tensor files must contain THWC pixels (uint8 or floats in [0, 1]).
python scripts/analyze_segment_outputs.py --before old.pt --after new.pt \
    --manifest checks.json --report report.json
"""

import argparse
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import torch


def load_tensors(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def streams(value):
    if isinstance(value, torch.Tensor):
        result = {"video": value}
    elif isinstance(value, dict) and "samples" in value:
        return streams(value["samples"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        result = dict(zip(("video", "audio"), value))
    elif isinstance(value, dict) and value:
        result = value
    else:
        raise ValueError("Use a tensor, [video, audio], or a dictionary of tensor streams.")
    if any(not isinstance(t, torch.Tensor) or t.is_nested or t.layout != torch.strided for t in result.values()):
        raise ValueError("Streams must be ordinary dense tensors; export nested AV streams separately.")
    return result


def intervals(value, length, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of [start, end) integer intervals.")
    previous = 0
    for pair in value:
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(type(x) is not int for x in pair)):
            raise ValueError(f"{name} requires integer [start, end) pairs.")
        start, end = pair
        if not 0 <= start < end <= length or start < previous:
            raise ValueError(f"{name} contains an out-of-range, overlapping, or unordered interval: {pair}.")
        previous = end
    return value


def timeline_check(value):
    if not isinstance(value, dict):
        raise ValueError("timeline must be an object with total_frames and delivered_intervals.")
    total = value.get("total_frames")
    if type(total) is not int or total <= 0:
        raise ValueError("timeline.total_frames must be a positive integer.")
    delivery = intervals(value.get("delivered_intervals"), total, "timeline.delivered_intervals")
    cursor = 0
    for start, end in delivery:
        if start != cursor:
            raise ValueError(f"Timeline has missing frames [{cursor}, {start}).")
        cursor = end
    if cursor != total:
        raise ValueError(f"Timeline has missing frames [{cursor}, {total}).")
    return {"total_frames": total, "delivered_intervals": delivery, "complete": True}


def compare_stream(before, after, config, edit_mask=None):
    if not isinstance(config, dict):
        raise ValueError("Each stream requires a configuration object.")
    info = {"before_shape": list(before.shape), "after_shape": list(after.shape),
            "before_dtype": str(before.dtype), "after_dtype": str(after.dtype)}
    failures = []
    if before.shape != after.shape:
        return info, ["shape changed"]
    if not before.is_floating_point() or not after.is_floating_point():
        raise ValueError("Latent streams must use real floating-point tensors.")
    if before.dtype != after.dtype:
        failures.append("dtype changed")
    axis = config.get("time_dim")
    if type(axis) is not int or not -before.ndim <= axis < before.ndim:
        raise ValueError("time_dim must identify the stream's temporal dimension.")
    axis %= before.ndim
    length = before.shape[axis]
    if before.numel() == 0:
        raise ValueError("Empty latent streams cannot establish a baseline.")
    editable = intervals(config.get("editable_intervals"), length, "editable_intervals")
    protected = intervals(config.get("protected_intervals", []), length, "protected_intervals")
    if edit_mask is not None:
        if not bool(torch.isfinite(edit_mask).all()) or bool(((edit_mask < 0) | (edit_mask > 1)).any()):
            raise ValueError("Edit masks must be finite values in [0, 1]; zero marks protected values.")
        edit_mask = torch.broadcast_to(edit_mask, before.shape)
    changed, forbidden = [], []
    maximum = outside_maximum = 0.0
    finite = True
    # Process one temporal slice at a time; do not allocate two full FP64 videos.
    for step in range(length):
        old = before.select(axis, step).to(device="cpu", dtype=torch.float64)
        new = after.select(axis, step).to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(old).all() and torch.isfinite(new).all()):
            finite = False
            continue
        differences = (new - old).abs()
        difference = float(differences.max())
        if not bool(torch.isfinite(differences).all()):
            finite = False
            continue
        allowed = any(a <= step < b for a, b in editable) and not any(a <= step < b for a, b in protected)
        maximum = max(maximum, difference)
        if difference:
            changed.append(step)
        outside = difference if not allowed else 0.0
        if allowed and edit_mask is not None:
            protected_pixels = edit_mask.select(axis, step).to("cpu") == 0
            if bool(protected_pixels.any()):
                outside = float(differences[protected_pixels].max())
        outside_maximum = max(outside_maximum, outside)
        if outside:
            forbidden.append(step)
    if not finite:
        failures.append("non-finite values in baseline, candidate, or their difference")
    if forbidden:
        failures.append("values changed outside the editable area or inside protected boundaries")
    info.update(time_dim=axis, finite=finite, editable_intervals=editable, protected_intervals=protected,
                changed_steps=changed, forbidden_changed_steps=forbidden, max_abs_diff=maximum,
                protected_max_abs_diff=outside_maximum, protected_exact=finite and not forbidden,
                spatial_mask_checked=edit_mask is not None)
    return info, failures


def read_frames(path, ffmpeg=None, ffprobe=None, analysis_width=256):
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() in (".pt", ".pth"):
        value = load_tensors(path)
        if isinstance(value, dict):
            value = value["frames"]
        return value, {"source": str(path), "format": "tensor", "resized": False}
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("Video analysis requires existing local ffmpeg and ffprobe; or supply a THWC .pt file.")
    probe = subprocess.run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "json", str(path)], capture_output=True, check=True)
    shape = json.loads(probe.stdout)["streams"][0]
    width = min(shape["width"], analysis_width)
    height = max(2, round(shape["height"] * width / shape["width"] / 2) * 2)
    decoded = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(path), "-map", "0:v:0",
                              "-vf", f"scale={width}:{height}", "-vsync", "0", "-pix_fmt", "rgb24",
                              "-f", "rawvideo", "pipe:1"], capture_output=True, check=True)
    size = width * height * 3
    if not decoded.stdout or len(decoded.stdout) % size:
        raise ValueError("Decoded byte count is not a whole number of RGB frames.")
    frames = torch.frombuffer(bytearray(decoded.stdout), dtype=torch.uint8).reshape(-1, height, width, 3)
    return frames, {"source": str(path), "format": "video", "original_size": [shape["width"], shape["height"]],
                    "analysis_size": [width, height], "resized": [width, height] != [shape["width"], shape["height"]]}


def frame_metrics(frames, expected_frames):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError("Decoded frames must be a THWC tensor with 1, 3, or 4 channels.")
    if frames.shape[0] != expected_frames or min(frames.shape[1:3]) < 2:
        raise ValueError("Decoded frame count differs from the timeline or image dimensions are invalid.")
    luminance, sharpness, adjacent, duplicates = [], [], [], []
    previous = None
    for index, source in enumerate(frames):
        frame = source[..., :3].to(device="cpu", dtype=torch.float32)
        if frames.dtype == torch.uint8:
            frame = frame / 255
        if not bool(torch.isfinite(frame).all()) or float(frame.min()) < 0 or float(frame.max()) > 1:
            raise ValueError("Decoded pixels must be finite uint8 values or floats in [0, 1].")
        gray = frame.mean(dim=-1)
        luminance.append(float(gray.mean()))
        sharpness.append(float(((gray[1:] - gray[:-1]).abs().mean() +
                                (gray[:, 1:] - gray[:, :-1]).abs().mean()) / 2))
        if previous is not None:
            difference = float((frame - previous).abs().mean())
            adjacent.append(difference)
            if difference == 0:
                if duplicates and duplicates[-1][1] == index:
                    duplicates[-1][1] = index + 1
                else:
                    duplicates.append([index - 1, index + 1])
        previous = frame
    peaks = sorted(enumerate(adjacent, start=1), key=lambda item: item[1], reverse=True)[:10]
    return {"shape": list(frames.shape), "frame_count": len(luminance), "mean_intensity": luminance,
            "edge_energy": sharpness, "adjacent_mean_abs_diff": adjacent, "identical_frame_runs": duplicates,
            "largest_frame_changes": [{"frame": index, "mean_abs_diff": value} for index, value in peaks if value > 0]}


def analyze_outputs(before, after, manifest, before_frames=None, after_frames=None, edit_masks=None):
    report = {"status": "pass", "baseline": {}, "hard_failures": [],
              "streams": {}, "quality_clues": {}, "unconfirmed": [
                  "Motion naturalness, semantic correctness, identity, and perceptual seam quality are not certified.",
                  "Output comparisons cannot prove which latent, seed, sigmas, or cache path the sampler used; execution tests are required."]}
    try:
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a JSON object.")
        json.dumps(manifest, allow_nan=False)
        report["baseline"] = manifest.get("baseline", {})
        report["timeline"] = timeline_check(manifest.get("timeline", {}))
        old, new = streams(before), streams(after)
        masks = streams(edit_masks) if edit_masks is not None else {}
        configs = manifest.get("streams", {})
        if not isinstance(configs, dict):
            raise ValueError("streams must map stream names to configuration objects.")
        if old.keys() != new.keys() or old.keys() != configs.keys():
            raise ValueError("Baseline, candidate, and manifest must name exactly the same streams.")
        if not masks.keys() <= old.keys():
            raise ValueError("Edit masks contain unknown streams.")
        for name in old:
            info, failures = compare_stream(old[name], new[name], configs[name], masks.get(name))
            report["streams"][name] = info
            report["hard_failures"].extend(f"{name}: {message}" for message in failures)
        for name, frames in (("before", before_frames), ("after", after_frames)):
            if frames is not None:
                report["quality_clues"][name] = frame_metrics(frames, report["timeline"]["total_frames"])
        quality = report["quality_clues"]
        if quality:
            quality["interpretation"] = ("Mean intensity, edge energy and adjacent differences are proxies, not perceptual scores. "
                                         "Identical frames can be normal stillness; only timeline metadata establishes duplicate delivery. "
                                         "Motion, cuts, occlusion, compression and resizing can change these measurements.")
        if "before" in quality and "after" in quality:
            if quality["before"]["shape"] == quality["after"]["shape"]:
                quality["intensity_delta"] = [b - a for a, b in zip(quality["before"]["mean_intensity"], quality["after"]["mean_intensity"])]
                quality["edge_energy_delta"] = [b - a for a, b in zip(quality["before"]["edge_energy"], quality["after"]["edge_energy"])]
                quality["paired_mean_abs_diff"] = []
                for old_frame, new_frame in zip(before_frames, after_frames):
                    old_frame = old_frame[..., :3].to(device="cpu", dtype=torch.float32) / (255 if before_frames.dtype == torch.uint8 else 1)
                    new_frame = new_frame[..., :3].to(device="cpu", dtype=torch.float32) / (255 if after_frames.dtype == torch.uint8 else 1)
                    quality["paired_mean_abs_diff"].append(float((new_frame - old_frame).abs().mean()))
            else:
                report["unconfirmed"].append("Decoded sizes differ; intensity/detail deltas were not compared. Use the same analysis scale.")
        if not quality:
            report["unconfirmed"].append("No decoded-frame measurements were supplied.")
    except (ValueError, TypeError, KeyError, RuntimeError) as error:
        report["hard_failures"].append(str(error))
    if report["hard_failures"]:
        report["status"] = "fail"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for argument in ("before", "after", "manifest"):
        parser.add_argument(f"--{argument}", required=True)
    parser.add_argument("--report", help="Write JSON here; also print it to stdout.")
    parser.add_argument("--before-frames", help="THWC .pt/.pth or a local video file.")
    parser.add_argument("--after-frames", help="THWC .pt/.pth or a local video file.")
    parser.add_argument("--edit-mask", help="Tensor file broadcastable to latent shape; zero values must remain unchanged.")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--video-analysis-width", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        if args.video_analysis_width < 2:
            raise ValueError("--video-analysis-width must be at least 2.")
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
        frames, sources = {}, {}
        for name, path in (("before", args.before_frames), ("after", args.after_frames)):
            if path:
                frames[name], sources[name] = read_frames(path, args.ffmpeg, args.ffprobe, args.video_analysis_width)
        report = analyze_outputs(load_tensors(args.before), load_tensors(args.after), manifest,
                                 frames.get("before"), frames.get("after"),
                                 load_tensors(args.edit_mask) if args.edit_mask else None)
        report["inputs"] = {"before": str(Path(args.before).resolve()), "after": str(Path(args.after).resolve()),
                            "manifest": str(Path(args.manifest).resolve()), "frames": sources}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, subprocess.CalledProcessError) as error:
        report = {"status": "fail", "hard_failures": [f"Input/read failure: {error}"],
                  "quality_clues": {}, "unconfirmed": ["Output validation did not complete."]}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        Path(args.report).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if report["hard_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
