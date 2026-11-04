"""Local segment conditioning, disk cache and overlap assembly."""

import copy
import hashlib
import math
import os
import tempfile
from pathlib import Path

import torch
import comfy.nested_tensor
import comfy.sample

from .segment_timing import frames_for_tokens, tokens_for_frames


def fingerprint(value):
    digest = hashlib.sha256()

    def update(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
        elif isinstance(item, comfy.nested_tensor.NestedTensor):
            update(item.unbind())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                if key != "uuid":  # ComfyUI recreates condition UUIDs independently of their contents.
                    update(key)
                    update(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item:
                update(entry)
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(repr(item).encode())
        else:
            digest.update(str(id(item)).encode())
        digest.update(b"\x00")

    update(value)
    return digest.hexdigest()


class SegmentCache:
    """Each node owns one temporary directory and only small metadata in RAM."""

    def __init__(self):
        self.directory = None
        self.entries = {}

    def retain(self, count):
        for index in list(self.entries):
            if index >= count:
                self.entries.pop(index)
                (Path(self.directory.name) / f"{index}.pt").unlink(missing_ok=True)

    def load(self, index, signature):
        if self.entries.get(index) != signature:
            return None
        path = Path(self.directory.name) / f"{index}.pt"
        if not path.exists():
            self.entries.pop(index)
            return None
        saved = torch.load(path, map_location="cpu", weights_only=True)
        return tuple(comfy.nested_tensor.NestedTensor(item) if isinstance(item, list) else item for item in saved)

    def save(self, index, signature, samples, denoised=None):
        if self.directory is None:
            self.directory = tempfile.TemporaryDirectory(prefix="pj_segments_")
        path = Path(self.directory.name) / f"{index}.pt"
        temporary = path.with_suffix(".tmp")
        saved = [list(item.unbind()) if item is not None and item.is_nested else item for item in (samples, denoised)]
        torch.save(saved, temporary)
        os.replace(temporary, path)
        self.entries[index] = signature


def sampling_signature(guider, sampler, sigmas):
    patcher = guider.model_patcher
    return fingerprint((id(patcher), str(patcher.patches_uuid), guider.original_conds,
                        guider.model_options, getattr(guider, "cfg", None),
                        sampler.sampler_function, sampler.extra_options, sampler.inpaint_options, sigmas))


def generate_noise(noise, latent_image, seed):
    if noise is None:
        return comfy.sample.prepare_noise(latent_image["samples"], seed, latent_image.get("batch_index"))
    if getattr(noise, "seed", seed) != seed:
        noise = copy.copy(noise)
        noise.seed = seed
    return noise.generate_noise(latent_image)


def slice_mask(mask, bounds):
    if mask is None:
        return None
    vs, ve, a_s, ae = bounds
    if mask.is_nested:
        parts = mask.unbind()
        video = parts[0] if parts[0].shape[2] == 1 else parts[0][:, :, vs:ve]
        sliced = [video.clone()]
        if len(parts) > 1:
            audio = parts[1] if parts[1].shape[-1] == 1 else parts[1][..., a_s:ae]
            sliced.append(audio.clone())
        return comfy.nested_tensor.NestedTensor(sliced)
    if mask.ndim == 5:
        return (mask if mask.shape[2] == 1 else mask[:, :, vs:ve]).clone()
    if mask.ndim == 4:
        return (mask if mask.shape[1] == 1 else mask[:, vs:ve]).clone()
    return mask


def _slice_refs(refs, bounds):
    vs, ve, a_s, ae = bounds
    result = []
    for ref in refs:
        if ref["kind"] not in ("video", "video_audio"):
            result.append(ref)
            continue
        if vs >= ref["latent"].shape[2]:
            continue
        sliced = dict(ref)
        sliced["latent"] = ref["latent"][:, :, vs:ve].clone()
        sliced["latent_t"] = sliced["latent"].shape[2]
        if ref.get("audio_latent") is not None:
            audio = ref["audio_latent"][..., a_s:ae].clone()
            sliced["audio_latent"] = audio if audio.shape[-1] else None
            sliced["ref_audio_t"] = audio.shape[-1]
            if not audio.shape[-1]:
                sliced["kind"] = "video"
        result.append(sliced)
    return result


def _slice_keyframes(keyframes, frame_start, frame_end):
    result = []
    for keyframe in keyframes:
        origin = keyframe["resolved_frame_index"]
        video = keyframe.get("latent")
        audio = keyframe.get("audio_latent")
        video_frames = frames_for_tokens(video.shape[2]) if video is not None else 0
        audio_frames = audio.shape[-1] * 24 / 40 if audio is not None else 0
        if origin >= frame_end or origin + max(video_frames, audio_frames) <= frame_start:
            continue
        sliced = dict(keyframe)
        offset = max(0, frame_start - origin)
        # Keep the reference's token phase; a crossing clip may retain context before zero.
        video_start = int(offset // 17) * 5
        offset = frames_for_tokens(video_start) if video is not None else offset
        if video is not None:
            video_end = min(video.shape[2], tokens_for_frames(frame_end - origin))
            sliced["latent"] = video[:, :, video_start:video_end].clone() if video_start < video_end else None
        if audio is not None:
            audio_start = round(offset * 40 / 24)
            audio_end = min(audio.shape[-1], round((frame_end - origin) * 40 / 24))
            sliced["audio_latent"] = audio[..., audio_start:audio_end].clone() if audio_start < audio_end else None
        sliced["resolved_frame_index"] = origin + offset - frame_start
        if sliced.get("latent") is not None or sliced.get("audio_latent") is not None:
            result.append(sliced)
    return result


def segment_guider(guider, bounds, sync_refs):
    frame_start, frame_end = (frames_for_tokens(v) for v in bounds[:2])
    raw_conds = {}
    for name, conditions in guider.original_conds.items():
        raw_conds[name] = []
        for condition in conditions:
            extra = dict(condition)
            embedding = extra.pop("cross_attn", None)
            extra.pop("uuid", None)
            if sync_refs and "minimax_refs" in extra:
                extra["minimax_refs"] = _slice_refs(extra["minimax_refs"], bounds)
            if "minimax_keyframes" in extra:
                extra["minimax_keyframes"] = _slice_keyframes(extra["minimax_keyframes"], frame_start, frame_end)
            raw_conds[name].append([embedding, extra])
    clone = copy.copy(guider)
    clone.original_conds = {}
    clone.inner_set_conds(raw_conds)
    return clone


def blend_sequence(chunks, intervals, original, blend_mode, is_audio=False):
    """Keep the original extent; only overlapping blocks need weighted arithmetic."""
    axis = original.ndim - 1 if is_audio else 2
    original = original.cpu()
    if len(chunks) == 1 and intervals[0] == (0, original.shape[axis]):
        return chunks[0]
    dtype = original.dtype
    for chunk in chunks:
        dtype = torch.promote_types(dtype, chunk.dtype)
    result = original.to(dtype=dtype, copy=True)
    if blend_mode in ("硬切", "hard_cut", "hard", "保留边界"):
        items = list(zip(chunks, intervals))
        if blend_mode == "保留边界":
            items.reverse()
        for chunk, (start, end) in items:
            result.narrow(axis, start, end - start).copy_(chunk)
        return result

    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    # Log weights keep dense multi-way overlaps from underflowing to zero.
    weights = []
    for i, (start, end) in enumerate(intervals):
        w = torch.zeros(end - start, dtype=compute_dtype)
        for j, (other_start, other_end) in enumerate(intervals):
            left, right = max(start, other_start), min(end, other_end)
            if i == j or left >= right:
                continue
            fade = torch.arange(1, right - left + 1, dtype=compute_dtype) / (right - left + 1)
            if blend_mode in ("余弦平滑(推荐)", "cosine"):
                fade = torch.sin(fade * (math.pi / 2)).square()
            if j > i:
                fade = fade.flip(0)
            w[left - start:right - start] += fade.log()
        weights.append(w)

    edges = sorted({point for interval in intervals for point in interval})
    for left, right in zip(edges, edges[1:]):
        active = [i for i, (start, end) in enumerate(intervals) if start <= left and end >= right]
        if not active:
            continue
        if len(active) == 1:
            i = active[0]
            result.narrow(axis, left, right - left).copy_(chunks[i].narrow(axis, left - intervals[i][0], right - left))
            continue
        # Bound temporary FP32 storage even for long, high-resolution overlaps.
        for offset in range(left, right, 8):
            length = min(8, right - offset)
            shape = list(result.shape)
            shape[axis] = length
            accum = torch.zeros(shape, dtype=compute_dtype)
            weight_shape = [1] * result.ndim
            weight_shape[axis] = length
            total = torch.zeros(weight_shape, dtype=compute_dtype)
            block_weights = torch.stack([weights[i][offset - intervals[i][0]:offset - intervals[i][0] + length] for i in active])
            block_weights = (block_weights - block_weights.amax(dim=0)).exp()
            for row, i in enumerate(active):
                start = offset - intervals[i][0]
                w = block_weights[row].reshape(weight_shape)
                accum.addcmul_(chunks[i].narrow(axis, start, length).to(dtype=compute_dtype), w)
                total.add_(w)
            result.narrow(axis, offset, length).copy_(accum / total)
    return result
