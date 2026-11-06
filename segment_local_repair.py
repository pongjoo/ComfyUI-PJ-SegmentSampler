"""Resample selected windows from a completed result, preserving both time boundaries."""

import torch
import comfy.nested_tensor
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
import latent_preview

from .segment_boundaries import restore_preserved
from .segment_revision import merge_intervals
from .segment_sampling import generate_noise, segment_guider, slice_mask
from .pj_h3_context import prepare_segment_guider


def repair_sigmas(guider, scheduler, steps, denoise):
    """Use BasicScheduler's contract, independently of the original generation sigmas."""
    if denoise <= 0:
        return torch.FloatTensor([])
    total_steps = int(steps / denoise) if denoise < 1 else steps
    sigmas = comfy.samplers.calculate_sigmas(
        guider.model_patcher.get_model_object("model_sampling"), scheduler, total_steps).cpu()
    return sigmas[-(steps + 1):]


def _streams(samples):
    return list(samples.unbind()) if samples.is_nested else [samples]


def _join(parts, nested):
    return comfy.nested_tensor.NestedTensor(parts) if nested else parts[0]


def slice_latent(latent, bounds):
    parts = _streams(latent["samples"])
    output = dict(latent, samples=_join(
        [part.narrow(2 if i == 0 else part.ndim - 1, bounds[i * 2],
                     bounds[i * 2 + 1] - bounds[i * 2]).clone() for i, part in enumerate(parts)],
        latent["samples"].is_nested))
    if "noise_mask" in output:
        output["noise_mask"] = slice_mask(output["noise_mask"], bounds)
    output.pop("pj_revision", None)
    return output


def repair_segments(noise, guider, sampler, sigmas, baseline, chunks, target_indices,
                    seed, offsets=(), boundary_tokens=2, sync_ref_video=True,
                    keep_audio=True, initial_latent=None, editable_ranges=None,
                    denoised_baseline=None, h3_context=None, delivery_bounds=None):
    """Return full samples, full denoised result and editable video intervals.

    baseline owns retained output/boundaries; initial_latent optionally provides newer
    segment interiors (for high-resolution refinement). Inputs are never mutated.
    """
    source = baseline if initial_latent is None else initial_latent
    original = _streams(baseline["samples"])
    incoming = _streams(source["samples"])
    if len(original) != len(incoming) or any(a.shape != b.shape for a, b in zip(original, incoming)):
        raise ValueError("局部重做的原结果与新输入潜空间形状不一致，请先完成对应尺寸的完整采样")
    targets = sorted(set(int(index) for index in target_indices))
    if not targets or any(index < 0 or index >= len(chunks) for index in targets):
        raise ValueError("局部重做目标段不在实际规划范围内")
    if len(sigmas) < 2:
        return baseline, baseline if denoised_baseline is None else denoised_baseline, []
    nested = baseline["samples"].is_nested
    result = [part.to(device="cpu", copy=True) for part in original]
    denoised_original = _streams((baseline if denoised_baseline is None else denoised_baseline)["samples"])
    denoised = [part.to(device="cpu", copy=True) for part in denoised_original]
    total_v = result[0].shape[2]
    deliveries = chunks if delivery_bounds is None else delivery_bounds
    if len(deliveries) != len(chunks) or any(
            not chunk[0] <= delivered[0] <= delivered[1] <= chunk[1]
            for chunk, delivered in zip(chunks, deliveries)):
        raise ValueError("局部回修交付区间必须位于对应采样窗口内")
    selected = merge_intervals(deliveries[index][:2] for index in targets)
    editable = merge_intervals(editable_ranges) if editable_ranges is not None else []
    if editable_ranges is None:
        for start, end in selected:
            left = start + (boundary_tokens if start > 0 else 0)
            right = end - (boundary_tokens if end < total_v else 0)
            if left >= right:
                raise ValueError("所选分段没有可重绘区域；请减少边界保护潜步或扩大目标范围")
            editable.append((left, right))
    if any(not any(start <= a < b <= end for start, end in selected) for a, b in editable):
        raise ValueError("局部修改范围必须由所选采样窗口完整覆盖")
    if not editable:
        return baseline, baseline if denoised_baseline is None else denoised_baseline, []
    written = []
    for index in targets:
        bounds = chunks[index]
        start, end = bounds[:2]
        delivery_start, delivery_end = deliveries[index][:2]
        chunk = slice_latent(source, bounds)
        parts = _streams(chunk["samples"])
        existing = chunk.get("noise_mask")
        masks = _streams(existing) if existing is not None and existing.is_nested else [existing]
        video = parts[0]
        mask = (comfy.sampler_helpers.prepare_mask(masks[0].float(), video.shape, video.device).clone()
                if masks[0] is not None else torch.ones(
                    (video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]),
                    device=video.device, dtype=torch.float32))
        time_edit = torch.zeros((1, 1, end - start, 1, 1), device=video.device, dtype=torch.float32)
        for left, right in editable:
            a, b = max(delivery_start, left), min(delivery_end, right)
            if a < b:
                time_edit[:, :, a - start:b - start] = 1
        # Earlier selected windows own intersections, so every time position is replaced once.
        for left, right in written:
            a, b = max(start, left), min(end, right)
            if a < b:
                time_edit[:, :, a - start:b - start] = 0
        mask.mul_(time_edit)
        dtype = torch.promote_types(video.dtype, result[0].dtype)
        video = video.to(dtype=dtype)
        retained = result[0][:, :, start:end].to(device=video.device, dtype=dtype)
        parts[0] = torch.where(mask > 0, video, retained)
        denoise_masks = [mask]
        if len(parts) > 1:
            if not keep_audio:
                raise ValueError("局部回修当前仅修改视频，请保留原音频")
            parts[1] = original[1][..., bounds[2]:bounds[3]].to(device=parts[1].device, copy=True)
            denoise_masks.append(torch.zeros((parts[1].shape[0], 1, 1, parts[1].shape[-1]),
                                             device=parts[1].device, dtype=torch.float32))
        sample_input = _join(parts, nested)
        denoise_mask = _join(denoise_masks, nested)
        segment_seed = (int(seed) + (int(offsets[index]) if index < len(offsets) else index)) & 0xFFFFFFFFFFFFFFFF
        noise_full = generate_noise(noise, source, segment_seed)
        local_noise = slice_latent({"samples": noise_full}, bounds)["samples"]
        del noise_full
        local_guider = (segment_guider(guider, bounds, sync_ref_video) if h3_context is None else
                        prepare_segment_guider(guider, h3_context, bounds, parts[0].shape, segment_index=index))
        x0_output = {}
        callback = latent_preview.prepare_callback(local_guider.model_patcher, len(sigmas) - 1, x0_output)
        sampled = local_guider.sample(local_noise, sample_input, sampler, sigmas,
                                      denoise_mask=denoise_mask, callback=callback,
                                      disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=segment_seed).cpu()
        restore_preserved(sampled, sample_input, denoise_mask)
        sampled_video = _streams(sampled)[0]
        if "x0" in x0_output:
            x0 = x0_output["x0"]
            if nested and not x0.is_nested:
                x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, [part.shape for part in parts]))
            x0 = local_guider.model_patcher.model.process_latent_out(x0.cpu())
            restore_preserved(x0, sample_input, denoise_mask)
            x0_video = _streams(x0)[0]
        else:
            x0_video = sampled_video
        if not torch.isfinite(sampled_video).all().item() or not torch.isfinite(x0_video).all().item():
            raise ValueError("局部采样产生非有限值，已保留上一次完整结果；请检查采样与模型设置")
        # Preserve promoted sampler precision without rounding retained values down.
        dtype = torch.promote_types(result[0].dtype, sampled_video.dtype)
        result[0] = result[0].to(dtype=dtype)
        result[0][:, :, start:end].copy_(sampled_video.to(dtype=dtype))
        dtype = torch.promote_types(denoised[0].dtype, x0_video.dtype)
        denoised[0] = denoised[0].to(dtype=dtype)
        region = denoised[0][:, :, start:end]
        region.copy_(torch.where(mask.cpu() > 0, x0_video.to(dtype=dtype), region))
        written.append((delivery_start, delivery_end))
    output = dict(source, samples=_join(result, nested))
    return output, dict(source, samples=_join(denoised, nested)), merge_intervals(editable)
