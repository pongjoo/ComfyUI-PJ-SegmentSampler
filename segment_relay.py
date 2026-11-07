"""Sequential H3 continuation with window noise and dependency-aware disk cache."""

from contextlib import ExitStack

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import latent_preview

from .segment_boundaries import constrain_boundaries, restore_preserved
from .segment_sampling import fingerprint, generate_noise, sampling_signature, segment_guider, slice_mask
from .pj_h3_context import prepare_segment_conditions


def sample_relay(noise, guider, sampler, sigmas, latent_image, chunks, cache,
                 seed, offsets, start_segment, sync_ref_video, h3_context=None):
    video, audio = latent_image["samples"].unbind()
    context = sampling_signature(guider, sampler, sigmas)
    if h3_context is not None:
        context = fingerprint((context, h3_context))
    noise_options = {k: v for k, v in vars(noise).items() if k != "seed"} if noise is not None else None
    cache.retain(len(chunks))
    previous = None
    previous_digest = None
    first = None
    assembled = None
    written = [0, 0]
    rebuilding = False

    with ExitStack() as condition_scope:
        for index, bounds in enumerate(chunks):
            vs, ve, a_start, a_end = bounds
            samples = comfy.nested_tensor.NestedTensor((
                video[:, :, vs:ve].to(device="cpu", copy=True),
                audio[..., a_start:a_end].to(device="cpu", copy=True),
            ))
            mask = slice_mask(latent_image.get("noise_mask"), bounds)
            offset = offsets[index] if index < len(offsets) else index
            segment_seed = (seed + offset) & 0xFFFFFFFFFFFFFFFF
            signature = fingerprint(("relay-22-v1", context, bounds, samples, mask,
                                     latent_image.get("batch_index"), sync_ref_video,
                                     type(noise), noise_options, segment_seed, previous_digest))
            cached = cache.load(index, signature) if not rebuilding and index + 1 < start_segment else None
            if cached is not None:
                sampled = cached[0]
                print(f"[PJ接力] 第 {index + 1}/{len(chunks)} 段复用缓存")
            else:
                if not rebuilding:
                    # Remove the entire dependent suffix before sampling, including on interruption.
                    cache.retain(index)
                    rebuilding = True
                    prepared = condition_scope.enter_context(prepare_segment_conditions(
                        h3_context, chunks, video.shape, range(index, len(chunks))))
                if previous is not None:
                    samples, mask = constrain_boundaries(samples, mask, bounds, [(chunks[index - 1], previous)])
                window_latent = dict(latent_image, samples=samples)
                if mask is not None:
                    window_latent["noise_mask"] = mask
                chunk_noise = generate_noise(noise, window_latent, segment_seed)
                if h3_context is None:
                    chunk_guider = segment_guider(guider, bounds, sync_ref_video)
                else:
                    previous_tail = None
                    if previous is not None:
                        overlap = min(ve, chunks[index - 1][1]) - vs
                        offset = vs - chunks[index - 1][0]
                        previous_tail = previous.unbind()[0][:, :, offset:offset + overlap]
                    chunk_guider = prepared.guider(guider, index, previous_tail)
                callback = latent_preview.prepare_callback(chunk_guider.model_patcher, sigmas.shape[-1] - 1)
                print(f"[PJ接力] 第 {index + 1}/{len(chunks)} 段采样，视频潜步 [{vs}:{ve}]，种子 {segment_seed}")
                sampled = chunk_guider.sample(
                    chunk_noise, samples, sampler, sigmas, denoise_mask=mask, callback=callback,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=segment_seed,
                ).to("cpu")
                restore_preserved(sampled, samples, mask)
                cache.save(index, signature, sampled)
                del chunk_noise, chunk_guider, window_latent, callback
                comfy.model_management.soft_empty_cache()

            parts = sampled.unbind()
            if first is None:
                first = sampled
                assembled = [torch.empty_like(source, device="cpu", dtype=torch.promote_types(source.dtype, part.dtype))
                             for source, part in zip((video, audio), parts)]
            for stream_index, (part, axis) in enumerate(zip(parts, (2, 3))):
                start, end = bounds[stream_index * 2:stream_index * 2 + 2]
                dtype = torch.promote_types(assembled[stream_index].dtype, part.dtype)
                if assembled[stream_index].dtype != dtype:
                    assembled[stream_index] = assembled[stream_index].to(dtype=dtype)
                # The preceding segment owns the overlap; copy only the new suffix.
                length = end - written[stream_index]
                assembled[stream_index].narrow(axis, written[stream_index], length).copy_(
                    part.narrow(axis, written[stream_index] - start, length))
                written[stream_index] = end
            previous = sampled
            previous_digest = fingerprint(sampled)

    output = dict(latent_image, samples=comfy.nested_tensor.NestedTensor(assembled))
    return output, {"samples": first}
