"""Constrain segment intersections through the native latent and noise-mask inputs."""

import torch
import comfy.nested_tensor
import comfy.sampler_helpers


def constrain_boundaries(samples, mask, bounds, sources):
    """Sources are (absolute AV bounds, samples), in overlap ownership order."""
    intersections = [(other, value) for other, value in sources
                     if max(bounds[0], other[0]) < min(bounds[1], other[1])]
    if not intersections:
        return samples, mask

    nested = samples.is_nested
    streams = list(samples.unbind()) if nested else [samples]
    masks = list(mask.unbind()) if mask is not None and mask.is_nested else [mask]
    output, output_masks = [], []
    for stream_index, stream in enumerate(streams):
        axis = 2 if stream_index == 0 else stream.ndim - 1
        start, end = bounds[stream_index * 2:stream_index * 2 + 2]
        parts = [(other[stream_index * 2:stream_index * 2 + 2],
                  value.unbind()[stream_index] if nested else value) for other, value in intersections]
        dtype = stream.dtype
        for _, value in parts:
            dtype = torch.promote_types(dtype, value.dtype)
        target = stream.to(dtype=dtype, copy=True)
        existing = masks[stream_index] if stream_index < len(masks) else None
        if existing is None:
            shape = list(stream.shape)
            shape[1] = 1
            denoise = torch.ones(shape, device=stream.device, dtype=torch.float32)
        else:
            denoise = comfy.sampler_helpers.prepare_mask(existing.float(), stream.shape, stream.device).clone()
        # Reverse writes give the first source ownership, including triple overlaps.
        for (source_start, source_end), source in reversed(parts):
            left, right = max(start, source_start), min(end, source_end)
            if left < right:
                target.narrow(axis, left - start, right - left).copy_(
                    source.narrow(axis, left - source_start, right - left))
                denoise.narrow(axis, left - start, right - left).zero_()
        output.append(target)
        output_masks.append(denoise)
    if nested:
        return comfy.nested_tensor.NestedTensor(output), comfy.nested_tensor.NestedTensor(output_masks)
    return output[0], output_masks[0]


def restore_boundaries(samples, bounds, sources):
    """Remove solver/latent-transform rounding drift from the already-constrained overlaps."""
    streams = list(samples.unbind()) if samples.is_nested else [samples]
    for other, value in reversed(sources):
        parts = list(value.unbind()) if value.is_nested else [value]
        for i, (stream, source) in enumerate(zip(streams, parts)):
            start, end = bounds[i * 2:i * 2 + 2]
            source_start, source_end = other[i * 2:i * 2 + 2]
            left, right = max(start, source_start), min(end, source_end)
            if left < right:
                axis = 2 if i == 0 else stream.ndim - 1
                stream.narrow(axis, left - start, right - left).copy_(
                    source.narrow(axis, left - source_start, right - left))
