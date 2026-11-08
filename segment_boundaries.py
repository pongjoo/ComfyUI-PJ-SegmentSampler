"""Constrain segment intersections through the native latent and noise-mask inputs."""

import torch
import comfy.nested_tensor
import comfy.sampler_helpers


def constrain_boundaries(samples, mask, bounds, sources, preserve_streams=None):
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
        if preserve_streams is not None and stream_index not in preserve_streams:
            parts = []
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
        editable = denoise > 0 if existing is not None else None
        # Reverse writes give the first source ownership, including triple overlaps.
        for (source_start, source_end), source in reversed(parts):
            left, right = max(start, source_start), min(end, source_end)
            if left < right:
                region = target.narrow(axis, left - start, right - left)
                carried = source.narrow(axis, left - source_start, right - left).to(device=target.device, dtype=target.dtype)
                if editable is not None:
                    carried = torch.where(editable.narrow(axis, left - start, right - left), carried, region)
                region.copy_(carried)
                denoise.narrow(axis, left - start, right - left).zero_()
        output.append(target)
        output_masks.append(denoise)
    if nested:
        return comfy.nested_tensor.NestedTensor(output), comfy.nested_tensor.NestedTensor(output_masks)
    return output[0], output_masks[0]


def restore_preserved(samples, original, mask):
    """Restore zero-mask values after solver/assembly rounding, in bounded blocks."""
    if mask is None:
        return
    streams = list(samples.unbind()) if samples.is_nested else [samples]
    originals = list(original.unbind()) if original.is_nested else [original]
    masks = list(mask.unbind()) if mask.is_nested else [mask]
    for i, (stream, source, part) in enumerate(zip(streams, originals, masks)):
        prepared = comfy.sampler_helpers.prepare_mask(part.float(), stream.shape, stream.device)
        axis = 2 if i == 0 else stream.ndim - 1
        for start in range(0, stream.shape[axis], 8):
            length = min(8, stream.shape[axis] - start)
            region = stream.narrow(axis, start, length)
            preserved = source.narrow(axis, start, length).to(device=stream.device, dtype=stream.dtype)
            region.copy_(torch.where(prepared.narrow(axis, start, length) == 0, preserved, region))
