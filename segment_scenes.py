"""Scene boundaries on the shared H3 timeline, including nondelivered padding."""

import bisect

import comfy.nested_tensor

from .segment_timing import FPS, AUDIO_FPS, frames_for_tokens


def same_scene(left, right, cuts):
    return bisect.bisect_right(cuts, left) == bisect.bisect_right(cuts, right)


def scene_deliveries(chunks, deliveries, cuts):
    result = []
    for bounds, delivered in zip(chunks, deliveries):
        start, end, audio_start, audio_end = delivered
        following = next((cut for cut in cuts if bounds[0] < cut < bounds[1]), None)
        if following is not None:
            end = min(end, following)
            audio_end = min(audio_end, round(frames_for_tokens(following) * AUDIO_FPS / FPS))
        result.append((start, end, audio_start, audio_end))
    return result


def split_scene_windows(chunks, cuts, total_v, total_a):
    if not cuts:
        return list(chunks)
    edges = [0, *cuts, total_v]
    result = set()
    for start, end, _, _ in chunks:
        for scene_start, scene_end in zip(edges, edges[1:]):
            left, right = max(start, scene_start), min(end, scene_end)
            if left >= right:
                continue
            padded_end = min(total_v, right if right % 5 == 2 else right + 2)
            audio_start = round(frames_for_tokens(left) * AUDIO_FPS / FPS)
            audio_end = total_a if padded_end == total_v else round(frames_for_tokens(padded_end) * AUDIO_FPS / FPS)
            result.add((left, padded_end, min(total_a, audio_start), min(total_a, audio_end)))
    # An overlap can create a short piece already covered by a later window.
    windows = []
    for bounds in sorted(result, key=lambda item: (item[0], -item[1])):
        if windows and same_scene(windows[-1][0], bounds[0], cuts) and bounds[1] <= windows[-1][1]:
            continue
        windows.append(bounds)
    return windows


def clip_scene_windows(chunks, cuts, total_a):
    """Keep author-assigned segment numbers while trimming windows before a cut."""
    result = []
    for start, end, audio_start, audio_end in chunks:
        cut = next((cut for cut in cuts if start < cut < end), None)
        if cut is not None:
            end = min(end, cut + 2)
            audio_end = min(total_a, round(frames_for_tokens(end) * AUDIO_FPS / FPS))
        result.append((start, end, audio_start, audio_end))
    return result


def pad_scene_video(samples, bounds, cuts):
    end = next((cut for cut in cuts if bounds[0] < cut < bounds[1]), None)
    if end is None:
        return samples
    nested = samples.is_nested
    parts = list(samples.unbind()) if nested else [samples]
    video = parts[0].clone()
    valid = end - bounds[0]
    video[:, :, valid:] = video[:, :, valid - 1:valid]
    parts[0] = video
    return comfy.nested_tensor.NestedTensor(parts) if nested else video
