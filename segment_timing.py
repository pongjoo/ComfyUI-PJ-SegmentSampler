"""H3 frame/latent boundaries shared by the planner and both samplers."""

import math

FPS = 24.0
AUDIO_FPS = 40.0


def frames_for_tokens(tokens):
    groups, tail = divmod(tokens, 5)
    return groups * 17 + (0, 1, 5, 9, 13)[tail]


def tokens_for_frames(frames):
    return max(0, math.ceil((frames - 5) / 17)) * 5 + 2


def segment_seconds(start, end, fps=FPS):
    return frames_for_tokens(start) / fps, frames_for_tokens(end) / fps


def plan_relay_segments(total_v, total_a, chunk_seconds):
    # A window needs at least one new five-token group beyond its seven-token prefix.
    groups = max(2, math.floor((chunk_seconds * FPS - 5) / 17 + 0.5))
    window = groups * 5 + 2
    chunks = []
    for start in range(0, total_v, window - 7):
        end = min(total_v, start + window)
        audio_start = min(total_a, round(frames_for_tokens(start) * AUDIO_FPS / FPS))
        audio_end = total_a if end == total_v else min(total_a, round(frames_for_tokens(end) * AUDIO_FPS / FPS))
        chunks.append((start, end, audio_start, audio_end))
        if end == total_v:
            break
    return chunks


def plan_segments(total_v, total_a, split_mode, chunk_seconds, overlap_seconds,
                  num_segments, custom_cuts, fps=FPS, align_51_frames=False):
    # align_51_frames is accepted only for saved API workflows; all plans use the native grid.
    step = 5
    frame_step = 17
    total_groups = max(0, (total_v - 2) // step)

    def nearest_groups(frames):
        # Ties favour the longer duration/overlap instead of Python's even rounding.
        return max(0, math.floor((frames - 5) / frame_step + 0.5))

    def interval(start, end):
        start = max(0, int(start // step) * step)
        end = min(total_v, max(start + 2, end))
        a_start = min(total_a, round(frames_for_tokens(start) * AUDIO_FPS / fps))
        a_end = total_a if end == total_v else min(total_a, round(frames_for_tokens(end) * AUDIO_FPS / fps))
        return start, end, a_start, a_end

    chunks = []
    if split_mode in ("按秒自动切分", "auto_by_seconds"):
        groups = min(total_groups, max(1, nearest_groups(chunk_seconds * fps)))
        win = min(total_v, groups * step + 2)
        overlap_groups = min(max(0, groups - 1), nearest_groups(overlap_seconds * fps))
        stride = max(step, (groups - overlap_groups) * step)
        for start in range(0, total_v, stride):
            chunks.append(interval(start, start + win))
            if chunks[-1][1] == total_v:
                break
    elif split_mode in ("固定分段数", "fixed_segments"):
        count = min(max(1, int(num_segments)), total_groups + 1)
        if count == 1:
            return [(0, total_v, 0, total_a)]
        starts = [i * (total_groups + 1) // count * step for i in range(count)]
        overlap = nearest_groups(overlap_seconds * fps) * step + 2
        for i, start in enumerate(starts):
            end = starts[i + 1] + overlap if i + 1 < count else total_v
            chunks.append(interval(start, end))
    elif split_mode in ("自定义切点", "custom", "custom_cuts"):
        duration = frames_for_tokens(total_v) / fps
        for pair in custom_cuts.split(","):
            try:
                start_sec, end_sec = map(float, pair.strip().split("-"))
            except ValueError as exc:
                raise ValueError(f"无效自定义切点 {pair!r}，请使用 0-6, 4.5-10.5 格式") from exc
            if not (math.isfinite(start_sec) and math.isfinite(end_sec) and 0 <= start_sec < end_sec):
                raise ValueError(f"无效自定义区间 {pair!r}：需要有限的非负起点且结束晚于开始")
            if start_sec >= duration:
                continue
            start = min(total_groups, math.floor(start_sec * fps / frame_step + 0.5)) * step
            end = nearest_groups(end_sec * fps) * step + 2
            chunks.append(interval(start, end))
        if not chunks:
            raise ValueError("自定义切点没有覆盖输入视频")
    else:
        raise ValueError(f"未知分段模式: {split_mode}")
    return sorted(set(chunks))


def resolve_plan(samples, segment_plan, split_mode, chunk_seconds, overlap_seconds,
                 num_segments, custom_cuts, fps, align_51_frames, full_length=False):
    streams = list(samples.unbind()) if samples.is_nested else [samples]
    video = streams[0]
    total_a = streams[1].shape[-1] if len(streams) > 1 else 0
    rules = dict(split_mode=split_mode, chunk_seconds=chunk_seconds,
                 overlap_seconds=overlap_seconds, num_segments=num_segments,
                 custom_cuts=custom_cuts, fps=fps, align_51_frames=align_51_frames)
    if segment_plan is not None:
        rules.update({key: segment_plan[key] for key in rules if key in segment_plan})
    # H3's model clock is fixed; source/playback fps must not alter AV conditioning.
    if samples.is_nested and video.ndim == 5 and video.shape[1] == 24:
        rules["fps"] = FPS
    total_v = video.shape[2]
    chunks = [(0, total_v, 0, total_a)] if full_length else plan_segments(total_v, total_a, **rules)
    return chunks, rules["fps"]
