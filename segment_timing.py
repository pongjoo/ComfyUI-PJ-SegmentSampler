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


def plan_segments(total_v, total_a, split_mode, chunk_seconds, overlap_seconds,
                  num_segments, custom_cuts, fps=FPS, align_51_frames=False):
    step = 15 if align_51_frames else 5
    frame_step = step * 17 / 5

    def window(frames):
        return max(1, math.floor((frames - 5) / frame_step)) * step + 2

    def interval(start, end):
        start = max(0, int(start // step) * step)
        end = min(total_v, max(start + 2, end))
        a_start = min(total_a, round(frames_for_tokens(start) * AUDIO_FPS / fps))
        a_end = total_a if end == total_v else min(total_a, round(frames_for_tokens(end) * AUDIO_FPS / fps))
        return start, end, a_start, a_end

    chunks = []
    if split_mode in ("按秒自动切分", "auto_by_seconds"):
        win = window(chunk_seconds * fps)
        stride = max(step, int((chunk_seconds - overlap_seconds) * fps // frame_step) * step)
        stride = min(stride, win - 2)
        for start in range(0, total_v, stride):
            chunks.append(interval(start, start + win))
            if chunks[-1][1] == total_v:
                break
    elif split_mode in ("固定分段数", "fixed_segments"):
        count = min(max(1, int(num_segments)), max(1, (total_v - 2) // step + 1))
        if count == 1:
            return [(0, total_v, 0, total_a)]
        overlap = max(2, overlap_seconds * fps * 5 / 17)
        win = min(total_v, math.ceil(((total_v + (count - 1) * overlap) / count - 2) / step) * step + 2)
        last_start = max(0, ((total_v - win) // step) * step)
        for i in range(count):
            start = (i * last_start // (count - 1) // step) * step
            chunks.append(interval(start, total_v if i == count - 1 else start + win))
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
            start = int(start_sec * fps // frame_step) * step
            end = max(0, math.ceil((end_sec * fps - 5) / frame_step)) * step + 2
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
