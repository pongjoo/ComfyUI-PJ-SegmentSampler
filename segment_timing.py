"""H3 frame/latent boundaries shared by the planner and both samplers."""

import math

FPS = 24.0
AUDIO_FPS = 40.0


def frames_for_tokens(tokens):
    groups, tail = divmod(tokens, 5)
    return groups * 17 + (0, 1, 5, 9, 13)[tail]


def tokens_for_frames(frames):
    return max(0, math.ceil((frames - 5) / 17)) * 5 + 2


def tokens_for_frames_floor(frames):
    return max(0, math.floor((frames - 5) / 17)) * 5 + 2


def segment_seconds(start, end, fps=FPS):
    return frames_for_tokens(start) / fps, frames_for_tokens(end) / fps


def zero_overlap(plan):
    return (plan is not None and plan.get("split_mode", "按秒自动切分") in
            ("按秒自动切分", "auto_by_seconds", "固定分段数", "fixed_segments")
            and plan.get("overlap_seconds") == 0)


def delivery_chunks(chunks, plan):
    """Legal sampling windows may include right padding that is not delivered."""
    if not zero_overlap(plan):
        return list(chunks)
    delivered = []
    for index, (start, end, audio_start, audio_end) in enumerate(chunks):
        if index + 1 < len(chunks):
            end = min(end, chunks[index + 1][0])
            audio_end = min(audio_end, chunks[index + 1][2])
        delivered.append((start, end, audio_start, audio_end))
    return delivered


def plan_relay_segments(total_v, total_a, chunk_seconds, *, complete_timeline=False, cut_segments=(),
                        split_mode="按秒自动切分", num_segments=3, custom_cuts=""):
    if complete_timeline and split_mode in ("固定分段数", "fixed_segments", "自定义切点", "custom", "custom_cuts"):
        return plan_segments(total_v, total_a, split_mode, chunk_seconds, 22 / FPS, num_segments, custom_cuts)
    total_frames = frames_for_tokens(total_v)
    budgets = None
    if split_mode in ("固定分段数", "fixed_segments"):
        count = max(1, int(num_segments))
        budgets = [total_frames / count] * count
    elif split_mode in ("自定义切点", "custom", "custom_cuts"):
        ranges = parse_custom_ranges(custom_cuts)
        cursor = 0.0
        for index, (start, end) in enumerate(ranges, 1):
            if not math.isclose(start, cursor, abs_tol=1e-6, rel_tol=0):
                raise ValueError(f"接力自定义第{index}段应从{cursor:g}秒开始，实际为{start:g}秒；区间必须从0连续排列，不能遗漏、重叠或倒序")
            cursor = end
        if tokens_for_frames_floor(cursor * FPS) != total_v:
            raise ValueError(f"接力自定义末端{cursor:g}秒未覆盖完整输入预算（H3对齐后{total_frames}帧/{total_frames / FPS:.3f}秒）；请补齐或调整最后一段")
        budgets = [(end - start) * FPS for start, end in ranges]
    elif split_mode not in ("按秒自动切分", "auto_by_seconds"):
        raise ValueError(f"未知接力规划模式: {split_mode}")

    if budgets is not None:
        chunks = []
        start = 0
        for index, budget in enumerate(budgets):
            length = max(0, math.floor((budget - 5) / 17 + 0.5)) * 5 + 2
            end = min(total_v, start + length)
            inherited = 2 if index + 1 in cut_segments else 7
            if end <= start or (index and end - start <= inherited):
                raise ValueError(f"接力第{index + 1}段对齐后没有新增画面；请减少段数或合并过短区间")
            audio_start = min(total_a, round(frames_for_tokens(start) * AUDIO_FPS / FPS))
            audio_end = total_a if end == total_v else min(total_a, round(frames_for_tokens(end) * AUDIO_FPS / FPS))
            chunks.append((start, end, audio_start, audio_end))
            if index + 1 < len(budgets):
                overlap = 2 if index + 2 in cut_segments else 7
                if end - start <= overlap:
                    raise ValueError(f"接力第{index + 1}段不足以提供下一段所需的边界；请减少段数或合并过短区间")
                start = end - overlap
        return chunks

    # A window needs at least one new five-token group beyond its seven-token prefix.
    groups = max(2, math.floor((chunk_seconds * FPS - 5) / 17 + 0.5))
    window = groups * 5 + 2
    budget_frames = max(chunk_seconds * FPS, frames_for_tokens(12))
    chunks = []
    start = 0
    while start < total_v:
        length = window
        if not complete_timeline:
            # Budget before overlap removal; do not add windows to replace that loss.
            remaining_frames = total_frames - len(chunks) * budget_frames
            length = min(window, max(0, math.floor((remaining_frames - 5) / 17 + 0.5)) * 5 + 2)
            if chunks and length <= (2 if len(chunks) + 1 in cut_segments else 7):
                break
        end = min(total_v, start + length)
        audio_start = min(total_a, round(frames_for_tokens(start) * AUDIO_FPS / FPS))
        audio_end = total_a if end == total_v else min(total_a, round(frames_for_tokens(end) * AUDIO_FPS / FPS))
        chunks.append((start, end, audio_start, audio_end))
        if end == total_v:
            break
        start += window - (2 if len(chunks) + 1 in cut_segments else 7)
    return chunks


def parse_custom_ranges(custom_cuts):
    ranges = []
    for pair in custom_cuts.split(","):
        try:
            start, end = map(float, pair.strip().split("-"))
        except ValueError as exc:
            raise ValueError(f"无效自定义切点 {pair!r}，请使用 0-6,6-12 格式") from exc
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"无效自定义区间 {pair!r}：需要有限的非负起点且结束晚于开始")
        ranges.append((start, end))
    return ranges


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
        for start_sec, end_sec in parse_custom_ranges(custom_cuts):
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
