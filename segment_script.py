"""User-authored segment scripts; execution metadata never enters text conditioning."""

import json
import math
import re

from .segment_timing import FPS, frames_for_tokens


def parse_segment_prompts(text):
    try:
        prompts = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"分段提示词 JSON 格式错误：第 {exc.lineno} 行，第 {exc.colno} 列") from exc
    if not isinstance(prompts, dict):
        raise ValueError('分段提示词需要 JSON 对象，例如 {"2":"人物转身"}')
    result = {}
    for key, entry in prompts.items():
        if not re.fullmatch(r"[0-9]+", key) or int(key) < 1:
            raise ValueError("分段提示词的键需要为从 1 开始的段号")
        if int(key) in result:
            raise ValueError(f"分段提示词包含重复段号：{key}")
        if not isinstance(entry, (str, dict)):
            raise ValueError(f"第 {key} 段需要提示词文本或分段设置对象")
        if isinstance(entry, dict):
            unknown = entry.keys() - {"prompt", "transition", "audio_transition", "dialogue"}
            if unknown:
                raise ValueError(f"第 {key} 段包含未知设置：{', '.join(sorted(unknown))}")
            if not isinstance(entry.get("prompt", ""), str):
                raise ValueError(f"第 {key} 段 prompt 需要为文本")
            if entry.get("transition", "continue") not in ("continue", "cut"):
                raise ValueError(f"第 {key} 段 transition 需要为 continue 或 cut")
            if entry.get("audio_transition", "continue") not in ("continue", "reset"):
                raise ValueError(f"第 {key} 段 audio_transition 需要为 continue 或 reset")
            if "dialogue" in entry:
                if not isinstance(entry["dialogue"], list):
                    raise ValueError(f"第 {key} 段 dialogue 需要为台词数组，空数组表示无对白")
                for line in entry["dialogue"]:
                    if (not isinstance(line, dict) or set(line) != {"speaker", "text", "start", "end"}
                            or any(not isinstance(line[field], str) for field in ("speaker", "text"))
                            or any(isinstance(line[field], bool) or not isinstance(line[field], (int, float))
                                   or not math.isfinite(line[field]) for field in ("start", "end"))):
                        raise ValueError(f"第 {key} 段台词需要 speaker、text 文本和 start、end 成片秒数")
        result[int(key)] = entry
    return result


def segment_settings(context, index):
    entry = (context or {}).get("segment_prompts", {}).get(index + 1, "")
    return {"prompt": entry} if isinstance(entry, str) else entry


def scene_cut_segments(context):
    return {index for index, entry in (context or {}).get("segment_prompts", {}).items()
            if isinstance(entry, dict) and entry.get("transition") == "cut" and index > 1}


def segment_prompt(context, index, bounds):
    entry = segment_settings(context, index)
    prompt = "\n".join(part for part in (context["prompt"], entry.get("prompt", "")) if part)
    if "dialogue" not in entry:
        return prompt
    if not entry["dialogue"]:
        instruction = "Only environmental sounds and non-vocal music; no speech, whispers, singing or vocalizations."
    else:
        origin = frames_for_tokens(bounds[0]) / FPS
        lines = [f'[{line["start"] - origin:.2f}s–{line["end"] - origin:.2f}s] '
                 f'{line["speaker"]} says <d>{line["text"]}</d>.' for line in entry["dialogue"]]
        instruction = "\n".join(lines) + "\nOnly these spoken words; environmental sounds without speech at other times."
    # Keep the native section order when the user supplies a complete H3 prompt.
    soundscape = re.search(r"(?m)^overall_soundscape\s*:", prompt)
    if soundscape:
        return prompt[:soundscape.start()].rstrip() + "\n" + instruction + "\n\n" + prompt[soundscape.start():]
    return prompt + ("\n" if prompt else "") + instruction
