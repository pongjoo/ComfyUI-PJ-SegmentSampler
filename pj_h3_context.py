"""Rebuild native H3 conditions for one timeline window without decoding its result."""

import json

from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

from .segment_sampling import segment_guider
from .segment_timing import frames_for_tokens


class PJ_H3SegmentContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "segment_prompts": ("STRING", {"multiline": True, "default": "{}", "tooltip": '各段补充提示词，JSON 格式，例如 {"2":"人物转身"}；段号从 1 开始'}),
            "reference_mode": (["原视频编辑（按时间裁剪）", "素材参考（保持完整）"],),
            "motion_context": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "vae": ("VAE", {"tooltip": "连接视频 VAE 时，参考素材也编码为 H3 潜空间条件；不接时只用于文本编码"}),
            "reference_images": ("IMAGE", {"tooltip": "每张图作为一个独立图片参考，对应 <Picture 1>、<Picture 2> 等"}),
            "reference_video": ("IMAGE", {"tooltip": "24 FPS 参考视频，对应 <Video 1>；编辑模式按目标时间切片"}),
            "reference_video_audio": ("AUDIO", {"tooltip": "参考视频的配套音轨，与视频同步切片"}),
            "reference_audio": ("AUDIO", {"tooltip": "独立音频素材，不按视频时间裁剪"}),
            "audio_vae": ("VAE",),
        }}

    RETURN_TYPES = ("PJ_H3_CONTEXT",)
    RETURN_NAMES = ("H3分段上下文",)
    FUNCTION = "prepare"
    CATEGORY = "PJ工具箱/分段采样"

    def prepare(self, clip, vae=None, prompt="", segment_prompts="{}",
                reference_mode="原视频编辑（按时间裁剪）", motion_context=True,
                reference_images=None, reference_video=None, reference_video_audio=None,
                reference_audio=None, audio_vae=None):
        try:
            prompts = json.loads(segment_prompts or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("分段补充提示词需要 JSON 对象，例如 {\"2\":\"人物转身\"}") from exc
        if not isinstance(prompts, dict) or any(
                not key.isdecimal() or int(key) < 1 or not isinstance(value, str)
                for key, value in prompts.items()):
            raise ValueError("分段补充提示词的键为从 1 开始的段号，值为提示词文本")
        if reference_video_audio is not None and reference_video is None:
            raise ValueError("配套视频音轨需要连接参考视频；独立音频请连接 reference_audio")
        return ({
            "clip": clip, "vae": vae, "prompt": prompt,
            "segment_prompts": {int(key): value for key, value in prompts.items()},
            "sync_video": reference_mode == "原视频编辑（按时间裁剪）",
            "motion_context": motion_context, "reference_images": reference_images,
            "reference_video": reference_video, "reference_video_audio": reference_video_audio,
            "reference_audio": reference_audio, "audio_vae": audio_vae,
        },)


def _audio_window(audio, frame_start, frame_end):
    if audio is None:
        return None
    rate = audio["sample_rate"]
    start, end = round(frame_start * rate / 24), round(frame_end * rate / 24)
    return dict(audio, waveform=audio["waveform"][..., start:end].clone())


def motion_keyframes(previous_tail, video_shape):
    """The caller supplies an overlapping, phase-zero H3 tail (2 + 5k tokens)."""
    if previous_tail.ndim != 5 or previous_tail.shape[1] != 24:
        raise ValueError("运动上下文需要原生 H3 视频潜空间")
    if previous_tail.shape[2] % 5 != 2 or previous_tail.shape[2] > video_shape[2]:
        raise ValueError("运动上下文必须为相位对齐的 5、22、39… 帧，且不长于当前采样窗口")
    if (previous_tail.shape[:2], previous_tail.shape[-2:]) != (video_shape[:2], video_shape[-2:]):
        raise ValueError("运动上下文与当前采样窗口的批次、通道和分辨率必须一致")
    return [{"resolved_frame_index": frames_for_tokens(index),
             "latent": previous_tail[:, :, index:index + 1].to(device="cpu", copy=True)}
            for index in range(previous_tail.shape[2])]


def prepare_segment_guider(base_guider, context, bounds, video_shape,
                           previous_tail=None, segment_index=0):
    """Keep the guider policy/negative branch; rebuild positive semantics from raw inputs."""
    guider = segment_guider(base_guider, bounds, context["sync_video"])
    if len(video_shape) != 5 or video_shape[1] != 24 or video_shape[2] != bounds[1] - bounds[0]:
        raise ValueError("H3 分段条件需要当前窗口的原生视频潜空间形状")
    frame_start = frames_for_tokens(bounds[0])
    frame_count = frames_for_tokens(video_shape[2])
    if bounds[0] % 5 or video_shape[2] % 5 != 2:
        raise ValueError("H3 分段条件需要按 17n+5 帧网格对齐的窗口")
    prompt = context["prompt"]
    addition = context["segment_prompts"].get(segment_index + 1, "")
    if addition:
        prompt = f"{prompt}\n{addition}" if prompt else addition
    images = context["reference_images"]
    video = context["reference_video"]
    video_audio = context["reference_video_audio"]
    if video is not None and context["sync_video"]:
        end = frame_start + frame_count
        if video.shape[0] < end:
            raise ValueError(f"参考视频不足以覆盖第 {segment_index + 1} 段 [{frame_start}:{end}]；续写请改用素材参考或提供完整视频")
        video = video[frame_start:end]
        video_audio = _audio_window(video_audio, frame_start, end)
    elif video is not None and video.shape[0] >= 5:
        # Match the native reference clip's floor rule for both paired streams.
        reference_frames = 5 + ((video.shape[0] - 5) // 17) * 17
        video = video[:reference_frames]
        video_audio = _audio_window(video_audio, 0, reference_frames)
    # The native reference node caps references to its target length. Independent
    # material references must retain their own duration; its empty target is discarded.
    native_length = max(frame_count, video.shape[0]) if video is not None and not context["sync_video"] else frame_count
    size = dict(width=video_shape[-1] * 16, height=video_shape[-2] * 16, length=native_length)
    if any(value is not None for value in (images, video, context["reference_audio"])):
        prepared = MiniMaxH3ReferenceToVideo.execute(
            context["clip"], prompt=prompt, vae=context["vae"], audio_vae=context["audio_vae"],
            ref_images={f"ref_image_{i + 1}": image[None] for i, image in enumerate(images)} if images is not None else None,
            ref_videos={"ref_video_1": video} if video is not None else None,
            ref_video_audios={"ref_video_audio_1": video_audio} if video_audio is not None else None,
            ref_audios={"ref_audio_1": context["reference_audio"]} if context["reference_audio"] is not None else None,
            ref_image_size="match", **size)
    else:
        prepared = MiniMaxH3ImageToVideo.execute(context["clip"], context["vae"], prompt=prompt, **size)
    positive = prepared.result[0]
    del prepared
    guides = motion_keyframes(previous_tail, tuple(video_shape)) if previous_tail is not None and context["motion_context"] else []
    templates = guider.original_conds.get("positive") or [{}]
    rebuilt = []
    for template in templates:
        # Native text/reference payloads must match the newly tokenized segment.
        extras = {key: value for key, value in template.items()
                  if key not in ("cross_attn", "uuid", "pooled_output", "minimax_token_tags", "minimax_refs")}
        for embedding, values in positive:
            metadata = dict(extras, **values)
            keyframes = list(extras.get("minimax_keyframes", [])) + list(values.get("minimax_keyframes", [])) + guides
            for keyframe in keyframes:
                latent = keyframe.get("latent")
                if latent is not None and latent.shape[-2:] != video_shape[-2:]:
                    raise ValueError("上游首尾/引导潜空间的分辨率与当前采样不匹配；请按当前分辨率重新建立引导条件")
            if keyframes:
                metadata["minimax_keyframes"] = keyframes
            rebuilt.append([embedding, metadata])
    guider.inner_set_conds({"positive": rebuilt})
    return guider
