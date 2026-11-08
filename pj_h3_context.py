"""Rebuild native H3 conditions for one timeline window without decoding its result."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import time

import torch
import comfy.model_management
import nodes
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

from .segment_sampling import segment_guider
from .segment_timing import frames_for_tokens, tokens_for_frames_floor
from .segment_script import parse_segment_prompts, segment_prompt
from .segment_audio import encode_soundtrack, lock_soundtrack


class PJ_H3SegmentContext(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PJ_H3SegmentContext", display_name="PJ H3分段条件准备", category="PJ工具箱/分段采样",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("segment_prompts", multiline=True, default="{}",
                                tooltip='各段补充提示词，例如 {"2":"人物转身"}；段号从 1 开始'),
                io.Combo.Input("reference_mode", options=["原视频编辑（按时间裁剪）", "素材参考（保持完整）"]),
                io.Boolean.Input("motion_context", default=True),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input("ref_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_image", tooltip="每个接口对应一张参考图；接入后自动增加下一接口"),
                    prefix="ref_image_", min=0, max=100)),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_video", tooltip="24 FPS 视频帧序列；编辑模式下各路视频均按目标时间切片"),
                    prefix="ref_video_", min=0, max=100)),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_video_audio", tooltip="连接同编号参考视频的配套音轨，随该视频同步切片"),
                    prefix="ref_video_audio_", min=0, max=100)),
                io.Autogrow.Input("ref_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_audio", tooltip="独立参考音频，不按目标视频时间切片"),
                    prefix="ref_audio_", min=0, max=100)),
                io.Vae.Input("audio_vae", optional=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               display_name="参考图像尺寸", tooltip="使用官方规则：match 匹配目标像素面积；max 使用参考图像的最大尺寸策略"),
                io.Int.Input("width", optional=True, default=1344, min=32, max=nodes.MAX_RESOLUTION,
                             step=32, display_name="目标宽度"),
                io.Int.Input("height", optional=True, default=768, min=32, max=nodes.MAX_RESOLUTION,
                             step=32, display_name="目标高度"),
                io.Int.Input("length", optional=True, default=124, min=5, max=100000, step=1,
                             display_name="总帧数", tooltip="24 FPS；按17n+5向下对齐。只决定目标潜空间长度，参考条件仍按段编码"),
                io.Combo.Input("audio_mode", options=["参考声音生成", "生成新声音", "保留指定音轨"],
                               default="参考声音生成", optional=True, display_name="音频用途"),
                io.Audio.Input("soundtrack", optional=True, display_name="指定音轨",
                               tooltip="保留模式的完整时间线音轨；最终原声请沿此音频分支连接视频合成"),
            ],
            outputs=[
                io.Custom("PJ_H3_CONTEXT").Output(display_name="H3分段上下文"),
                io.Latent.Output(display_name="目标音视频潜空间"),
                io.Conditioning.Output(display_name="基础正面条件",
                                       tooltip="连接引导器；实际提示词和参考素材由PJ采样器读取H3分段上下文后逐段编码"),
            ],
        )

    @classmethod
    def execute(cls, width=1344, height=768, length=124, **kwargs):
        context, = cls.prepare(**kwargs)
        length = frames_for_tokens(tokens_for_frames_floor(length))
        base = MiniMaxH3ImageToVideo.execute(context["clip"], context["vae"], prompt="",
                                           width=width, height=height, length=length)
        latent = base.result[1]
        if context["audio_mode"] == "保留指定音轨":
            context["locked_audio"] = encode_soundtrack(context["audio_vae"], context["soundtrack"], length)
            samples, mask = lock_soundtrack(latent["samples"], None, context["locked_audio"])
            latent = dict(latent, samples=samples, noise_mask=mask)
        return io.NodeOutput(context, latent, base.result[0])

    @classmethod
    def prepare(cls, clip, vae=None, prompt="", segment_prompts="{}",
                reference_mode="原视频编辑（按时间裁剪）", motion_context=True,
                reference_images=None, reference_video=None, reference_video_audio=None,
                reference_audio=None, audio_vae=None, ref_images=None, ref_image_size="match",
                ref_videos=None, ref_video_audios=None, ref_audios=None,
                audio_mode="参考声音生成", soundtrack=None):
        prompts = parse_segment_prompts(segment_prompts)
        videos = _reference_inputs(ref_videos, reference_video, "ref_video")
        video_audios = _reference_inputs(ref_video_audios, reference_video_audio, "ref_video_audio")
        audios = _reference_inputs(ref_audios, reference_audio, "ref_audio")
        for name in video_audios:
            if "ref_video_" + name.rsplit("_", 1)[-1] not in videos:
                raise ValueError(f"配套视频音轨 {name} 需要连接同编号参考视频；独立音频请连接独立参考音频接口")
        images = list(reference_images) if reference_images is not None else []
        for name in sorted(ref_images or {}, key=lambda name: int(name.rsplit("_", 1)[-1])):
            image = ref_images[name]
            if image is not None:
                images.append(image[0].clone() if image.shape[0] > 1 else image[0])
        return ({
            "clip": clip, "vae": vae, "prompt": prompt,
            "segment_prompts": prompts,
            "audio_mode": audio_mode, "soundtrack": soundtrack,
            "sync_video": reference_mode == "原视频编辑（按时间裁剪）",
            "motion_context": motion_context, "reference_images": images or None,
            "ref_image_size": ref_image_size,
            "ref_videos": videos, "ref_video_audios": video_audios,
            "ref_audios": audios, "audio_vae": audio_vae,
        },)


def _reference_inputs(dynamic, legacy, prefix):
    values = {name: value for name, value in (dynamic or {}).items() if value is not None}
    if legacy is not None:
        first = f"{prefix}_0"
        if first in values:
            raise ValueError(f"{first} 重复输入：请只使用动态接口或旧版单输入参数之一")
        values[first] = legacy
    return {name: values[name] for name in sorted(values, key=lambda name: int(name.rsplit("_", 1)[-1]))}


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
    positive = _encode_segment_conditions(context, bounds, video_shape, segment_index)
    return _apply_segment_conditions(base_guider, context, bounds, video_shape, positive, previous_tail)


def _encode_segment_conditions(context, bounds, video_shape, segment_index, scene_cuts=()):
    if len(video_shape) != 5 or video_shape[1] != 24 or video_shape[2] != bounds[1] - bounds[0]:
        raise ValueError("H3 分段条件需要当前窗口的原生视频潜空间形状")
    frame_start = frames_for_tokens(bounds[0])
    frame_count = frames_for_tokens(video_shape[2])
    if bounds[0] % 5 or video_shape[2] % 5 != 2:
        raise ValueError("H3 分段条件需要按 17n+5 帧网格对齐的窗口")
    prompt = segment_prompt(context, segment_index, bounds)
    images = context["reference_images"]
    videos, video_audios = {}, {}
    for name, source in context["ref_videos"].items():
        audio_name = "ref_video_audio_" + name.rsplit("_", 1)[-1]
        video_audio = context["ref_video_audios"].get(audio_name) if context.get("audio_mode", "参考声音生成") == "参考声音生成" else None
        if context["sync_video"]:
            end = frame_start + frame_count
            scene_end = next((frames_for_tokens(cut) for cut in scene_cuts if bounds[0] < cut < bounds[1]), end)
            visual_end = min(end, scene_end)
            if source.shape[0] < visual_end:
                raise ValueError(f"参考视频不足以覆盖第 {segment_index + 1} 段 [{frame_start}:{visual_end}]（{name}）；续写请改用素材参考或提供完整视频")
            video = source[frame_start:visual_end]
            if visual_end < end:
                video = torch.cat((video, video[-1:].expand(end - visual_end, -1, -1, -1)))
            video_audio = _audio_window(video_audio, frame_start, end)
        else:
            # Match each native reference clip's floor rule for its paired streams.
            reference_frames = frames_for_tokens(tokens_for_frames_floor(source.shape[0]))
            video = source[:reference_frames]
            video_audio = _audio_window(video_audio, 0, reference_frames)
        videos[name] = video
        if video_audio is not None:
            video_audios[audio_name] = video_audio
    # The native reference node caps references to its target length. Independent
    # material references must retain their own duration; its empty target is discarded.
    native_length = max([frame_count] + [video.shape[0] for video in videos.values()])
    size = dict(width=video_shape[-1] * 16, height=video_shape[-2] * 16, length=native_length)
    audios = context["ref_audios"] if context.get("audio_mode", "参考声音生成") == "参考声音生成" else {}
    if images or videos or audios:
        prepared = MiniMaxH3ReferenceToVideo.execute(
            context["clip"], prompt=prompt, vae=context["vae"], audio_vae=context["audio_vae"],
            ref_images={f"ref_image_{i + 1}": image[None] for i, image in enumerate(images)} if images is not None else None,
            ref_videos=videos or None, ref_video_audios=video_audios or None,
            ref_audios=audios or None,
            ref_image_size=context["ref_image_size"], **size)
    else:
        prepared = MiniMaxH3ImageToVideo.execute(context["clip"], context["vae"], prompt=prompt, **size)
    return prepared.result[0]


def _apply_segment_conditions(base_guider, context, bounds, video_shape, positive, previous_tail=None, scene_cuts=()):
    guider = segment_guider(base_guider, bounds, context["sync_video"], scene_cuts=scene_cuts)
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


def _conditions_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _conditions_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_conditions_to_cpu(item) for item in value)
    return value


class _PreparedSegmentConditions:
    def __init__(self, directory, context, chunks, video_shape, scene_cuts):
        self.directory = Path(directory)
        self.context = context
        self.chunks = chunks
        self.video_shape = tuple(video_shape)
        self.locked_audio = None
        self.scene_cuts = scene_cuts

    def shape(self, index):
        start, end = self.chunks[index][:2]
        return (*self.video_shape[:2], end - start, *self.video_shape[3:])

    def guider(self, base_guider, index, previous_tail=None):
        positive = torch.load(self.directory / f"{index}.pt", map_location="cpu", weights_only=True)
        return _apply_segment_conditions(base_guider, self.context, self.chunks[index],
                                         self.shape(index), positive, previous_tail, self.scene_cuts)

    def constrain_audio(self, samples, mask, index):
        if self.locked_audio is None:
            return samples, mask
        start, end = self.chunks[index][2:]
        return lock_soundtrack(samples, mask, self.locked_audio[..., start:end])


@contextmanager
def prepare_segment_conditions(context, chunks, video_shape, indices, scene_cuts=()):
    """Encode only scheduled windows before sampling; discard all files on exit."""
    indices = list(indices)
    if context is None or not indices:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="pj_conditions_") as directory:
        prepared = _PreparedSegmentConditions(directory, context, chunks, video_shape, scene_cuts)
        started = time.perf_counter()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        try:
            if context.get("audio_mode") == "保留指定音轨":
                prepared.locked_audio = context.get("locked_audio")
                if prepared.locked_audio is None or prepared.locked_audio.shape[-1] < max(chunk[3] for chunk in chunks):
                    prepared.locked_audio = encode_soundtrack(context["audio_vae"], context.get("soundtrack"),
                                                              frames_for_tokens(video_shape[2]))
            for index in indices:
                comfy.model_management.throw_exception_if_processing_interrupted()
                print(f"[PJ条件] 预编码第 {index + 1}/{len(chunks)} 段")
                positive = (_encode_segment_conditions(context, chunks[index], prepared.shape(index), index, scene_cuts)
                            if scene_cuts else _encode_segment_conditions(context, chunks[index], prepared.shape(index), index))
                torch.save(_conditions_to_cpu(positive), prepared.directory / f"{index}.pt")
                del positive
        finally:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
        print(f"[PJ条件] {len(indices)} 段准备完成（{time.perf_counter() - started:.1f}秒），开始连续采样")
        yield prepared
