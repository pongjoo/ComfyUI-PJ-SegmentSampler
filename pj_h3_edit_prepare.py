"""Prepare source-backed H3 video editing with an external pixel mask."""

import torch
import torch.nn.functional as F
import comfy.nested_tensor
import comfy.sampler_helpers

from .segment_timing import frames_for_tokens


class PJ_H3EditPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "av_latent": ("LATENT", {"tooltip": "接 MiniMaxH3ReferenceToVideo 的第2个输出：目标音视频潜空间"}),
            "source_latent": ("LATENT", {"tooltip": "原视频以相同尺寸、相同时间范围经H3视频VAE编码的结果"}),
            "edit_mask": ("MASK", {"tooltip": "SAM或手绘遮罩：白色允许编辑，黑色保留原视频。与编码视频同尺寸，逐帧对应；单张遮罩应用于全片"}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("编辑潜空间",)
    FUNCTION = "prepare"
    CATEGORY = "PJ工具箱/分段采样"

    def prepare(self, av_latent, source_latent, edit_mask):
        samples = av_latent["samples"]
        if not samples.is_nested or len(samples.unbind()) != 2:
            raise ValueError("目标输入需要H3音视频潜空间，请连接 MiniMaxH3ReferenceToVideo 的第2个输出")
        video, audio = samples.unbind()
        source = source_latent["samples"]
        if video.ndim != 5 or video.shape[1] != 24 or video.shape[2] % 5 != 2:
            raise ValueError("目标输入需要原生H3视频布局和17n+5帧长度")
        if source.is_nested or source.shape != video.shape:
            raise ValueError("原视频编码与目标视频潜空间形状不一致，请统一分辨率、帧数和批次数")
        frames = frames_for_tokens(video.shape[2])
        height, width = video.shape[-2:]
        if edit_mask.ndim != 3 or edit_mask.shape[1:] != (height * 16, width * 16):
            raise ValueError("编辑遮罩必须为[帧数,高度,宽度]，且尺寸与原视频编码前一致；请先统一缩放视频和遮罩")
        if edit_mask.shape[0] != 1 and edit_mask.shape[0] < frames:
            raise ValueError(f"编辑遮罩不足{frames}帧；请连接完整跟踪遮罩，或使用单张静态遮罩")
        mask = edit_mask[:frames].to(device=source.device, dtype=torch.float32)
        if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError("编辑遮罩需要有限的0～1数值")
        # Match H3's 2x2 latent patches (32x32 pixels); union each native temporal span.
        pooled = F.max_pool2d(mask[:, None], 32, 32, ceil_mode=True)[:, 0]
        if mask.shape[0] == 1:
            pooled = pooled.expand(video.shape[2], -1, -1)
        else:
            spans = [frames_for_tokens(i + 1) - frames_for_tokens(i) for i in range(video.shape[2])]
            pooled = torch.stack([part.amax(dim=0) for part in pooled.split(spans)])
        video_mask = pooled.repeat_interleave(2, -2).repeat_interleave(2, -1)[None, None, :, :height, :width]

        existing = av_latent.get("noise_mask")
        masks = list(existing.unbind()) if existing is not None and existing.is_nested else [existing]
        if masks[0] is not None:
            video_mask = video_mask * comfy.sampler_helpers.prepare_mask(masks[0].float(), source.shape, source.device)
        audio_mask = masks[1].clone() if len(masks) > 1 else torch.ones(
            (audio.shape[0], 1, 1, audio.shape[-1]), dtype=torch.float32, device=audio.device)
        output = dict(av_latent, samples=comfy.nested_tensor.NestedTensor((source.clone(), audio)),
                      noise_mask=comfy.nested_tensor.NestedTensor((video_mask, audio_mask)))
        return (output,)
