"""Encode an authoritative soundtrack and lock it through the native AV mask."""

import torch
import torch.nn.functional as F
import comfy.nested_tensor
from comfy_extras.nodes_audio import VAEEncodeAudio

from .segment_timing import FPS, AUDIO_FPS


def encode_soundtrack(vae, audio, frames):
    if vae is None or audio is None:
        raise ValueError("保留指定音轨需要连接指定音轨和音频 VAE")
    samples = round(frames / FPS * audio["sample_rate"])
    waveform = audio["waveform"][..., :samples]
    if waveform.shape[-1] < samples:
        waveform = F.pad(waveform, (0, samples - waveform.shape[-1]))
    encoded = VAEEncodeAudio.execute(vae, dict(audio, waveform=waveform)).result[0]["samples"]
    length = round(frames / FPS * AUDIO_FPS)
    encoded = encoded[..., :length]
    if encoded.shape[-1] < length:
        encoded = torch.cat((encoded, encoded[..., -1:].expand(*encoded.shape[:-1], length - encoded.shape[-1])), dim=-1)
    return encoded.to(device="cpu", copy=True)


def lock_soundtrack(samples, mask, soundtrack):
    if not samples.is_nested:
        raise ValueError("保留指定音轨需要 H3 音视频潜空间")
    video, audio = samples.unbind()
    if soundtrack.shape[1:] != audio.shape[1:]:
        raise ValueError("指定音轨编码与当前 H3 音频潜空间的形状不匹配")
    audio = soundtrack.to(device=audio.device).expand(audio.shape[0], -1, -1, -1).clone()
    if mask is not None and mask.is_nested:
        video_mask = mask.unbind()[0]
    elif mask is not None:
        video_mask = mask
    else:
        video_mask = torch.ones((video.shape[0], 1, video.shape[2], 1, 1), device=video.device)
    audio_mask = torch.zeros((audio.shape[0], 1, 1, audio.shape[-1]), device=audio.device)
    return comfy.nested_tensor.NestedTensor((video, audio)), comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
