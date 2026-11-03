import os
import sys
import math
import logging
import torch

import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.nested_tensor
import latent_preview

log = logging.getLogger("PJ_SegmentSampler")


def is_h3_aligned(total_vt: int) -> bool:
    """MiniMax H3 latent temporal dimension satisfies T = 5k + 2."""
    return (total_vt - 2) % 5 == 0


def plan_segments(
    total_vt: int,
    total_at: int,
    split_mode: str = "auto_by_seconds",
    chunk_seconds: float = 8.0,
    overlap_seconds: float = 2.0,
    num_segments: int = 2,
    custom_cuts: str = "",
    fps: float = 24.0,
    align_51_frames: bool = False,
) -> list[tuple[int, int, int, int]]:
    """
    Plan chunk intervals (v_start, v_end, a_start, a_end).
    For MiniMax H3:
      - Video latent steps follow FRAME_PER_TOKEN = (1, 4, 4, 4, 4) with period 5.
      - Any start at a multiple of 5 (5p) matches the full-sequence frame cadence.
      - Any chunk length of 5m + 2 matches the VAE and DiT temporal phase.
      - Audio latent FPS is 40 Hz, video FPS is 24 Hz -> a = round(frames * 5 / 3).
    """
    is_h3 = is_h3_aligned(total_vt)
    total_k = (total_vt - 2) // 5 if is_h3 else max(1, total_vt // 5)

    # 1. Custom cuts mode
    if split_mode == "custom_cuts" and custom_cuts and custom_cuts.strip():
        chunks = []
        parts = custom_cuts.split(",")
        for p_str in parts:
            p_str = p_str.strip()
            if not p_str or "-" not in p_str:
                continue
            s_val, e_val = [float(x.strip()) for x in p_str.split("-")]
            if is_h3:
                p_start = max(0, int(round(s_val * fps / 17.0)))
                p_end = max(p_start + 1, int(round((e_val * fps - 5.0) / 17.0)))
                v_start = 5 * p_start
                v_end = min(total_vt, 5 * p_end + 2)
                a_start = int(round(17 * p_start * 5.0 / 3.0)) if total_at > 0 else 0
                a_end = (
                    total_at
                    if v_end >= total_vt
                    else int(round((17 * p_end + 5) * 5.0 / 3.0))
                )
            else:
                v_start = max(0, int(round(s_val * fps)))
                v_end = min(total_vt, int(round(e_val * fps)))
                a_start = int(round(s_val * 40.0)) if total_at > 0 else 0
                a_end = min(total_at, int(round(e_val * 40.0))) if total_at > 0 else 0

            if total_at > 0:
                a_end = min(total_at, max(a_start + 1, a_end))
            chunks.append((v_start, v_end, a_start, a_end))
        if chunks:
            if chunks[-1][1] < total_vt:
                chunks[-1] = (chunks[-1][0], total_vt, chunks[-1][2], total_at)
            return chunks

    # 2. Determine target segments & overlap
    if is_h3:
        target_chunk_frames = chunk_seconds * fps
        raw_chunk_k = max(2, int(round((target_chunk_frames - 5.0) / 17.0)))

        if align_51_frames:
            # Snap chunk_k so that (17*k + 5) is divisible by 3 (i.e. k % 3 == 2)
            # This guarantees that the chunk duration maps to an EXACT integer number of 40Hz audio tokens (0.00ms drift)
            rem = raw_chunk_k % 3
            if rem == 2:
                chunk_k = max(2, raw_chunk_k)
            elif rem == 1:
                chunk_k = max(2, raw_chunk_k + 1)
            else:
                chunk_k = max(2, raw_chunk_k - 1)
        else:
            chunk_k = raw_chunk_k

        if chunk_k >= total_k:
            return [(0, total_vt, 0, total_at)]

        if split_mode == "fixed_segments":
            n_segs = max(2, int(num_segments))
            raw_stride_k = max(1, (total_k - 1) // (n_segs - 1))
            if align_51_frames:
                m_stride = max(1, round(raw_stride_k / 3.0))
                stride_k = 3 * m_stride
                chunk_k = max(chunk_k, stride_k + 2)
            else:
                stride_k = raw_stride_k
                chunk_k = max(chunk_k, stride_k + 1)
        else:
            # auto_by_seconds
            if align_51_frames:
                m_stride = max(1, chunk_k // 3)
                stride_k = 3 * m_stride
                if stride_k >= chunk_k:
                    stride_k = max(3, 3 * (m_stride - 1))
            else:
                target_overlap_frames = overlap_seconds * fps
                overlap_k = max(1, int(round((target_overlap_frames - 5.0) / 17.0)))
                if overlap_k >= chunk_k:
                    overlap_k = max(1, chunk_k - 1)
                stride_k = max(1, chunk_k - overlap_k)

        p_starts = []
        curr_p = 0
        while curr_p + chunk_k < total_k:
            p_starts.append(curr_p)
            curr_p += stride_k
        p_starts.append(curr_p)

        if align_51_frames:
            last_p_aligned = 3 * (max(0, total_k - chunk_k) // 3)
            if last_p_aligned not in p_starts and last_p_aligned < total_k:
                p_starts.append(last_p_aligned)
        else:
            last_p = max(0, total_k - chunk_k)
            if last_p not in p_starts and last_p < total_k:
                p_starts.append(last_p)

        p_starts = sorted(list(set(p_starts)))

        chunks = []
        for i, p in enumerate(p_starts):
            v_start = 5 * p
            if i == len(p_starts) - 1:
                v_end = total_vt
                a_end = total_at
            else:
                v_end = min(total_vt, 5 * (p + chunk_k) + 2)
                chunk_frames = 17 * ((v_end - v_start - 2) // 5) + 5
                a_end = min(
                    total_at,
                    int(round(17 * p * 5.0 / 3.0))
                    + int(round(chunk_frames * 5.0 / 3.0)),
                )
            a_start = int(round(17 * p * 5.0 / 3.0)) if total_at > 0 else 0
            chunks.append((v_start, v_end, a_start, a_end))
        return chunks
    else:
        # Non-H3 general video latents
        target_chunk_frames = int(round(chunk_seconds * fps))
        overlap_frames = int(round(overlap_seconds * fps))
        step = max(1, target_chunk_frames - overlap_frames)
        n_segs = max(2, math.ceil(total_vt / step))
        chunks = []
        for i in range(n_segs):
            v_start = i * step
            v_end = min(total_vt, v_start + target_chunk_frames)
            if v_start >= total_vt:
                break
            a_start = int(round(v_start / total_vt * total_at)) if total_at > 0 else 0
            a_end = int(round(v_end / total_vt * total_at)) if total_at > 0 else 0
            chunks.append((v_start, v_end, a_start, a_end))
            if v_end >= total_vt:
                break
        return chunks


def compute_blend_weights(
    intervals: list[tuple[int, int]],
    mode: str = "cosine",
    target_overlap_len: int | None = None,
) -> list[torch.Tensor]:
    """Compute mathematically normalized blending weights with tight lip-sync window support."""
    weights = []
    n = len(intervals)
    for i, (start, end) in enumerate(intervals):
        chunk_len = end - start
        w = torch.ones(chunk_len, dtype=torch.float32)
        if n > 1:
            # Left overlap ramp (rising from 0 to 1)
            if i > 0:
                prev_end = intervals[i - 1][1]
                overlap_left = prev_end - start
                if overlap_left > 0:
                    ramp_len = min(chunk_len, overlap_left)
                    t_win = (
                        ramp_len
                        if (target_overlap_len is None or target_overlap_len <= 0)
                        else min(ramp_len, target_overlap_len)
                    )
                    offset = (ramp_len - t_win) // 2
                    w_ramp = torch.zeros(ramp_len, dtype=torch.float32)
                    t = torch.arange(t_win, dtype=torch.float32)
                    if mode == "cosine":
                        ramp_core = 0.5 * (1.0 - torch.cos(math.pi * (t + 0.5) / t_win))
                    elif mode == "linear":
                        ramp_core = (t + 0.5) / t_win
                    elif mode == "hard_cut":
                        ramp_core = (t >= (t_win // 2)).float()
                    else:
                        ramp_core = 0.5 * (1.0 - torch.cos(math.pi * (t + 0.5) / t_win))
                    w_ramp[offset : offset + t_win] = ramp_core
                    if offset + t_win < ramp_len:
                        w_ramp[offset + t_win :] = 1.0
                    w[:ramp_len] = w_ramp

            # Right overlap ramp (falling from 1 to 0)
            if i < n - 1:
                next_start = intervals[i + 1][0]
                overlap_right = end - next_start
                if overlap_right > 0:
                    ramp_len = min(chunk_len, overlap_right)
                    t_win = (
                        ramp_len
                        if (target_overlap_len is None or target_overlap_len <= 0)
                        else min(ramp_len, target_overlap_len)
                    )
                    offset = (ramp_len - t_win) // 2
                    w_ramp = torch.ones(ramp_len, dtype=torch.float32)
                    t = torch.arange(t_win, dtype=torch.float32)
                    if mode == "cosine":
                        ramp_core = 0.5 * (1.0 + torch.cos(math.pi * (t + 0.5) / t_win))
                    elif mode == "linear":
                        ramp_core = 1.0 - (t + 0.5) / t_win
                    elif mode == "hard_cut":
                        ramp_core = (t < (t_win // 2)).float()
                    else:
                        ramp_core = 0.5 * (1.0 + torch.cos(math.pi * (t + 0.5) / t_win))
                    w_ramp[offset : offset + t_win] = ramp_core
                    if offset + t_win < ramp_len:
                        w_ramp[offset + t_win :] = 0.0
                    w[-ramp_len:] = w_ramp
        weights.append(w)
    return weights


class PJ_SegmentSampler:
    """
    PJ分段采样器 (PJ Segment Sampler)
    专为高分辨率长视频二采设计（支持 MiniMax H3 联合音视频潜空间及通用视频潜空间）。
    保留一采完整构图，沿时间轴将潜空间无损切分为多段带重叠区的窗口，
    各段共享全局时间切片噪声（保证重叠区噪声100%一致），
    低显存依次采样并平滑融合缝合，从根本上解决高分辨率长视频二采显存溢出与灰褐色噪点崩溃问题。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE", ),
                "guider": ("GUIDER", ),
                "sampler": ("SAMPLER", ),
                "sigmas": ("SIGMAS", ),
                "latent_image": ("LATENT", ),
                "split_mode": (
                    ["auto_by_seconds", "fixed_segments", "custom_cuts"],
                    {"default": "auto_by_seconds"},
                ),
                "chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": "每段最长秒数（auto_by_seconds 模式生效）",
                    },
                ),
                "overlap_seconds": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "相邻片段重叠秒数（设为 0 表示无重叠硬切）",
                    },
                ),
                "num_segments": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": 20,
                        "step": 1,
                        "tooltip": "固定分段数量（fixed_segments 模式生效）",
                    },
                ),
                "custom_cuts": (
                    "STRING",
                    {
                        "default": "0-8, 6-14, 12-20",
                        "multiline": False,
                        "tooltip": "自定义切分时间范围（秒），如：0-8, 6-14, 12-20",
                    },
                ),
                "blend_mode": (
                    ["cosine", "linear", "hard_cut"],
                    {"default": "cosine", "tooltip": "重叠区融合方式：余弦平滑（推荐）、线性过渡、硬切换"},
                ),
                "align_51_frames": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "开启51帧整数吸附",
                        "label_off": "关闭(经典连续切分)",
                        "tooltip": "开启后将窗口长度与步长自动吸附到51帧(85步音频/2.125秒)整数倍网格，消除理论浮点时间差；关闭则使用经典连续切分，保持原始设定，便于对比嘴型表现。",
                    },
                ),
                "keep_original_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "保留一采音频(无瑕疵/推荐)",
                        "label_off": "二采音频融合",
                        "tooltip": "二采核心目的是超分细化视频画面。勾选此项将直接保留一采原本完整无瑕的音频，彻底杜绝潜空间重叠区的声学相位抵消、空洞杂音或爆音！",
                    },
                ),
                "unload_model_after_sampling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "卸载模型释放显存(推荐)",
                        "label_off": "保留模型在显存",
                        "tooltip": "二采采样完成后自动将占巨量显存的 UNet/DiT 大模型卸载到 CPU 内存并清空显存缓存，让下游的 VAE 解码拥有充足纯净显存，从根源上杜绝 23G 爆显存风险！",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 1.0,
                        "tooltip": "视频帧率，MiniMax H3 默认为 24.0",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom/PJ"

    def execute(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        split_mode="auto_by_seconds",
        chunk_seconds=8.0,
        overlap_seconds=2.0,
        num_segments=2,
        custom_cuts="0-8, 6-14, 12-20",
        blend_mode="cosine",
        align_51_frames=False,
        keep_original_audio=True,
        unload_model_after_sampling=True,
        fps=24.0,
    ):
        latent = latent_image.copy()
        raw_samples = latent["samples"]

        # Fix empty latent channels if needed
        raw_samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            raw_samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = raw_samples

        # Detect nested tensor (MiniMax H3 joint video+audio) vs single tensor
        is_nested = hasattr(raw_samples, "is_nested") and raw_samples.is_nested
        if is_nested:
            streams = list(raw_samples.unbind())
            full_video = streams[0]  # [B, 24, T_v, H, W]
            full_audio = streams[1] if len(streams) > 1 else None  # [B, 32, 2, T_a]
            total_vt = full_video.shape[2]
            total_at = full_audio.shape[-1] if full_audio is not None else 0
        else:
            full_video = raw_samples
            full_audio = None
            total_vt = full_video.shape[2] if full_video.ndim >= 5 else full_video.shape[1]
            total_at = 0

        total_duration = total_vt / fps if not is_h3_aligned(total_vt) else (17 * ((total_vt - 2) // 5) + 5) / fps
        print(f"\n[PJ分段采样器] 输入潜空间总计: {total_vt} 视频步 ({total_duration:.2f}s), 音频步: {total_at}")

        # Plan chunks
        chunks = plan_segments(
            total_vt=total_vt,
            total_at=total_at,
            split_mode=split_mode,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            num_segments=num_segments,
            custom_cuts=custom_cuts,
            fps=fps,
            align_51_frames=align_51_frames,
        )

        n_chunks = len(chunks)
        print(f"[PJ分段采样器] 模式: {split_mode}, 计划切分: {n_chunks} 段")
        for idx, (vs, ve, as_, ae) in enumerate(chunks):
            v_sec = f"~{vs/fps:.2f}s - {ve/fps:.2f}s"
            print(f"  - 分段 {idx + 1}/{n_chunks}: 视频潜空间 [{vs}:{ve}] (长度 {ve - vs}, {v_sec}), 音频 [{as_}:{ae}] (长度 {ae - as_})")

        noise_mask = latent.get("noise_mask", None)

        # Single segment bypass
        if n_chunks <= 1:
            print("[PJ分段采样器] 仅需 1 段，执行直通采样...")
            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output
            )
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            samples = guider.sample(
                noise.generate_noise(latent),
                raw_samples,
                sampler,
                sigmas,
                denoise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=getattr(noise, "seed", None),
            )
            samples = samples.to(comfy.model_management.intermediate_device())

            out = latent.copy()
            out.pop("downscale_ratio_spacial", None)
            out.pop("downscale_ratio_temporal", None)
            out["samples"] = samples

            if "x0" in x0_output:
                x0 = x0_output["x0"]
                if is_nested and not getattr(x0, "is_nested", False):
                    latent_shapes = [x.shape for x in samples.unbind()]
                    x0 = comfy.nested_tensor.NestedTensor(
                        comfy.utils.unpack_latents(x0, latent_shapes)
                    )
                x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
                out_denoised = latent.copy()
                out_denoised["samples"] = x0_out
            else:
                out_denoised = out

            if unload_model_after_sampling:
                print("[PJ分段采样器] 采样完成：正在将主模型卸载到 CPU，彻底释放显存给后续 VAE 解码...")
                try:
                    comfy.model_management.unload_all_models()
                    comfy.model_management.soft_empty_cache(True)
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    print("[PJ分段采样器] ✅ 主模型已成功卸载，GPU 显存已完全释放！后续 VAE 解码将拥有充足纯净显存！")
                except Exception as exc:
                    log.warning(f"[PJ分段采样器] 卸载模型提示: {exc}")

            return (out, out_denoised)

        # Multi-segment chunked sampling
        # Step 1: Generate global noise once from full latent to guarantee shared identical noise in overlap regions
        print("[PJ分段采样器] 正在为完整序列生成全局噪声 (保证重叠区全局噪声切片一致)...")
        global_noise = noise.generate_noise(latent)
        if is_nested:
            gn_streams = list(global_noise.unbind())
            gn_video = gn_streams[0]
            gn_audio = gn_streams[1] if len(gn_streams) > 1 else None
        else:
            gn_video = global_noise
            gn_audio = None

        sampled_chunks = []
        x0_chunks = []
        has_any_x0 = False
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Step 2: Sequentially sample each chunk
        for idx, (v_start, v_end, a_start, a_end) in enumerate(chunks):
            print(f"\n[PJ分段采样器] >>> 开始采样第 {idx + 1}/{n_chunks} 段 (视频 [{v_start}:{v_end}]) ...")

            # Slice latent
            chunk_v = full_video[:, :, v_start:v_end, :, :].clone()
            if is_nested and full_audio is not None and a_end > a_start:
                chunk_a = full_audio[:, :, :, a_start:a_end].clone()
                chunk_latent_samples = comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))
            else:
                chunk_latent_samples = chunk_v

            # Slice noise
            chunk_nv = gn_video[:, :, v_start:v_end, :, :].clone()
            if is_nested and gn_audio is not None and a_end > a_start:
                chunk_na = gn_audio[:, :, :, a_start:a_end].clone()
                chunk_noise = comfy.nested_tensor.NestedTensor((chunk_nv, chunk_na))
            else:
                chunk_noise = chunk_nv

            # Slice noise mask if present
            chunk_mask = None
            if noise_mask is not None:
                if hasattr(noise_mask, "is_nested") and noise_mask.is_nested:
                    nm_streams = list(noise_mask.unbind())
                    chunk_mv = nm_streams[0][:, :, v_start:v_end, :, :].clone()
                    chunk_ma = nm_streams[1][:, :, :, a_start:a_end].clone() if len(nm_streams) > 1 else None
                    chunk_mask = comfy.nested_tensor.NestedTensor((chunk_mv, chunk_ma)) if chunk_ma is not None else chunk_mv
                elif noise_mask.ndim >= 5:
                    chunk_mask = noise_mask[:, :, v_start:v_end, :, :].clone()
                elif noise_mask.ndim == 4:
                    chunk_mask = noise_mask[:, v_start:v_end, :, :].clone()

            # Callback
            chunk_x0_output = {}
            chunk_callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, chunk_x0_output
            )

            chunk_out = guider.sample(
                chunk_noise,
                chunk_latent_samples,
                sampler,
                sigmas,
                denoise_mask=chunk_mask,
                callback=chunk_callback,
                disable_pbar=disable_pbar,
                seed=getattr(noise, "seed", None),
            )
            # Store on CPU/intermediate device to avoid VRAM piling up
            chunk_out = chunk_out.to("cpu")
            sampled_chunks.append(chunk_out)

            if "x0" in chunk_x0_output:
                has_any_x0 = True
                x0_c = chunk_x0_output["x0"]
                if is_nested and not getattr(x0_c, "is_nested", False):
                    latent_shapes = [x.shape for x in chunk_out.unbind()]
                    x0_c = comfy.nested_tensor.NestedTensor(
                        comfy.utils.unpack_latents(x0_c, latent_shapes)
                    )
                x0_chunks.append(x0_c.to("cpu"))
            else:
                x0_chunks.append(None)

            # Clean up temporary GPU chunk tensors before cache clearing
            del chunk_v, chunk_latent_samples
            if "chunk_a" in locals() and chunk_a is not None:
                del chunk_a
            if "chunk_noise" in locals() and chunk_noise is not None:
                del chunk_noise

            # Soft empty cache to release activation memory between chunks
            comfy.model_management.soft_empty_cache()
            print(f"[PJ分段采样器] <<< 第 {idx + 1}/{n_chunks} 段采样完成，已释放峰值显存缓存。")

        # Step 3: Overlap Blending
        print(f"\n[PJ分段采样器] 正在执行视频重叠区无缝融合 (融合算法: {blend_mode}) ...")
        video_intervals = [(c[0], c[1]) for c in chunks]
        audio_intervals = [(c[2], c[3]) for c in chunks]

        # Calculate tight transition window (in latents) from overlap_seconds to eliminate double-lip ghosting
        target_ov_latents = (
            max(1, int(round(overlap_seconds * fps / 17.0 * 5.0)))
            if overlap_seconds > 0
            else None
        )
        video_weights = compute_blend_weights(
            video_intervals, mode=blend_mode, target_overlap_len=target_ov_latents
        )
        target_ov_audio = (
            int(round(overlap_seconds * 40.0)) if overlap_seconds > 0 else None
        )
        audio_weights = (
            compute_blend_weights(
                audio_intervals, mode=blend_mode, target_overlap_len=target_ov_audio
            )
            if (is_nested and full_audio is not None and total_at > 0 and not keep_original_audio)
            else None
        )

        # Allocate blended full video tensor
        blended_video = torch.zeros_like(full_video, device="cpu", dtype=torch.float32)
        weight_video = torch.zeros(
            [1, 1, total_vt, 1, 1], device="cpu", dtype=torch.float32
        )

        for i, (vs, ve) in enumerate(video_intervals):
            c_sample = sampled_chunks[i]
            c_vid = (c_sample.unbind()[0] if is_nested else c_sample).float()
            w_v = video_weights[i].view(1, 1, -1, 1, 1)

            blended_video[:, :, vs:ve, :, :] += c_vid * w_v
            weight_video[:, :, vs:ve, :, :] += w_v

        blended_video = (blended_video / weight_video.clamp(min=1e-8)).to(
            full_video.dtype
        )

        # Handle Audio:
        if is_nested and full_audio is not None and total_at > 0:
            if keep_original_audio:
                # Golden standard: directly preserve the untouched pristine 1st-pass audio!
                print("[PJ分段采样器] 音频策略: 100% 保留一采原版完整音频 (零接缝、零相位抵消、零音质损耗)！")
                final_samples = comfy.nested_tensor.NestedTensor((blended_video, full_audio))
            else:
                # Blended 2nd pass audio
                blended_audio = torch.zeros_like(full_audio, device="cpu", dtype=torch.float32)
                weight_audio = torch.zeros(
                    [1, 1, 1, total_at], device="cpu", dtype=torch.float32
                )

                for i, (as_, ae) in enumerate(audio_intervals):
                    c_sample = sampled_chunks[i]
                    c_aud = c_sample.unbind()[1].float()
                    w_a = audio_weights[i].view(1, 1, 1, -1)

                    blended_audio[:, :, :, as_:ae] += c_aud * w_a
                    weight_audio[:, :, :, as_:ae] += w_a

                blended_audio = (blended_audio / weight_audio.clamp(min=1e-8)).to(
                    full_audio.dtype
                )
                final_samples = comfy.nested_tensor.NestedTensor((blended_video, blended_audio))
        else:
            final_samples = blended_video

        final_samples = final_samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = final_samples

        # Blend denoised output (x0) if available
        if has_any_x0 and all(x is not None for x in x0_chunks):
            blended_x0_v = torch.zeros_like(full_video, device="cpu", dtype=torch.float32)
            for i, (vs, ve) in enumerate(video_intervals):
                x0_c = x0_chunks[i]
                x0_vid = (x0_c.unbind()[0] if is_nested else x0_c).float()
                w_v = video_weights[i].view(1, 1, -1, 1, 1)
                blended_x0_v[:, :, vs:ve, :, :] += x0_vid * w_v

            blended_x0_v = (blended_x0_v / weight_video.clamp(min=1e-8)).to(
                full_video.dtype
            )

            if is_nested and full_audio is not None and total_at > 0:
                if keep_original_audio:
                    final_x0 = comfy.nested_tensor.NestedTensor((blended_x0_v, full_audio))
                else:
                    blended_x0_a = torch.zeros_like(full_audio, device="cpu", dtype=torch.float32)
                    for i, (as_, ae) in enumerate(audio_intervals):
                        x0_c = x0_chunks[i]
                        x0_aud = x0_c.unbind()[1].float()
                        w_a = audio_weights[i].view(1, 1, 1, -1)
                        blended_x0_a[:, :, :, as_:ae] += x0_aud * w_a
                    blended_x0_a = (blended_x0_a / weight_audio.clamp(min=1e-8)).to(
                        full_audio.dtype
                    )
                    final_x0 = comfy.nested_tensor.NestedTensor((blended_x0_v, blended_x0_a))
            else:
                final_x0 = blended_x0_v

            x0_out = guider.model_patcher.model.process_latent_out(final_x0.cpu())
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        if unload_model_after_sampling:
            print("[PJ分段采样器] 采样完成：正在将主模型卸载到 CPU，彻底释放显存给后续 VAE 解码...")
            try:
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache(True)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                print("[PJ分段采样器] ✅ 主模型已成功卸载，GPU 显存已完全释放！后续 VAE 解码将拥有充足纯净显存！")
            except Exception as exc:
                log.warning(f"[PJ分段采样器] 卸载模型提示: {exc}")

        print("[PJ分段采样器] 全部多段采样与融合完成！准备输出结果。\n")
        return (out, out_denoised)


# Aliases
PJSegmentSampler = PJ_SegmentSampler
