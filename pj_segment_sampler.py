import torch

import comfy.utils
import comfy.model_management
import comfy.nested_tensor
import latent_preview

from .segment_timing import plan_segments, resolve_plan, segment_seconds
from .segment_boundaries import constrain_boundaries, restore_preserved
from .segment_sampling import (SegmentCache, sampling_signature, fingerprint, generate_noise,
                               slice_mask, segment_guider, blend_sequence)


class PJ_SegmentSampler:
    """
    PJ二采分段精修采样器 (Second-Pass Segment Sampler)
    专为 MiniMax H3 及长视频 2.0MP 高清精修设计：
    1. 按时间切分潜空间，降低单段采样工作量；
    2. 使用 H3 17n+5 帧网格，视频重叠区继承前段结果；
    3. 局部重跑从目标段起重建后续缓存；
    4. 可保留输入音频潜空间；不保证画面与音频嘴型一致；
    5. 可在采样结束后请求 ComfyUI 卸载模型！
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE", {"tooltip": "随机噪声源 (连接 RandomNoise 节点)"}),
                "guider": ("GUIDER", {"tooltip": "引导器 (连接 BasicGuider 节点)"}),
                "sampler": ("SAMPLER", {"tooltip": "采样算法 (推荐 er_sde 或 euler)"}),
                "sigmas": ("SIGMAS", {"tooltip": "调度器步长 (二采高清精修，通常 2~3 步)"}),
                "latent_image": ("LATENT", {"tooltip": "输入待精修的高清潜空间 (来自 3D 潜空间超分放大后的 AV Latent)"}),
            },
            "optional": {
                "segment_plan": (
                    "PJ_SEGMENT_PLAN",
                    {
                        "tooltip": "可选；连接时按规划分段，未连接时按输入完整长度采样",
                    },
                ),
                "run_mode": (
                    ["智能缓存(仅重跑未锁定分段)", "全部精修(刷新全部缓存)"]
                    + [f"仅精修第{i}段" for i in range(1, 21)]
                    + ["按缺陷秒数精修 (下方输入秒数)", "自定义指定分段(下方输入)"],
                    {
                        "default": "智能缓存(仅重跑未锁定分段)",
                        "tooltip": "从选定段开始重采并重建后续段；前面的有效缓存可复用，后续锁定不能阻止接缝更新",
                    },
                ),
                "target_time_second": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "按缺陷秒数定位起始段，并重建该段及后续段",
                    },
                ),
                "custom_rerun_segments": (
                    "STRING",
                    {
                        "default": "5",
                        "multiline": False,
                        "tooltip": "从所选段号中最早的一段起重建后续段，例如2,5表示从第2段起重做",
                    },
                ),
                "seg1_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第1段；前段重做会重建后续段"}),
                "seg2_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第2段；前段重做会重建后续段"}),
                "seg3_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第3段；前段重做会重建后续段"}),
                "seg4_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第4段；前段重做会重建后续段"}),
                "seg5_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第5段；前段重做会重建后续段"}),
                "seg6_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第6段；前段重做会重建后续段"}),
                "seg7_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第7段；前段重做会重建后续段"}),
                "seg8_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第8段；前段重做会重建后续段"}),
                "seg9_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第9段；前段重做会重建后续段"}),
                "seg10_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第10段；前段重做会重建后续段"}),
                "seg11_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第11段；前段重做会重建后续段"}),
                "seg12_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第12段；前段重做会重建后续段"}),
                "seg13_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第13段；前段重做会重建后续段"}),
                "seg14_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第14段；前段重做会重建后续段"}),
                "seg15_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第15段；前段重做会重建后续段"}),
                "seg16_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第16段；前段重做会重建后续段"}),
                "seg17_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第17段；前段重做会重建后续段"}),
                "seg18_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第18段；前段重做会重建后续段"}),
                "seg19_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第19段；前段重做会重建后续段"}),
                "seg20_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "有效缓存且前段未重做时复用第20段；前段重做会重建后续段"}),
                "blend_mode": (
                    ["余弦平滑(推荐)", "线性过渡", "硬切"],
                    {"default": "余弦平滑(推荐)", "tooltip": "仅用于未保留原音时的音频重叠过渡；视频始终承接前段并去重拼接"},
                ),
                "keep_original_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "开启(原音保真)",
                        "label_off": "二采融合",
                        "tooltip": "输出直接保留输入音频潜空间；音频仍参与模型联合采样，但其新结果不写回",
                    },
                ),
                "unload_model_after_sampling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "开启(释放显存)",
                        "label_off": "保留显存",
                        "tooltip": "采样完成后请求ComfyUI卸载模型；VAE解码仍受分辨率、切块和可用显存影响",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("完整潜空间", "去噪潜空间")
    FUNCTION = "execute"
    CATEGORY = "PJ工具箱/分段采样"

    def __init__(self):
        self._cache = SegmentCache()

    def plan_segments(self, total_v, total_a, split_mode, chunk_seconds, overlap_seconds,
                      num_segments, custom_cuts, fps, align_51_frames=False):
        return plan_segments(total_v, total_a, split_mode, chunk_seconds, overlap_seconds,
                             num_segments, custom_cuts, fps, align_51_frames)

    def execute(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        segment_plan=None,
        split_mode="按秒自动切分",
        chunk_seconds=8.0,
        overlap_seconds=2.0,
        num_segments=2,
        custom_cuts="0-8, 6-14, 12-20",
        align_51_frames=False,
        run_mode="全部精修(刷新全部缓存)",
        target_time_second=0.0,
        custom_rerun_segments="5",
        seg1_lock=False,
        seg2_lock=False,
        seg3_lock=False,
        seg4_lock=False,
        seg5_lock=False,
        seg6_lock=False,
        seg7_lock=False,
        seg8_lock=False,
        blend_mode="余弦平滑(推荐)",
        keep_original_audio=True,
        unload_model_after_sampling=True,
        fps=24.0,
        **kwargs,
    ):

        # Compatibility fallbacks
        if isinstance(run_mode, bool) or str(run_mode).lower() in ("true", "false"):
            run_mode = "智能缓存(仅重跑未锁定分段)"
        if "preserve_original_audio" in kwargs:
            keep_original_audio = kwargs["preserve_original_audio"]
        if blend_mode in ("cosine", "余弦平滑(推荐)"):
            norm_blend = "cosine"
        elif blend_mode in ("linear", "线性过渡"):
            norm_blend = "linear"
        else:
            norm_blend = "hard_cut"

        latent = latent_image.get("samples")
        if latent is None:
            raise ValueError("latent_image 中未找到 samples 张量")

        is_nested = hasattr(latent, "is_nested") and latent.is_nested
        if is_nested:
            streams = list(latent.unbind())
            full_video = streams[0]
            full_audio = streams[1] if len(streams) > 1 else None
        else:
            full_video = latent
            full_audio = None

        orig_audio_for_bypass = full_audio.cpu().clone() if full_audio is not None and keep_original_audio else None

        total_v = full_video.shape[2]
        total_a = full_audio.shape[-1] if full_audio is not None else 0

        chunks, fps = resolve_plan(
            latent, segment_plan, split_mode, chunk_seconds, overlap_seconds,
            num_segments, custom_cuts, fps, align_51_frames,
            full_length=segment_plan is None or "不分段" in str(run_mode) or "全量采样" in str(run_mode),
        )
        n_chunks = len(chunks)
        self._cache.retain(n_chunks)
        context_signature = sampling_signature(guider, sampler, sigmas)

        print("\n" + "=" * 65)
        print(f"[PJ二采分段采样器] 输入潜空间总计: {total_v} 视频步 (~{segment_seconds(0, total_v, fps)[1]:.2f}s), 音频: {total_a} 步")
        print(f"[PJ二采分段采样器] 运行模式: {run_mode} | 计划切分: {n_chunks} 段")
        for i, (vs, ve, as_, ae) in enumerate(chunks):
            print(f"  - 分段 {i + 1}/{n_chunks}: 视频潜空间 [{vs}:{ve}], 音频 [{as_}:{ae}]")
        print("=" * 65)

        # Segment control configuration (支持到 20 段，兼容形参以及 kwargs)
        explicit_locks = [
            seg1_lock, seg2_lock, seg3_lock, seg4_lock,
            seg5_lock, seg6_lock, seg7_lock, seg8_lock
        ]
        locks = [
            bool(kwargs[f"seg{i}_lock"] if f"seg{i}_lock" in kwargs else (explicit_locks[i - 1] if i <= 8 else False))
            for i in range(1, 21)
        ]

        # 计算目标缺陷秒数所属的分段编号 (如果处于缺陷秒数精修模式)
        target_defect_seg = None
        if "缺陷秒数" in str(run_mode):
            for seg_i, (vs, ve, as_, ae) in enumerate(chunks):
                s_sec, e_sec = segment_seconds(vs, ve, fps)
                if s_sec <= target_time_second < e_sec:
                    target_defect_seg = seg_i + 1
                    break
            if target_defect_seg is not None:
                print(f"[PJ二采分段采样器] 目标秒数 {target_time_second:.2f}s 匹配第 {target_defect_seg} 段，从该段起重建后续结果")
            else:
                target_defect_seg = 1
                print(f"[PJ二采分段采样器] ⚠️ 目标秒数 {target_time_second:.2f}s 超出范围，默认精修第 1 段")

        sampled_chunks = []
        x0_chunks = []
        has_any_x0 = False
        rebuilding = False
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        for idx, (v_start, v_end, a_start, a_end) in enumerate(chunks):
            chunk_v = full_video[:, :, v_start:v_end].clone()
            if is_nested and full_audio is not None:
                chunk_a = full_audio[..., a_start:a_end].clone()
                chunk_latent_samples = comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))
            else:
                chunk_latent_samples = chunk_v
            bounds = (v_start, v_end, a_start, a_end)
            chunk_mask = slice_mask(latent_image.get("noise_mask"), bounds)
            cache_signature = fingerprint(("refine-boundaries-v1", context_signature, bounds, chunk_latent_samples,
                                           chunk_mask, latent_image.get("batch_index"), type(noise),
                                           {k: v for k, v in vars(noise).items() if k != "seed"} if noise is not None else None))
            cache_valid = self._cache.entries.get(idx) == cache_signature
            seg_lock = locks[idx] if idx < len(locks) else False

            should_resample = True
            if run_mode == "全部精修(刷新全部缓存)":
                should_resample = True
            elif "缺陷秒数" in str(run_mode):
                should_resample = (idx + 1) == target_defect_seg
            elif run_mode.startswith("仅精修第"):
                target_seg_num = int(run_mode.replace("仅精修第", "").replace("段", "").strip())
                should_resample = (idx + 1) == target_seg_num
            elif "自定义" in run_mode:
                targets = [int(s.strip()) for s in str(custom_rerun_segments).split(",") if s.strip()]
                should_resample = (idx + 1) in targets
            elif run_mode == "智能缓存(仅重跑未锁定分段)":
                if seg_lock and cache_valid:
                    should_resample = False
                else:
                    should_resample = True

            # Cache hit check
            cached = self._cache.load(idx, cache_signature) if not rebuilding and not should_resample else None
            if cached is not None:
                sampled_chunks.append(cached[0])
                x0_chunks.append(cached[1])
                has_any_x0 |= cached[1] is not None
                print(f"[PJ分段采样] 第 {idx + 1}/{n_chunks} 段复用缓存")
                continue

            if not rebuilding:
                # Invalidate the dependent suffix before sampling, including on cancellation.
                self._cache.retain(idx)
                rebuilding = True
            sources = [(chunks[i], value.unbind()[0] if is_nested else value)
                       for i, value in enumerate(sampled_chunks)
                       if max(v_start, chunks[i][0]) < min(v_end, chunks[i][1])]
            if sources:
                video_mask = chunk_mask.unbind()[0] if chunk_mask is not None and chunk_mask.is_nested else chunk_mask
                chunk_v, video_mask = constrain_boundaries(chunk_v, video_mask, bounds, sources)
                if is_nested:
                    audio_mask = (chunk_mask.unbind()[1] if chunk_mask is not None and chunk_mask.is_nested
                                  else torch.ones((chunk_a.shape[0], 1, 1, chunk_a.shape[-1]),
                                                  device=chunk_a.device, dtype=torch.float32))
                    chunk_latent_samples = comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))
                    chunk_mask = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
                else:
                    chunk_latent_samples, chunk_mask = chunk_v, video_mask
            del sources

            print(f"\n[PJ二采分段采样器] >>> 开始精修第 {idx + 1}/{n_chunks} 段 (视频 [{v_start}:{v_end}]) ...")

            # Generate from the full shape so equal seeds share noise at equal absolute positions.
            segment_noise = generate_noise(noise, latent_image, getattr(noise, "seed", 0))
            if is_nested:
                noise_streams = segment_noise.unbind()
                chunk_noise = comfy.nested_tensor.NestedTensor((
                    noise_streams[0][:, :, v_start:v_end].clone(),
                    noise_streams[1][..., a_start:a_end].clone(),
                ))
                del noise_streams
            else:
                chunk_noise = segment_noise[:, :, v_start:v_end].clone()
            del segment_noise
            chunk_guider = segment_guider(guider, bounds, True)

            chunk_x0_output = {}
            chunk_callback = latent_preview.prepare_callback(
                chunk_guider.model_patcher, sigmas.shape[-1] - 1, chunk_x0_output
            )
            chunk_out = chunk_guider.sample(
                chunk_noise,
                chunk_latent_samples,
                sampler,
                sigmas,
                denoise_mask=chunk_mask,
                callback=chunk_callback,
                disable_pbar=disable_pbar,
                seed=getattr(noise, "seed", None),
            )
            chunk_out_cpu = chunk_out.to("cpu")
            restore_preserved(chunk_out_cpu, chunk_latent_samples, chunk_mask)
            sampled_chunks.append(chunk_out_cpu)

            if "x0" in chunk_x0_output:
                has_any_x0 = True
                x0_c = chunk_x0_output["x0"]
                if is_nested and not getattr(x0_c, "is_nested", False):
                    latent_shapes = [x.shape for x in chunk_out.unbind()]
                    x0_c = comfy.nested_tensor.NestedTensor(
                        comfy.utils.unpack_latents(x0_c, latent_shapes)
                    )
                x0_chunks.append(chunk_guider.model_patcher.model.process_latent_out(x0_c.cpu()))
                restore_preserved(x0_chunks[-1], chunk_latent_samples, chunk_mask)
                del x0_c
            else:
                x0_chunks.append(None)
            self._cache.save(idx, cache_signature, chunk_out_cpu, x0_chunks[-1])

            del chunk_v, chunk_latent_samples, chunk_out, chunk_x0_output, chunk_guider
            if "chunk_a" in locals() and chunk_a is not None:
                del chunk_a
            if "chunk_noise" in locals() and chunk_noise is not None:
                del chunk_noise
            comfy.model_management.soft_empty_cache()
            print(f"[PJ二采分段采样器] <<< 第 {idx + 1}/{n_chunks} 段精修完成，已存入缓存。")

        # Blending
        print("\n[PJ二采分段采样器] 视频重叠区保留前段结果，按完整时间线拼接")
        video_intervals = [(c[0], c[1]) for c in chunks]
        audio_intervals = [(c[2], c[3]) for c in chunks]

        chunk_videos = []
        chunk_audios = []
        for c in sampled_chunks:
            if hasattr(c, "is_nested") and c.is_nested:
                st = list(c.unbind())
                chunk_videos.append(st[0])
                if len(st) > 1:
                    chunk_audios.append(st[1])
            else:
                chunk_videos.append(c)

        blended_video = blend_sequence(
            chunk_videos, video_intervals, full_video, "保留边界"
        )

        if keep_original_audio and orig_audio_for_bypass is not None:
            print("[PJ二采分段采样器] 输出保留输入音频潜空间")
            blended_audio = orig_audio_for_bypass
        elif chunk_audios and total_a > 0:
            blended_audio = blend_sequence(
                chunk_audios, audio_intervals, full_audio, norm_blend, is_audio=True
            )
        else:
            blended_audio = None

        if blended_audio is not None:
            result_samples = comfy.nested_tensor.NestedTensor((blended_video, blended_audio))
        else:
            result_samples = blended_video

        # X0 output processing
        result_denoised = None
        if has_any_x0:
            valid_x0_videos = []
            valid_x0_audios = []
            for i, x0_c in enumerate(x0_chunks):
                if x0_c is not None:
                    if hasattr(x0_c, "is_nested") and x0_c.is_nested:
                        x0_streams = list(x0_c.unbind())
                        valid_x0_videos.append(x0_streams[0])
                        valid_x0_audios.append(x0_streams[1])
                    else:
                        valid_x0_videos.append(x0_c)
                else:
                    valid_x0_videos.append(chunk_videos[i])
                    if chunk_audios:
                        valid_x0_audios.append(chunk_audios[i])
            blended_x0_video = blend_sequence(
                valid_x0_videos, video_intervals, full_video, "保留边界"
            )
            if blended_audio is not None:
                denoised_audio = blended_audio if keep_original_audio else blend_sequence(
                    valid_x0_audios, audio_intervals, full_audio, norm_blend, is_audio=True
                )
                result_denoised = comfy.nested_tensor.NestedTensor((blended_x0_video, denoised_audio))
            else:
                result_denoised = blended_x0_video

        if unload_model_after_sampling:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            print("[PJ分段采样] 已请求卸载模型；解码峰值仍取决于分辨率及 VAE 配置")

        print("[PJ二采分段采样器] 全部多段二采精修完成！准备输出结果。\n" + "=" * 65)

        out_denoised = dict(latent_image, samples=result_denoised if result_denoised is not None else result_samples)
        return (dict(latent_image, samples=result_samples), out_denoised)


PJSegmentSampler = PJ_SegmentSampler
