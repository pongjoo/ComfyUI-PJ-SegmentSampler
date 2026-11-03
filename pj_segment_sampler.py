import os
import sys
import math
import logging
import gc
import torch

import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.nested_tensor
import latent_preview

# Global in-memory cache for Second Pass segment latents
_SECOND_PASS_CACHE = {}


class PJ_SegmentSampler:
    """
    PJ二采分段精修采样器 (Second-Pass Segment Sampler)
    专为 MiniMax H3 及长视频 2.0MP 高清精修设计：
    1. 纯潜空间切分，阻断 40GB+ 显存爆炸；
    2. 支持 MiniMax H3 严苛的 17k+5 周期与 51 帧 RoPE 相位对齐；
    3. 支持各分段独立缓存与局部重跑；
    4. 100% 保留一采原版连续高保真音频，零接缝与声学相位损耗；
    5. 采样结束自动将 20GB DiT 模型卸载至 CPU，彻底解决 VAE 解码爆显存！
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
                "split_mode": (
                    ["按秒自动切分", "固定分段数", "自定义切点"],
                    {"default": "按秒自动切分", "tooltip": "二采切分规划策略"},
                ),
                "chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 2.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "自动模式下单段最大时长（秒），推荐 6.0 ~ 8.0 秒",
                    },
                ),
                "overlap_seconds": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.25,
                        "tooltip": "相邻分段重叠过渡时长（秒），推荐 1.5 ~ 2.5 秒",
                    },
                ),
                "num_segments": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": 20,
                        "step": 1,
                        "tooltip": "固定分段数模式下的分段数量",
                    },
                ),
                "custom_cuts": (
                    "STRING",
                    {
                        "default": "0-8, 6-14, 12-20",
                        "multiline": False,
                        "tooltip": "自定义切分秒数范围，逗号分隔，如：0-8, 6-14, 12-20",
                    },
                ),
                "align_51_frames": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "开启51帧吸附",
                        "label_off": "关闭",
                        "tooltip": "开启后将窗口长度吸附到51帧/85音频步网格，确保绝对RoPE相位一致，杜绝嘴型漂移；关闭则使用经典连续切分",
                    },
                ),
                "run_mode": (
                    [
                        "智能缓存(仅重跑未锁定分段)",
                        "全部精修(刷新全部缓存)",
                        "仅精修第1段",
                        "仅精修第2段",
                        "仅精修第3段",
                        "仅精修第4段",
                        "仅精修第5段",
                        "仅精修第6段",
                        "仅精修第7段",
                        "仅精修第8段",
                        "自定义指定分段(下方输入)",
                    ],
                    {
                        "default": "智能缓存(仅重跑未锁定分段)",
                        "tooltip": "二采精修控制台：可选择只对某一段进行二采重采样，其余段直接从缓存复用",
                    },
                ),
                "custom_rerun_segments": (
                    "STRING",
                    {
                        "default": "5",
                        "multiline": False,
                        "tooltip": "自定义重抽分段编号，支持单段或多段逗号分隔，例如：5 或 2, 5, 7",
                    },
                ),
                "seg1_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾后，跳过第1段二采计算直接读缓存"},
                ),
                "seg2_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾后，跳过第2段二采计算直接读缓存"},
                ),
                "seg3_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾后，跳过第3段二采计算直接读缓存"},
                ),
                "seg4_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾后，跳过第4段二采计算直接读缓存"},
                ),
                "seg5_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾后，跳过第5段二采计算直接读缓存"},
                ),
                "seg6_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾后，跳过第6段二采计算直接读缓存"},
                ),
                "seg7_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾后，跳过第7段二采计算直接读缓存"},
                ),
                "seg8_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾后，跳过第8段二采计算直接读缓存"},
                ),
                "blend_mode": (
                    ["余弦平滑(推荐)", "线性过渡", "硬切"],
                    {"default": "余弦平滑(推荐)", "tooltip": "重叠区无缝平滑算法"},
                ),
                "keep_original_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "开启(原音保真)",
                        "label_off": "二采融合",
                        "tooltip": "二采核心目的是超分细化视频画面。强烈建议勾选此项，100%保留一采无瑕疵连续音频，杜绝声学相位抵消与爆音！",
                    },
                ),
                "unload_model_after_sampling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "开启(释放显存)",
                        "label_off": "保留显存",
                        "tooltip": "二采完成后自动将 20GB DiT 大模型卸载至 CPU 内存，使下游 VAE 解码拥有充足纯净显存，从根源消除 23G+ 爆显存风险！",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "视频帧率，MiniMax H3 官方为 24.0"},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("完整潜空间", "去噪潜空间")
    FUNCTION = "execute"
    CATEGORY = "PJ工具箱/分段采样"

    def plan_segments(
        self,
        total_v,
        total_a,
        split_mode,
        chunk_seconds,
        overlap_seconds,
        num_segments,
        custom_cuts,
        fps,
        align_51_frames=False,
    ):
        v_fps = fps / 3.4
        a_fps = 40.0
        chunks = []

        # Mapping for split_mode compatibility
        if split_mode in ("auto_by_seconds", "按秒自动切分"):
            if align_51_frames:
                target_chunk_v = max(17, int(round(chunk_seconds * v_fps)))
                m_chunk = max(1, round((target_chunk_v - 2) / 15.0))
                win_v = m_chunk * 15 + 2
                target_ov_v = max(2, int(round(overlap_seconds * v_fps)))
                m_ov = max(1, round((target_ov_v - 2) / 15.0))
                stride_m = max(1, m_chunk - m_ov)
                stride_v = stride_m * 15
                v_cur = 0
                while v_cur < total_v:
                    v_start = v_cur
                    v_end = min(total_v, v_start + win_v)
                    if v_end < total_v:
                        rem = (v_end - 2) % 15
                        if rem != 0:
                            v_end = min(total_v, v_end + (15 - rem))
                    a_start = int(round(v_start * (85.0 / 15.0)))
                    a_end = min(total_a, int(round(v_end * (85.0 / 15.0))))
                    chunks.append((v_start, v_end, a_start, a_end))
                    if v_end >= total_v:
                        break
                    v_cur += stride_v
            else:
                win_v = max(5, int(round(chunk_seconds * v_fps)))
                rem = (win_v - 2) % 5
                if rem != 0:
                    win_v = win_v + (5 - rem)
                stride_v = max(5, int(round((chunk_seconds - overlap_seconds) * v_fps)))
                stride_v = (stride_v // 5) * 5
                if stride_v <= 0:
                    stride_v = 5
                v_cur = 0
                while v_cur < total_v:
                    v_start = (v_cur // 5) * 5
                    v_end = min(total_v, v_start + win_v)
                    if v_end < total_v:
                        rem_end = (v_end - 2) % 5
                        if rem_end != 0:
                            v_end = min(total_v, v_end + (5 - rem_end))
                    sec_start = v_start / v_fps
                    sec_end = v_end / v_fps
                    a_start = int(round(sec_start * a_fps))
                    a_end = min(total_a, int(round(sec_end * a_fps)))
                    chunks.append((v_start, v_end, a_start, a_end))
                    if v_end >= total_v:
                        break
                    v_cur += stride_v

        elif split_mode in ("fixed_segments", "固定分段数"):
            win_v = max(5, int(math.ceil(total_v / num_segments)))
            rem = (win_v - 2) % 5
            if rem != 0:
                win_v += 5 - rem
            ov_v = max(2, int(round(overlap_seconds * v_fps)))
            for i in range(num_segments):
                v_start = max(0, i * (win_v - ov_v))
                v_start = (v_start // 5) * 5
                v_end = min(total_v, v_start + win_v)
                if i == num_segments - 1:
                    v_end = total_v
                sec_s = v_start / v_fps
                sec_e = v_end / v_fps
                a_s = int(round(sec_s * a_fps))
                a_e = min(total_a, int(round(sec_e * a_fps)))
                chunks.append((v_start, v_end, a_s, a_e))

        elif split_mode in ("custom_cuts", "自定义切点"):
            pairs = [p.strip() for p in custom_cuts.split(",") if "-" in p]
            for p in pairs:
                try:
                    s_str, e_str = p.split("-")
                    s_sec, e_sec = float(s_str.strip()), float(e_str.strip())
                    v_start = max(0, int(round(s_sec * v_fps)))
                    v_start = (v_start // 5) * 5
                    v_end = min(total_v, int(round(e_sec * v_fps)))
                    a_start = max(0, int(round(s_sec * a_fps)))
                    a_end = min(total_a, int(round(e_sec * a_fps)))
                    chunks.append((v_start, v_end, a_start, a_end))
                except Exception as e:
                    print(f"[PJ二采分段] 解析自定义切点失败: '{p}', 错误: {e}")

        if not chunks:
            chunks = [(0, total_v, 0, total_a)]
        return chunks

    def execute(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        split_mode="按秒自动切分",
        chunk_seconds=8.0,
        overlap_seconds=2.0,
        num_segments=2,
        custom_cuts="0-8, 6-14, 12-20",
        align_51_frames=False,
        run_mode="全部精修(刷新全部缓存)",
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
        global _SECOND_PASS_CACHE

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

        noise_mask = latent_image.get("noise_mask", None)

        is_nested = hasattr(latent, "is_nested") and latent.is_nested
        if is_nested:
            streams = list(latent.unbind())
            full_video = streams[0]
            full_audio = streams[1] if len(streams) > 1 else None
        else:
            full_video = latent
            full_audio = None

        orig_audio_for_bypass = full_audio.clone() if full_audio is not None else None

        total_v = full_video.shape[2]
        total_a = full_audio.shape[-1] if full_audio is not None else 0

        chunks = self.plan_segments(
            total_v,
            total_a,
            split_mode,
            chunk_seconds,
            overlap_seconds,
            num_segments,
            custom_cuts,
            fps,
            align_51_frames=align_51_frames,
        )
        n_chunks = len(chunks)

        print("\n" + "=" * 65)
        print(f"[PJ二采分段采样器] 输入潜空间总计: {total_v} 视频步 (~{total_v / (fps / 3.4):.2f}s), 音频: {total_a} 步")
        print(f"[PJ二采分段采样器] 运行模式: {run_mode} | 计划切分: {n_chunks} 段")
        for i, (vs, ve, as_, ae) in enumerate(chunks):
            print(f"  - 分段 {i + 1}/{n_chunks}: 视频潜空间 [{vs}:{ve}], 音频 [{as_}:{ae}]")
        print("=" * 65)

        locks = [
            seg1_lock, seg2_lock, seg3_lock, seg4_lock,
            seg5_lock, seg6_lock, seg7_lock, seg8_lock
        ]

        # Generate global noise
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

        for idx, (v_start, v_end, a_start, a_end) in enumerate(chunks):
            cache_key = f"second_pass_seg_{idx}"
            seg_lock = locks[idx] if idx < len(locks) else False

            should_resample = True
            if run_mode == "全部精修(刷新全部缓存)":
                should_resample = True
            elif run_mode.startswith("仅精修第"):
                target_seg_num = int(run_mode.replace("仅精修第", "").replace("段", "").strip())
                should_resample = (idx + 1) == target_seg_num
            elif "自定义" in run_mode:
                try:
                    targets = [int(s.strip()) for s in str(custom_rerun_segments).split(",") if s.strip().isdigit()]
                    should_resample = (idx + 1) in targets
                except Exception:
                    should_resample = True
            elif run_mode == "智能缓存(仅重跑未锁定分段)":
                if seg_lock and cache_key in _SECOND_PASS_CACHE:
                    should_resample = False
                else:
                    should_resample = True

            # Cache hit check
            if not should_resample and cache_key in _SECOND_PASS_CACHE:
                print(f"\n[PJ二采分段采样器] ⚡ 第 {idx + 1}/{n_chunks} 段二采已锁定，直接从缓存加载 (耗时 0.00s)！")
                cached_chunk = _SECOND_PASS_CACHE[cache_key]
                sampled_chunks.append(cached_chunk)
                continue

            print(f"\n[PJ二采分段采样器] >>> 开始精修第 {idx + 1}/{n_chunks} 段 (视频 [{v_start}:{v_end}]) ...")

            chunk_v = full_video[:, :, v_start:v_end, :, :].clone()
            if is_nested and full_audio is not None and a_end > a_start:
                chunk_a = full_audio[:, :, :, a_start:a_end].clone()
                chunk_latent_samples = comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))
            else:
                chunk_latent_samples = chunk_v

            chunk_nv = gn_video[:, :, v_start:v_end, :, :].clone()
            if is_nested and gn_audio is not None and a_end > a_start:
                chunk_na = gn_audio[:, :, :, a_start:a_end].clone()
                chunk_noise = comfy.nested_tensor.NestedTensor((chunk_nv, chunk_na))
            else:
                chunk_noise = chunk_nv

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
            chunk_out_cpu = chunk_out.to("cpu")
            sampled_chunks.append(chunk_out_cpu)
            _SECOND_PASS_CACHE[cache_key] = chunk_out_cpu

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

            del chunk_v, chunk_latent_samples
            if "chunk_a" in locals() and chunk_a is not None:
                del chunk_a
            if "chunk_noise" in locals() and chunk_noise is not None:
                del chunk_noise
            comfy.model_management.soft_empty_cache()
            print(f"[PJ二采分段采样器] <<< 第 {idx + 1}/{n_chunks} 段精修完成，已存入缓存。")

        # Blending
        print(f"\n[PJ二采分段采样器] 正在执行视频重叠区无缝融合 (融合算法: {blend_mode}) ...")
        video_intervals = [(c[0], c[1]) for c in chunks]
        audio_intervals = [(c[2], c[3]) for c in chunks]

        target_ov_latents = max(1, int(round(overlap_seconds * (fps / 3.4)))) if overlap_seconds > 0 else None

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

        blended_video = self._blend_sequence(
            chunk_videos, video_intervals, total_v, norm_blend, target_ov_latents
        )

        if keep_original_audio and orig_audio_for_bypass is not None:
            print("[PJ二采分段采样器] 音频策略: 100% 保留一采原版连续完整音频 (零接缝、零相位抵消、零音质损耗)！")
            blended_audio = orig_audio_for_bypass
        elif chunk_audios and total_a > 0:
            target_ov_audio = max(1, int(round(overlap_seconds * 40.0))) if overlap_seconds > 0 else None
            blended_audio = self._blend_sequence(
                chunk_audios, audio_intervals, total_a, norm_blend, target_ov_audio, is_audio=True
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
            for i, x0_c in enumerate(x0_chunks):
                if x0_c is not None:
                    if hasattr(x0_c, "is_nested") and x0_c.is_nested:
                        valid_x0_videos.append(list(x0_c.unbind())[0])
                    else:
                        valid_x0_videos.append(x0_c)
                else:
                    valid_x0_videos.append(chunk_videos[i])
            blended_x0_video = self._blend_sequence(
                valid_x0_videos, video_intervals, total_v, norm_blend, target_ov_latents
            )
            if blended_audio is not None:
                result_denoised = comfy.nested_tensor.NestedTensor((blended_x0_video, blended_audio))
            else:
                result_denoised = blended_x0_video

        # Auto Unload Model
        if unload_model_after_sampling:
            print("[PJ二采分段采样器] 采样完成：正在将主模型卸载到 CPU，彻底释放显存给后续 VAE 解码...")
            try:
                if hasattr(guider, "model_patcher"):
                    guider.model_patcher.detach(unpatch_all=False)
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache(True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[PJ二采分段采样器] ✅ 主模型已成功卸载，GPU 显存已完全释放！后续 VAE 解码将拥有充足纯净显存！")
            except Exception as e:
                print(f"[PJ二采分段采样器] 卸载模型提示: {e}")

        print("[PJ二采分段采样器] 全部多段二采精修完成！准备输出结果。\n" + "=" * 65)

        out_denoised = {"samples": result_denoised} if result_denoised is not None else {"samples": result_samples}
        return ({"samples": result_samples}, out_denoised)

    def _blend_sequence(self, chunks, intervals, total_len, blend_mode, target_ov_latents=None, is_audio=False):
        n = len(chunks)
        if n == 1:
            return chunks[0]

        device = torch.device("cpu")
        sample_chunk = chunks[0]
        dtype = sample_chunk.dtype

        if is_audio:
            accum = torch.zeros(
                (sample_chunk.shape[0], sample_chunk.shape[1], sample_chunk.shape[2], total_len),
                device=device,
                dtype=dtype,
            )
            weight = torch.zeros((1, 1, 1, total_len), device=device, dtype=dtype)
        else:
            accum = torch.zeros(
                (sample_chunk.shape[0], sample_chunk.shape[1], total_len, sample_chunk.shape[3], sample_chunk.shape[4]),
                device=device,
                dtype=dtype,
            )
            weight = torch.zeros((1, 1, total_len, 1, 1), device=device, dtype=dtype)

        for i, (chunk, (start, end)) in enumerate(zip(chunks, intervals)):
            length = end - start
            if is_audio:
                w = torch.ones((1, 1, 1, length), device=device, dtype=dtype)
            else:
                w = torch.ones((1, 1, length, 1, 1), device=device, dtype=dtype)

            if i > 0 and blend_mode != "hard_cut":
                prev_start, prev_end = intervals[i - 1]
                ov = prev_end - start
                if ov > 0:
                    fade_len = min(ov, target_ov_latents) if target_ov_latents else ov
                    t = torch.linspace(0.0, 1.0, fade_len, device=device, dtype=dtype)
                    fade = 0.5 * (1.0 - torch.cos(math.pi * t)) if blend_mode == "cosine" else t
                    if is_audio:
                        w[:, :, :, :fade_len] = fade.view(1, 1, 1, -1)
                    else:
                        w[:, :, :fade_len, :, :] = fade.view(1, 1, -1, 1, 1)

            if i < n - 1 and blend_mode != "hard_cut":
                next_start, next_end = intervals[i + 1]
                ov = end - next_start
                if ov > 0:
                    fade_len = min(ov, target_ov_latents) if target_ov_latents else ov
                    t = torch.linspace(1.0, 0.0, fade_len, device=device, dtype=dtype)
                    fade = 0.5 * (1.0 - torch.cos(math.pi * t)) if blend_mode == "cosine" else t
                    if is_audio:
                        w[:, :, :, -fade_len:] = fade.view(1, 1, 1, -1)
                    else:
                        w[:, :, -fade_len:, :, :] = fade.view(1, 1, -1, 1, 1)

            if is_audio:
                accum[:, :, :, start:end] += chunk * w
                weight[:, :, :, start:end] += w
            else:
                accum[:, :, start:end, :, :] += chunk * w
                weight[:, :, start:end, :, :] += w

        weight = torch.clamp(weight, min=1e-6)
        blended = accum / weight
        return blended


# Alias class name
PJSegmentSampler = PJ_SegmentSampler
