import os
import sys
import copy
import math
import logging
import gc
import torch

import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.nested_tensor
import latent_preview

# Global in-memory cache for First Pass segment latents
_FIRST_PASS_CACHE = {}


def _slice_cond_refs(cond, v_start, v_end, a_start=None, a_end=None):
    """
    Slices the reference video / audio latents in conditioning to align
    precisely with the current time segment.
    """
    if not cond:
        return cond
    new_cond = []
    for item in cond:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            emb, extra = item[0], item[1]
            if isinstance(extra, dict) and "minimax_refs" in extra:
                new_extra = dict(extra)
                new_refs = []
                for ref in extra["minimax_refs"]:
                    if isinstance(ref, dict) and ref.get("kind") in ("video", "video_audio") and "latent" in ref:
                        ref_copy = dict(ref)
                        full_z = ref["latent"]
                        # Slicing temporal dimension: shape [B, C, T, H, W]
                        s = max(0, min(v_start, full_z.shape[2]))
                        e = max(s + 1, min(v_end, full_z.shape[2]))
                        sliced_z = full_z[:, :, s:e, :, :].clone()
                        ref_copy["latent"] = sliced_z
                        ref_copy["latent_t"] = sliced_z.shape[2]
                        if "audio_latent" in ref and ref["audio_latent"] is not None and a_start is not None and a_end is not None:
                            full_az = ref["audio_latent"]
                            as_ = max(0, min(a_start, full_az.shape[-1]))
                            ae_ = max(as_ + 1, min(a_end, full_az.shape[-1]))
                            ref_copy["audio_latent"] = full_az[..., as_:ae_].clone()
                            ref_copy["ref_audio_t"] = ae_ - as_
                        new_refs.append(ref_copy)
                    else:
                        new_refs.append(ref)
                new_extra["minimax_refs"] = new_refs
                new_cond.append([emb, new_extra])
                continue
        new_cond.append(item)
    return new_cond


class PJ_FirstPass_SegmentSampler:
    """
    PJ一采分段重抽采样器 (First-Pass Segment Sampler with Independent Re-roll & Cache)
    专为长视频一采（低清骨架生成）设计：
    1. 按照预设规则精准分段，支持参考视频时钟同步（每段只看对应时间段的原视频）；
    2. 支持各分段独立设置随机种子与【锁定缓存】；
    3. 哪段效果不好，仅单独重跑该段一采，好段0秒直接读缓存，大幅节省算力与时间；
    4. 重叠区全局噪声切片共享 + 余弦平滑无缝融合。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE", {"tooltip": "基础噪声源 (可连接 RandomNoise 节点)"}),
                "guider": ("GUIDER", {"tooltip": "引导器 (连接 BasicGuider，内含模型与提示词/参考条件)"}),
                "sampler": ("SAMPLER", {"tooltip": "采样算法 (推荐 er_sde 或 euler)"}),
                "sigmas": ("SIGMAS", {"tooltip": "调度器步长 (通常为一采全降噪，如 6 步)"}),
                "latent_image": ("LATENT", {"tooltip": "一采空潜空间 (来自 EmptyMiniMaxH3LatentAV)"}),
            },
            "optional": {
                "split_mode": (
                    ["按秒自动切分", "固定分段数", "自定义切点"],
                    {"default": "按秒自动切分", "tooltip": "分段规划策略"},
                ),
                "chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 2.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "自动模式下单段最大时长（秒），推荐 5.0 ~ 7.0 秒",
                    },
                ),
                "overlap_seconds": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.25,
                        "tooltip": "相邻分段重叠过渡时长（秒），推荐 1.0 ~ 2.0 秒",
                    },
                ),
                "num_segments": (
                    "INT",
                    {
                        "default": 3,
                        "min": 2,
                        "max": 10,
                        "step": 1,
                        "tooltip": "固定分段数模式下的切分数量",
                    },
                ),
                "custom_cuts": (
                    "STRING",
                    {
                        "default": "0-6, 4.5-10.5, 9-15",
                        "multiline": False,
                        "tooltip": "自定义切分秒数范围，逗号分隔，如：0-6, 4.5-10.5, 9-15",
                    },
                ),
                "align_51_frames": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "开启51帧吸附",
                        "label_off": "关闭",
                        "tooltip": "开启后将窗口长度吸附到51帧/85音频步网格，确保绝对RoPE相位一致",
                    },
                ),
                "sync_ref_video": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "开启(时序同步)",
                        "label_off": "关闭",
                        "tooltip": "【核心功能】自动将参考视频按分段时区精准裁切，保证各段角色动作与原视频对应秒数100%对齐！",
                    },
                ),
                "run_mode": (
                    [
                        "智能缓存(仅重跑未锁定分段)",
                        "全部重新采样(刷新全部缓存)",
                        "仅重抽第1段",
                        "仅重抽第2段",
                        "仅重抽第3段",
                        "仅重抽第4段",
                        "仅重抽第5段",
                        "仅重抽第6段",
                        "仅重抽第7段",
                        "仅重抽第8段",
                        "自定义指定分段(下方输入)",
                    ],
                    {
                        "default": "智能缓存(仅重跑未锁定分段)",
                        "tooltip": "【重抽控制台】哪段一采不满意直接选该段单独重抽，其余段0秒提取缓存，极速出片！",
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
                "base_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "全局基础随机种子 (若连接了 noise 输入端口，将优先以其种子为基准)",
                    },
                ),
                "seg1_seed_offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "第1段种子偏移 (实际种子 = 基础种子 + 偏移)"},
                ),
                "seg1_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾锁定后跳过计算，0秒读缓存"},
                ),
                "seg2_seed_offset": (
                    "INT",
                    {"default": 1, "min": 0, "max": 999999, "step": 1, "tooltip": "第2段种子偏移"},
                ),
                "seg2_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾锁定后跳过计算，0秒读缓存"},
                ),
                "seg3_seed_offset": (
                    "INT",
                    {"default": 2, "min": 0, "max": 999999, "step": 1, "tooltip": "第3段种子偏移"},
                ),
                "seg3_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾锁定后跳过计算，0秒读缓存"},
                ),
                "seg4_seed_offset": (
                    "INT",
                    {"default": 3, "min": 0, "max": 999999, "step": 1, "tooltip": "第4段种子偏移"},
                ),
                "seg4_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾锁定后跳过计算，0秒读缓存"},
                ),
                "seg5_seed_offset": (
                    "INT",
                    {"default": 4, "min": 0, "max": 999999, "step": 1, "tooltip": "第5段种子偏移"},
                ),
                "seg5_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾锁定后跳过计算，0秒读缓存"},
                ),
                "seg6_seed_offset": (
                    "INT",
                    {"default": 5, "min": 0, "max": 999999, "step": 1, "tooltip": "第6段种子偏移"},
                ),
                "seg6_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾锁定后跳过计算，0秒读缓存"},
                ),
                "seg7_seed_offset": (
                    "INT",
                    {"default": 6, "min": 0, "max": 999999, "step": 1, "tooltip": "第7段种子偏移"},
                ),
                "seg7_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾锁定后跳过计算，0秒读缓存"},
                ),
                "seg8_seed_offset": (
                    "INT",
                    {"default": 7, "min": 0, "max": 999999, "step": 1, "tooltip": "第8段种子偏移"},
                ),
                "seg8_lock": (
                    "BOOLEAN",
                    {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "打勾锁定后跳过计算，0秒读缓存"},
                ),
                "blend_mode": (
                    ["余弦平滑(推荐)", "线性过渡", "硬切"],
                    {"default": "余弦平滑(推荐)", "tooltip": "重叠过渡区无缝平滑算法"},
                ),
                "unload_model_after_sampling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "开启(释放显存)",
                        "label_off": "保留显存",
                        "tooltip": "若一采后直接运行3D超分，开启此项可清空GPU显存，给超分腾出充足空间",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "视频帧率，MiniMax H3 官方为 24.0"},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("完整潜空间", "分段潜空间列表")
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

        if split_mode in ("按秒自动切分", "auto_by_seconds"):
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

        elif split_mode in ("固定分段数", "fixed_segments"):
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

        elif split_mode in ("自定义切点", "custom", "custom_cuts"):
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
                    print(f"[PJ一采分段] 解析自定义切点失败: '{p}', 错误: {e}")

        # Fallback if no valid chunks planned
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
        chunk_seconds=6.0,
        overlap_seconds=1.5,
        num_segments=3,
        custom_cuts="0-6, 4.5-10.5, 9-15",
        align_51_frames=False,
        sync_ref_video=True,
        run_mode="智能缓存(仅重跑未锁定分段)",
        custom_rerun_segments="5",
        base_seed=0,
        seg1_seed_offset=0,
        seg1_lock=False,
        seg2_seed_offset=1,
        seg2_lock=False,
        seg3_seed_offset=2,
        seg3_lock=False,
        seg4_seed_offset=3,
        seg4_lock=False,
        seg5_seed_offset=4,
        seg5_lock=False,
        seg6_seed_offset=5,
        seg6_lock=False,
        seg7_seed_offset=6,
        seg7_lock=False,
        seg8_seed_offset=7,
        seg8_lock=False,
        blend_mode="余弦平滑(推荐)",
        unload_model_after_sampling=False,
        fps=24.0,
    ):
        global _FIRST_PASS_CACHE

        if isinstance(run_mode, bool) or str(run_mode).lower() in ("true", "false"):
            run_mode = "智能缓存(仅重跑未锁定分段)"
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

        total_v = full_video.shape[2]
        total_a = full_audio.shape[-1] if full_audio is not None else 0

        # Plan chunks
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
        print(f"[PJ一采分段采样器] 输入总时长: {total_v} 视频步 (~{total_v / (fps / 3.4):.2f}s), 音频: {total_a} 步")
        print(f"[PJ一采分段采样器] 运行模式: {run_mode} | 计划分段: {n_chunks} 段")
        for i, (vs, ve, as_, ae) in enumerate(chunks):
            print(f"  - 分段 {i + 1}/{n_chunks}: 视频潜空间 [{vs}:{ve}], 音频 [{as_}:{ae}]")
        print("=" * 65)

        # Segment control configuration
        locks = [
            seg1_lock, seg2_lock, seg3_lock, seg4_lock,
            seg5_lock, seg6_lock, seg7_lock, seg8_lock
        ]
        offsets = [
            seg1_seed_offset, seg2_seed_offset, seg3_seed_offset, seg4_seed_offset,
            seg5_seed_offset, seg6_seed_offset, seg7_seed_offset, seg8_seed_offset
        ]

        # Resolve effective base seed
        seed_val = getattr(noise, "seed", None)
        if seed_val is None or not isinstance(seed_val, int):
            seed_val = base_seed

        # Pre-generate global noise tensor
        global_noise = noise.generate_noise(latent)
        if is_nested:
            gn_streams = list(global_noise.unbind())
            gn_video = gn_streams[0]
            gn_audio = gn_streams[1] if len(gn_streams) > 1 else None
        else:
            gn_video = global_noise
            gn_audio = None

        sampled_chunks = []
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        for idx, (v_start, v_end, a_start, a_end) in enumerate(chunks):
            cache_key = f"first_pass_seg_{idx}"
            seg_lock = locks[idx] if idx < len(locks) else False
            seg_offset = offsets[idx] if idx < len(offsets) else idx
            seg_seed = (seed_val + seg_offset) & 0xFFFFFFFFFFFFFFFF

            # Determine whether to re-sample this segment
            should_resample = True
            if run_mode == "全部重新采样(刷新全部缓存)":
                should_resample = True
            elif run_mode.startswith("仅重抽第"):
                target_seg_num = int(run_mode.replace("仅重抽第", "").replace("段", "").strip())
                should_resample = (idx + 1) == target_seg_num
            elif "自定义" in run_mode:
                try:
                    targets = [int(s.strip()) for s in str(custom_rerun_segments).split(",") if s.strip().isdigit()]
                    should_resample = (idx + 1) in targets
                except Exception:
                    should_resample = True
            elif run_mode == "智能缓存(仅重跑未锁定分段)":
                if seg_lock and cache_key in _FIRST_PASS_CACHE:
                    should_resample = False
                else:
                    should_resample = True

            # If segment is cached and locked -> 0s instant load!
            if not should_resample and cache_key in _FIRST_PASS_CACHE:
                print(f"\n[PJ一采分段采样器] ⚡ 第 {idx + 1}/{n_chunks} 段已锁定，直接从缓存加载 (耗时 0.00s，不占 GPU 算力)！")
                cached_chunk = _FIRST_PASS_CACHE[cache_key]
                sampled_chunks.append(cached_chunk)
                continue

            # Need to sample this segment
            print(f"\n[PJ一采分段采样器] 🎲 >>> 开始采样第 {idx + 1}/{n_chunks} 段 (视频 [{v_start}:{v_end}], 独立种子: {seg_seed}) ...")

            # Slice latent
            chunk_v = full_video[:, :, v_start:v_end, :, :].clone()
            if is_nested and full_audio is not None and a_end > a_start:
                chunk_a = full_audio[:, :, :, a_start:a_end].clone()
                chunk_latent_samples = comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))
            else:
                chunk_latent_samples = chunk_v

            # Slice noise from global noise for shared temporal consistency
            chunk_nv = gn_video[:, :, v_start:v_end, :, :].clone()
            if is_nested and gn_audio is not None and a_end > a_start:
                chunk_na = gn_audio[:, :, :, a_start:a_end].clone()
                chunk_noise = comfy.nested_tensor.NestedTensor((chunk_nv, chunk_na))
            else:
                chunk_noise = chunk_nv

            # Slice reference video in conditioning if sync_ref_video is enabled
            chunk_guider = guider
            if sync_ref_video and hasattr(guider, "conds") and isinstance(guider.conds, dict):
                try:
                    chunk_guider = copy.copy(guider)
                    chunk_guider.conds = dict(guider.conds)
                    if "positive" in guider.conds:
                        chunk_guider.conds["positive"] = _slice_cond_refs(
                            guider.conds["positive"], v_start, v_end, a_start, a_end
                        )
                    if "negative" in guider.conds:
                        chunk_guider.conds["negative"] = _slice_cond_refs(
                            guider.conds["negative"], v_start, v_end, a_start, a_end
                        )
                except Exception as e:
                    print(f"[PJ一采分段采样器] 提示: 切片参考视频潜空间时跳过: {e}")
                    chunk_guider = guider

            chunk_x0_output = {}
            chunk_callback = latent_preview.prepare_callback(
                chunk_guider.model_patcher, sigmas.shape[-1] - 1, chunk_x0_output
            )

            # Sample the segment
            chunk_out = chunk_guider.sample(
                chunk_noise,
                chunk_latent_samples,
                sampler,
                sigmas,
                denoise_mask=None,
                callback=chunk_callback,
                disable_pbar=disable_pbar,
                seed=seg_seed,
            )

            # Save to CPU and update cache
            chunk_out_cpu = chunk_out.to("cpu")
            sampled_chunks.append(chunk_out_cpu)
            _FIRST_PASS_CACHE[cache_key] = chunk_out_cpu
            print(f"[PJ一采分段采样器] <<< 第 {idx + 1}/{n_chunks} 段采样完成，已存入分段缓存池。")

            # Clean temporary memory
            del chunk_v, chunk_latent_samples
            if "chunk_a" in locals() and chunk_a is not None:
                del chunk_a
            if "chunk_noise" in locals() and chunk_noise is not None:
                del chunk_noise
            comfy.model_management.soft_empty_cache()

        # Step 3: Seamless Overlap Blending
        print(f"\n[PJ一采分段采样器] 正在执行全序列重叠区平滑融合 (融合算法: {blend_mode}) ...")
        video_intervals = [(c[0], c[1]) for c in chunks]
        audio_intervals = [(c[2], c[3]) for c in chunks]

        target_ov_latents = max(1, int(round(overlap_seconds * (fps / 3.4)))) if overlap_seconds > 0 else None

        # Blend video latents
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

        # Perform blending
        blended_video = self._blend_sequence(
            chunk_videos, video_intervals, total_v, blend_mode, target_ov_latents
        )

        # Blend or assemble audio
        if chunk_audios and total_a > 0:
            target_ov_audio = max(1, int(round(overlap_seconds * 40.0))) if overlap_seconds > 0 else None
            blended_audio = self._blend_sequence(
                chunk_audios, audio_intervals, total_a, blend_mode, target_ov_audio, is_audio=True
            )
            result_samples = comfy.nested_tensor.NestedTensor((blended_video, blended_audio))
        else:
            result_samples = blended_video

        # Optional Model Unload
        if unload_model_after_sampling:
            print("[PJ一采分段采样器] 采样完成：正在将大模型卸载到 CPU 内存以释放显存...")
            try:
                if hasattr(guider, "model_patcher"):
                    guider.model_patcher.detach(unpatch_all=False)
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache(True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[PJ一采分段采样器] ✅ 一采主模型已成功卸载，显存已清空，后续 3D 超分将拥有极致显存空间！")
            except Exception as e:
                print(f"[PJ一采分段采样器] 卸载模型提示: {e}")

        print("[PJ一采分段采样器] 全部一采分段生成与融合完成！准备输出结果。\n" + "=" * 65)
        return ({"samples": result_samples}, {"samples": sampled_chunks[0]})

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

        # Compute weights with Cosine blending
        for i, (chunk, (start, end)) in enumerate(zip(chunks, intervals)):
            length = end - start
            if is_audio:
                w = torch.ones((1, 1, 1, length), device=device, dtype=dtype)
            else:
                w = torch.ones((1, 1, length, 1, 1), device=device, dtype=dtype)

            # Left overlap
            if i > 0 and blend_mode != "硬切":
                prev_start, prev_end = intervals[i - 1]
                ov = prev_end - start
                if ov > 0:
                    fade_len = min(ov, target_ov_latents) if target_ov_latents else ov
                    t = torch.linspace(0.0, 1.0, fade_len, device=device, dtype=dtype)
                    if blend_mode == "余弦平滑(推荐)":
                        fade = 0.5 * (1.0 - torch.cos(math.pi * t))
                    else:
                        fade = t
                    if is_audio:
                        w[:, :, :, :fade_len] = fade.view(1, 1, 1, -1)
                    else:
                        w[:, :, :fade_len, :, :] = fade.view(1, 1, -1, 1, 1)

            # Right overlap
            if i < n - 1 and blend_mode != "硬切":
                next_start, next_end = intervals[i + 1]
                ov = end - next_start
                if ov > 0:
                    fade_len = min(ov, target_ov_latents) if target_ov_latents else ov
                    t = torch.linspace(1.0, 0.0, fade_len, device=device, dtype=dtype)
                    if blend_mode == "余弦平滑(推荐)":
                        fade = 0.5 * (1.0 - torch.cos(math.pi * t))
                    else:
                        fade = t
                    if is_audio:
                        w[:, :, :, -fade_len:] = fade.view(1, 1, 1, -1)
                    else:
                        w[:, :, -fade_len:, :, :] = fade.view(1, 1, -1, 1, 1)

            # Accumulate
            if is_audio:
                accum[:, :, :, start:end] += chunk * w
                weight[:, :, :, start:end] += w
            else:
                accum[:, :, start:end, :, :] += chunk * w
                weight[:, :, start:end, :, :] += w

        weight = torch.clamp(weight, min=1e-6)
        blended = accum / weight
        return blended
