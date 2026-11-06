import torch

import comfy.utils
import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import latent_preview

from .segment_boundaries import constrain_boundaries, restore_preserved
from .segment_timing import (plan_segments, plan_relay_segments, resolve_plan, segment_seconds,
                            zero_overlap, delivery_chunks)
from .segment_relay import sample_relay
from .pj_h3_context import prepare_segment_guider
from .segment_local_repair import repair_segments, repair_sigmas, slice_latent
from .segment_revision import RevisionStore
from .segment_sampling import (SegmentCache, sampling_signature, fingerprint, generate_noise,
                               slice_mask, segment_guider, blend_sequence)


class PJ_FirstPass_SegmentSampler:
    """
    PJ一采分段重抽采样器 (First-Pass Segment Sampler with Independent Re-roll & Cache)
    专为长视频一采（低清骨架生成）设计：
    1. 按照预设规则分段，可按时间裁剪参考视频潜空间条件；
    2. 支持各分段独立设置随机种子与【锁定缓存】；
    3. 哪段效果不好，仅单独重跑该段一采，锁定且输入未变化的分段可从临时磁盘缓存复用；
    4. 相同种子共享绝对位置噪声，独立偏移重抽对应分段。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE", {"tooltip": "基础噪声源 (可连接 RandomNoise 节点)", "lazy": True}),
                "guider": ("GUIDER", {"tooltip": "引导器 (连接 BasicGuider，内含模型与提示词/参考条件)", "lazy": True}),
                "sampler": ("SAMPLER", {"tooltip": "采样算法 (推荐 er_sde 或 euler)", "lazy": True}),
                "sigmas": ("SIGMAS", {"tooltip": "调度器步长 (通常为一采全降噪，如 6 步)", "lazy": True}),
                "latent_image": ("LATENT", {"tooltip": "目标潜空间，可接H3空音视频潜空间或PJ编辑区域准备的输出", "lazy": True}),
            },
            "optional": {
                "segment_plan": (
                    "PJ_SEGMENT_PLAN",
                    {
                        "tooltip": "可选；连接时按规划分段，未连接时按输入完整长度采样",
                        "lazy": True,
                    },
                ),
                "sync_ref_video": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "开启(时序同步)",
                        "label_off": "关闭",
                        "tooltip": "按H3模型时间裁剪参考视频及其音轨；不保证生成动作逐帧复现参考",
                    },
                ),
                "run_mode": (
                    ["智能缓存(仅重跑未锁定分段)", "整段全量采样(一采不分段)", "全部重新采样(刷新全部缓存)"]
                    + [f"仅重抽第{i}段" for i in range(1, 21)]
                    + ["按缺陷秒数重抽 (下方输入秒数)", "自定义指定分段(下方输入)", "全部目标段"],
                    {
                        "default": "智能缓存(仅重跑未锁定分段)",
                        "tooltip": "【重抽控制台】哪段一采不满意直接选该段单独重抽；若选【整段全量采样(一采不分段)】则一采一口气整段生成！",
                    },
                ),
                "target_time_second": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "若选择【按缺陷秒数重抽】，直接输入画面异常的秒数（如12.5s），节点自动识别所属分段并执行重抽！",
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
                "seg1_seed_offset": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "第1段种子偏移 (实际种子 = 基础种子 + 偏移)"}),
                "seg1_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg2_seed_offset": ("INT", {"default": 1, "min": 0, "max": 999999, "step": 1, "tooltip": "第2段种子偏移"}),
                "seg2_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg3_seed_offset": ("INT", {"default": 2, "min": 0, "max": 999999, "step": 1, "tooltip": "第3段种子偏移"}),
                "seg3_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg4_seed_offset": ("INT", {"default": 3, "min": 0, "max": 999999, "step": 1, "tooltip": "第4段种子偏移"}),
                "seg4_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg5_seed_offset": ("INT", {"default": 4, "min": 0, "max": 999999, "step": 1, "tooltip": "第5段种子偏移"}),
                "seg5_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg6_seed_offset": ("INT", {"default": 5, "min": 0, "max": 999999, "step": 1, "tooltip": "第6段种子偏移"}),
                "seg6_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg7_seed_offset": ("INT", {"default": 6, "min": 0, "max": 999999, "step": 1, "tooltip": "第7段种子偏移"}),
                "seg7_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg8_seed_offset": ("INT", {"default": 7, "min": 0, "max": 999999, "step": 1, "tooltip": "第8段种子偏移"}),
                "seg8_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg9_seed_offset": ("INT", {"default": 8, "min": 0, "max": 999999, "step": 1, "tooltip": "第9段种子偏移"}),
                "seg9_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg10_seed_offset": ("INT", {"default": 9, "min": 0, "max": 999999, "step": 1, "tooltip": "第10段种子偏移"}),
                "seg10_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg11_seed_offset": ("INT", {"default": 10, "min": 0, "max": 999999, "step": 1, "tooltip": "第11段种子偏移"}),
                "seg11_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg12_seed_offset": ("INT", {"default": 11, "min": 0, "max": 999999, "step": 1, "tooltip": "第12段种子偏移"}),
                "seg12_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg13_seed_offset": ("INT", {"default": 12, "min": 0, "max": 999999, "step": 1, "tooltip": "第13段种子偏移"}),
                "seg13_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg14_seed_offset": ("INT", {"default": 13, "min": 0, "max": 999999, "step": 1, "tooltip": "第14段种子偏移"}),
                "seg14_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg15_seed_offset": ("INT", {"default": 14, "min": 0, "max": 999999, "step": 1, "tooltip": "第15段种子偏移"}),
                "seg15_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg16_seed_offset": ("INT", {"default": 15, "min": 0, "max": 999999, "step": 1, "tooltip": "第16段种子偏移"}),
                "seg16_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg17_seed_offset": ("INT", {"default": 16, "min": 0, "max": 999999, "step": 1, "tooltip": "第17段种子偏移"}),
                "seg17_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg18_seed_offset": ("INT", {"default": 17, "min": 0, "max": 999999, "step": 1, "tooltip": "第18段种子偏移"}),
                "seg18_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg19_seed_offset": ("INT", {"default": 18, "min": 0, "max": 999999, "step": 1, "tooltip": "第19段种子偏移"}),
                "seg19_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "seg20_seed_offset": ("INT", {"default": 19, "min": 0, "max": 999999, "step": 1, "tooltip": "第20段种子偏移"}),
                "seg20_lock": ("BOOLEAN", {"default": False, "label_on": "已锁定", "label_off": "未锁定", "tooltip": "输入及采样配置未变化时复用本节点临时磁盘缓存"}),
                "blend_mode": (
                    ["余弦平滑(推荐)", "线性过渡", "硬切"],
                    {"default": "余弦平滑(推荐)", "tooltip": "重叠区融合算法"},
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
                "boundary_constraints": ("BOOLEAN", {
                    "default": False,
                    "label_on": "开启(实验)", "label_off": "关闭(独立采样)",
                    "tooltip": "首次生成固定前段重叠内容；局部重做保护有效缓存中的前后边界。开启后不再混合两份重叠内容。切换后重建缓存；真实接缝画质需验证。",
                }),
                "generation_mode": (["分段编辑", "整段采样", "22帧接力"], {
                    "default": "分段编辑",
                    "tooltip": "接力按规划器目标段长生成，固定22帧重叠；不使用普通切点和重叠设置。未连接规划器时整段采样。",
                }),
                "relay_action": (["全部生成", "从指定段重做"], {"default": "全部生成"}),
                "relay_start_segment": ("INT", {
                    "default": 1, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "重做该段及之后所有段；前段缓存失效或缺失时从更早处开始。修改对应种子偏移可得到不同结果。",
                }),
                "repair_action": (["原采样方式", "载入回修基准", "仅重绘目标段", "复用一采结果"], {"default": "原采样方式"}),
                "repair_denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repair_scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "repair_steps": ("INT", {"default": 6, "min": 1, "max": 10000}),
                "repair_boundary_tokens": ("INT", {"default": 2, "min": 0, "max": 1000,
                    "tooltip": "局部重绘两端保留的视频潜步数；不是解码后的画面帧数"}),
                "repair_request_id": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "相同请求复用已完成结果；增加此值可用相同种子再重绘一次"}),
                "repair_baseline": ("LATENT", {"tooltip": "可选：接官方采样器等已有完整结果，作为首次回修基准", "lazy": True}),
                "h3_context": ("PJ_H3_CONTEXT", {"tooltip": "可选：按段重建 H3 提示词、参考素材与运动引导", "lazy": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("完整潜空间", "首段潜空间")
    FUNCTION = "execute"
    CATEGORY = "PJ工具箱/分段采样"

    def __init__(self):
        self._cache = SegmentCache()
        self._revisions = RevisionStore()
        self._baseline_key = None

    def check_lazy_status(self, repair_action="原采样方式", **kwargs):
        if repair_action == "原采样方式":
            required = ["noise", "guider", "sampler", "sigmas", "latent_image", "segment_plan", "h3_context"]
        elif repair_action == "载入回修基准":
            required = ["repair_baseline"]
        else:
            required = ["noise", "guider", "sampler", "segment_plan", "h3_context"] if repair_action == "仅重绘目标段" and kwargs.get("repair_denoise", 0.4) > 0 else []
            if not self._revisions.available():
                required.append("repair_baseline")
        return [name for name in required if name in kwargs and kwargs[name] is None]

    def _commit_generation(self, outputs, fps):
        samples = outputs[0]["samples"]
        streams = samples.unbind() if samples.is_nested else [samples]
        committed = self._revisions.commit(outputs[0], outputs[1], changes=[(0, streams[0].shape[2])],
                                           audio_changes=[(0, streams[1].shape[-1])] if len(streams) > 1 else [],
                                           fps=fps, reset=True)
        self._baseline_key = None
        return committed

    def _repair(self, noise, guider, sampler, latent_image, segment_plan, run_mode,
                target_time_second, custom_rerun_segments, base_seed, generation_mode,
                sync_ref_video, repair_action, repair_denoise, repair_scheduler, repair_steps,
                repair_boundary_tokens, repair_request_id, repair_baseline, fps, kwargs, h3_context):
        saved = self._revisions.load()
        if repair_action == "载入回修基准" and repair_baseline is None:
            raise ValueError("载入回修基准需要连接已有完整潜空间")
        if repair_baseline is not None and (saved is None or repair_action == "载入回修基准"):
            baseline_key = fingerprint(repair_baseline)
            saved = self._commit_generation((repair_baseline, None), fps)
            self._baseline_key = baseline_key
        if saved is None:
            raise ValueError("没有已成功的一采结果；请先完成一次采样，或连接已有完整潜空间到回修基准")
        baseline = saved[0]
        if repair_action in ("载入回修基准", "复用一采结果") or repair_denoise <= 0:
            return baseline, saved[1] if saved[1] is not None else dict(baseline)
        samples = baseline["samples"]
        streams = samples.unbind() if samples.is_nested else [samples]
        if generation_mode == "22帧接力" and segment_plan is not None:
            chunks = plan_relay_segments(streams[0].shape[2], streams[1].shape[-1] if len(streams) > 1 else 0,
                                         segment_plan.get("chunk_seconds", 6.0))
            deliveries = chunks
        else:
            chunks, fps = resolve_plan(samples, segment_plan, "按秒自动切分", 6.0, 1.5,
                                       3, "0-6,4.5-10.5,9-15", fps, False,
                                       full_length=segment_plan is None)
            deliveries = delivery_chunks(chunks, segment_plan)
        if "缺陷秒数" in str(run_mode):
            targets = [i for i, bounds in enumerate(deliveries)
                       if segment_seconds(*bounds[:2], fps)[0] <= target_time_second < segment_seconds(*bounds[:2], fps)[1]][:1]
        elif str(run_mode).startswith("仅重抽第"):
            targets = [int(run_mode.replace("仅重抽第", "").replace("段", "").strip()) - 1]
        elif "自定义" in str(run_mode):
            targets = [int(value.strip()) - 1 for value in str(custom_rerun_segments).split(",") if value.strip()]
        else:
            targets = list(range(len(chunks)))
        seed = getattr(noise, "seed", base_seed)
        if not isinstance(seed, int):
            seed = base_seed
        offsets = [int(kwargs.get(f"seg{i}_seed_offset", i - 1)) for i in range(1, 21)]
        sigmas = repair_sigmas(guider, repair_scheduler, repair_steps, repair_denoise)
        request_key = fingerprint((sampling_signature(guider, sampler, sigmas), targets, chunks, deliveries, seed,
                                   offsets, repair_boundary_tokens, repair_request_id, sync_ref_video,
                                   self._baseline_key, type(noise), vars(noise) if noise is not None else None, h3_context))
        if self._revisions.check_request(request_key):
            return baseline, slice_latent(baseline, chunks[0])
        output, _, changes = repair_segments(noise, guider, sampler, sigmas, baseline, chunks,
                                                targets, seed, offsets, repair_boundary_tokens,
                                                sync_ref_video, h3_context=h3_context, delivery_bounds=deliveries)
        return self._revisions.commit(output, slice_latent(output, chunks[0]), changes=changes,
                                      fps=fps, request_key=request_key)

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
        chunk_seconds=6.0,
        overlap_seconds=1.5,
        num_segments=3,
        custom_cuts="0-6, 4.5-10.5, 9-15",
        align_51_frames=False,
        sync_ref_video=True,
        run_mode="智能缓存(仅重跑未锁定分段)",
        target_time_second=0.0,
        custom_rerun_segments="5",
        base_seed=0,
        blend_mode="余弦平滑(推荐)",
        unload_model_after_sampling=False,
        fps=24.0,
        boundary_constraints=False,
        generation_mode="分段编辑",
        relay_action="全部生成",
        relay_start_segment=1,
        repair_action="原采样方式",
        repair_denoise=0.4,
        repair_scheduler="simple",
        repair_steps=6,
        repair_boundary_tokens=2,
        repair_request_id=0,
        repair_baseline=None,
        h3_context=None,
        **kwargs,
    ):
        if repair_action != "原采样方式":
            result = self._repair(noise, guider, sampler, latent_image, segment_plan, run_mode,
                                  target_time_second, custom_rerun_segments, base_seed, generation_mode,
                                  sync_ref_video, repair_action, repair_denoise, repair_scheduler, repair_steps,
                                  repair_boundary_tokens, repair_request_id, repair_baseline, fps, kwargs, h3_context)
            if unload_model_after_sampling:
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
            return result
        if generation_mode == "整段采样" or segment_plan is None:
            run_mode = "整段全量采样(一采不分段)"

        if isinstance(run_mode, bool) or str(run_mode).lower() in ("true", "false"):
            run_mode = "智能缓存(仅重跑未锁定分段)"
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

        if generation_mode == "22帧接力" and segment_plan is not None:
            if (not is_nested or len(streams) != 2 or full_video.ndim != 5 or full_video.shape[1] != 24
                    or total_v < 2 or total_v % 5 != 2 or full_audio.ndim != 4 or full_audio.shape[1:3] != (32, 2)):
                raise ValueError("22帧接力需要H3原生音视频潜空间，视频长度为5n+2潜步")
            chunks = plan_relay_segments(total_v, total_a, segment_plan.get("chunk_seconds", chunk_seconds))
            seed = getattr(noise, "seed", base_seed)
            if not isinstance(seed, int):
                seed = base_seed
            offsets = [int(kwargs.get(f"seg{i}_seed_offset", i - 1)) for i in range(1, 21)]
            start_segment = max(1, int(relay_start_segment)) if relay_action == "从指定段重做" else 1
            result = sample_relay(noise, guider, sampler, sigmas, latent_image, chunks, self._cache,
                                  seed, offsets, start_segment, sync_ref_video, h3_context=h3_context)
            if unload_model_after_sampling:
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
            return self._commit_generation(result, fps)

        chunks, fps = resolve_plan(
            latent, segment_plan, split_mode, chunk_seconds, overlap_seconds,
            num_segments, custom_cuts, fps, align_51_frames,
            full_length=segment_plan is None or generation_mode == "整段采样" or "不分段" in str(run_mode) or "全量采样" in str(run_mode),
        )
        n_chunks = len(chunks)
        no_overlap = zero_overlap(segment_plan)
        deliveries = delivery_chunks(chunks, segment_plan)
        constrain = boundary_constraints and n_chunks > 1 and not no_overlap
        self._cache.retain(n_chunks)
        context_signature = sampling_signature(guider, sampler, sigmas)
        if h3_context is not None:
            context_signature = fingerprint((context_signature, h3_context))
        if no_overlap:
            context_signature = fingerprint((context_signature, "delivery-v1", deliveries))
        if constrain:
            context_signature = fingerprint((context_signature, "boundaries-v2", chunks, latent_image))

        print("\n" + "=" * 65)
        print(f"[PJ一采分段采样器] 输入总时长: {total_v} 视频步 (~{segment_seconds(0, total_v, fps)[1]:.2f}s), 音频: {total_a} 步")
        print(f"[PJ一采分段采样器] 运行模式: {run_mode} | 计划分段: {n_chunks} 段")
        for i, (vs, ve, as_, ae) in enumerate(chunks):
            print(f"  - 分段 {i + 1}/{n_chunks}: 视频潜空间 [{vs}:{ve}], 音频 [{as_}:{ae}]")
        print("=" * 65)

        # Segment control configuration (支持到 20 段)
        locks = [bool(kwargs.get(f"seg{i}_lock", False)) for i in range(1, 21)]
        offsets = [int(kwargs.get(f"seg{i}_seed_offset", i - 1)) for i in range(1, 21)]

        # Resolve effective base seed
        seed_val = getattr(noise, "seed", None)
        if seed_val is None or not isinstance(seed_val, int):
            seed_val = base_seed

        # 计算目标缺陷秒数所属的分段编号 (如果处于缺陷秒数重抽模式)
        target_defect_seg = None
        if "缺陷秒数" in str(run_mode):
            for seg_i, (vs, ve, as_, ae) in enumerate(deliveries):
                s_sec, e_sec = segment_seconds(vs, ve, fps)
                if s_sec <= target_time_second < e_sec:
                    target_defect_seg = seg_i + 1
                    break
            if target_defect_seg is not None:
                print(f"[PJ一采分段采样器] 🎯 目标秒数 {target_time_second:.2f}s 自动匹配到第 {target_defect_seg} 段，将仅对该段重抽！")
            else:
                target_defect_seg = 1
                print(f"[PJ一采分段采样器] ⚠️ 目标秒数 {target_time_second:.2f}s 超出范围，默认重抽第 1 段")

        signatures = []
        retained = []
        for idx, bounds in enumerate(chunks):
            vs, ve, a_s, ae = bounds
            inputs = [full_video[:, :, vs:ve]]
            if full_audio is not None:
                inputs.append(full_audio[..., a_s:ae])
            signature = fingerprint((context_signature, bounds, inputs,
                                     slice_mask(latent_image.get("noise_mask"), bounds),
                                     latent_image.get("batch_index"), sync_ref_video, type(noise),
                                     {k: v for k, v in vars(noise).items() if k != "seed"} if noise is not None else None))
            signatures.append(signature)
            should_resample = True
            if "缺陷秒数" in str(run_mode):
                should_resample = idx + 1 == target_defect_seg
            elif run_mode.startswith("仅重抽第"):
                should_resample = idx + 1 == int(run_mode.replace("仅重抽第", "").replace("段", "").strip())
            elif "自定义" in run_mode:
                targets = [int(s.strip()) for s in str(custom_rerun_segments).split(",") if s.strip()]
                should_resample = idx + 1 in targets
            elif run_mode == "智能缓存(仅重跑未锁定分段)":
                should_resample = not (locks[idx] if idx < len(locks) else False)
            cached = self._cache.load(idx, signature) if not should_resample else None
            retained.append(cached[0] if cached is not None else None)

        if h3_context is not None and h3_context["motion_context"] and not no_overlap:
            for idx in range(1, n_chunks):
                start, end = chunks[idx][:2]
                previous_start, previous_end = chunks[idx - 1][:2]
                overlap = min(end, previous_end) - start
                if (previous_start <= start < previous_end and start % 5 == 0 and overlap % 5 == 2
                        and retained[idx - 1] is None):
                    retained[idx] = None
            for idx, value in enumerate(retained):
                if value is None:
                    self._cache.entries.pop(idx, None)

        # Invalidate before work: an interrupted boundary rebuild must not expose a mixed generation.
        rebuilding = constrain and any(value is None for value in retained)
        if rebuilding:
            self._cache.entries.clear()
        sampled_chunks = []
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        if constrain:
            print("[PJ一采] 接缝约束：固定已有重叠内容；缓存缺失的分段将重新生成")
        for idx, (v_start, v_end, a_start, a_end) in enumerate(chunks):
            if retained[idx] is not None:
                sampled_chunks.append(retained[idx])
                print(f"[PJ分段采样] 第 {idx + 1}/{n_chunks} 段复用缓存")
                continue
            chunk_v = full_video[:, :, v_start:v_end].clone()
            if is_nested and full_audio is not None:
                chunk_a = full_audio[..., a_start:a_end].clone()
                chunk_latent_samples = comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))
            else:
                chunk_latent_samples = chunk_v
            bounds = (v_start, v_end, a_start, a_end)
            chunk_mask = slice_mask(latent_image.get("noise_mask"), bounds)
            if constrain:
                sources = [(chunks[i], value) for i, value in enumerate(retained) if value is not None]
                sources += [(chunks[i], value) for i, value in enumerate(sampled_chunks) if retained[i] is None]
                chunk_latent_samples, chunk_mask = constrain_boundaries(chunk_latent_samples, chunk_mask, bounds, sources)
                del sources
            seg_offset = offsets[idx] if idx < len(offsets) else idx
            seg_seed = (seed_val + seg_offset) & 0xFFFFFFFFFFFFFFFF

            # Need to sample this segment
            print(f"\n[PJ一采分段采样器] 🎲 >>> 开始采样第 {idx + 1}/{n_chunks} 段 (视频 [{v_start}:{v_end}], 独立种子: {seg_seed}) ...")

            # Generate from the full shape so equal seeds share noise at equal absolute positions.
            segment_noise = generate_noise(noise, latent_image, seg_seed)
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
            if h3_context is None:
                chunk_guider = segment_guider(guider, bounds, sync_ref_video)
            else:
                previous_tail = None
                if sampled_chunks and not no_overlap:
                    previous_start, previous_end = chunks[idx - 1][:2]
                    overlap = min(v_end, previous_end) - v_start
                    if previous_start <= v_start < previous_end and v_start % 5 == 0 and overlap % 5 == 2:
                        previous_video = sampled_chunks[-1].unbind()[0] if is_nested else sampled_chunks[-1]
                        offset = v_start - previous_start
                        previous_tail = previous_video[:, :, offset:offset + overlap]
                chunk_guider = prepare_segment_guider(guider, h3_context, bounds, chunk_v.shape,
                                                       previous_tail=previous_tail, segment_index=idx)

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
                seed=seg_seed,
            )

            # Save to CPU and update cache
            chunk_out_cpu = chunk_out.to("cpu")
            restore_preserved(chunk_out_cpu, chunk_latent_samples, chunk_mask)
            sampled_chunks.append(chunk_out_cpu)
            if not constrain:
                self._cache.save(idx, signatures[idx], chunk_out_cpu)
            print(f"[PJ一采分段采样器] <<< 第 {idx + 1}/{n_chunks} 段采样完成。")

            # Clean temporary memory
            del chunk_v, chunk_latent_samples, chunk_out, chunk_x0_output, chunk_guider
            if "chunk_a" in locals() and chunk_a is not None:
                del chunk_a
            if "chunk_noise" in locals() and chunk_noise is not None:
                del chunk_noise
            comfy.model_management.soft_empty_cache()

        if constrain:
            blend_mode = "保留边界"
        # Assemble on the original timeline.
        print(f"\n[PJ一采分段采样器] 正在执行全序列重叠区融合 (融合算法: {blend_mode}) ...")
        video_intervals = [(c[0], c[1]) for c in chunks]
        audio_intervals = [(c[2], c[3]) for c in chunks]

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
        blended_video = blend_sequence(
            chunk_videos, video_intervals, full_video, blend_mode,
            delivery_intervals=[bounds[:2] for bounds in deliveries],
        )

        # Blend or assemble audio
        if chunk_audios and total_a > 0:
            blended_audio = blend_sequence(
                chunk_audios, audio_intervals, full_audio, blend_mode, is_audio=True,
                delivery_intervals=[bounds[2:] for bounds in deliveries],
            )
            result_samples = comfy.nested_tensor.NestedTensor((blended_video, blended_audio))
        else:
            result_samples = blended_video

        restore_preserved(result_samples, latent, latent_image.get("noise_mask"))

        if rebuilding:
            for idx, samples in enumerate(sampled_chunks):
                self._cache.save(idx, signatures[idx], samples)

        if unload_model_after_sampling:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            print("[PJ分段采样] 已请求卸载模型；解码峰值仍取决于分辨率及 VAE 配置")

        print("[PJ一采分段采样器] 全部一采分段生成与融合完成！准备输出结果。\n" + "=" * 65)
        output = dict(latent_image, samples=result_samples)
        return self._commit_generation((output, {"samples": sampled_chunks[0]}), fps)
