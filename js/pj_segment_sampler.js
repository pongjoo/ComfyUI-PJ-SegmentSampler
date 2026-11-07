import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HIDDEN_TAG = "pj_hidden";

// ==========================================
// 1. 中文标签与映射字典 (完整汉化核心)
// ==========================================
const WIDGET_LABELS = {
    // 规划模式相关
    "split_mode": "切分规划模式",
    "chunk_seconds": "目标段长(参考秒数)",
    "overlap_seconds": "目标重叠(参考秒数)",
    "num_segments": "固定分段数量",
    "custom_cuts": "自定义切点范围",
    "fps": "视频帧率",
    "duration_seconds": "视频总时长(秒)",

    // 一采专有功能
    "sync_ref_video": "按分段裁剪参考条件",
    "boundary_constraints": "接缝约束(实验)",
    "base_seed": "基础随机种子",
    "generation_mode": "生成方式",
    "relay_action": "本次处理",
    "relay_start_segment": "从第几段重做",
    "prompt": "全局提示词",
    "segment_prompts": "各段补充提示词",
    "reference_mode": "参考用途",
    "motion_context": "前段运动引导",

    // 运行重抽与缓存锁
    "run_mode": "本次处理范围",
    "target_time_second": "缺陷定位秒数",
    "custom_rerun_segments": "目标段号(逗号分隔)",

    // 算法与系统控制
    "blend_mode": "重叠区过渡方式",
    "keep_original_audio": "保留一采原音频",
    "unload_model_after_sampling": "完成后卸载模型"
};

// 动态填充 1~20 段的标签
for (let i = 1; i <= 20; i++) {
    WIDGET_LABELS[`seg${i}_lock`] = `锁定第${i}段`;
    WIDGET_LABELS[`seg${i}_seed_offset`] = `第${i}段种子偏移`;
}

const INPUT_LABELS = {
    "latent": "输入潜空间(可选)",
    "images": "视频图像(可选)",
    "audio": "音频原声(可选)",
    "noise": "随机噪声",
    "guider": "引导器",
    "sampler": "采样算法",
    "sigmas": "调度步长",
    "latent_image": "输入潜空间",
    "av_latent": "H3目标音视频潜空间",
    "source_latent": "原视频编码潜空间",
    "edit_mask": "空间编辑遮罩(白改黑留)",
    "segment_plan": "分段规划总线",
    "source_result": "放大前一采结果(版本)",
    "repair_baseline": "已有一采结果",
    "h3_context": "H3分段上下文",
    "clip": "文本编码器",
    "vae": "视频VAE",
    "reference_images": "图片参考",
    "reference_video": "视频参考",
    "reference_video_audio": "视频配套音轨",
    "reference_audio": "独立音频参考",
    "audio_vae": "音频VAE"
};

const OUTPUT_LABELS = {
    "output": "完整潜空间",
    "denoised_output": "去噪潜空间",
    "分段规划(PLAN)": "分段规划总线",
    "总分段数": "总分段数",
    "分段信息明细": "分段明细文本",
    "图像透传": "图像透传",
    "音频透传": "音频透传"
};

// ==========================================
// 2. 控件显隐与动态尺寸辅助函数
// ==========================================
function toggleWidget(node, widget, show) {
    if (!widget) return;
    const slotIndex = node.inputs?.findIndex(input => input.widget?.name === widget.name) ?? -1;
    const input = node.inputs?.[slotIndex];
    const connected = slotIndex >= 0 && (node.isInputConnected?.(slotIndex) ?? input.link != null);
    show = show || connected;
    // Widget sockets are drawn separately by current LiteGraph, even for hidden rows.
    if (input && typeof input.draw === "function" && !input._pjVisibilityDraw) {
        const draw = input.draw;
        input.draw = function () {
            const current = node.widgets?.find(w => w.name === this.widget?.name);
            if (current?.type !== HIDDEN_TAG) return draw.apply(this, arguments);
        };
        Object.defineProperty(input, "_pjVisibilityDraw", { value: true });
    }
    if (!widget._origProps) {
        widget._origProps = {
            origType: widget.type,
            origComputeSize: widget.computeSize
        };
    }
    if (show) {
        widget.type = widget._origProps.origType;
        widget.computeSize = widget._origProps.origComputeSize;
    } else {
        widget.type = HIDDEN_TAG;
        widget.computeSize = () => [0, -4];
    }
    widget.hidden = !show;
}

function autoResizeNode(node) {
    [20, 100, 250].forEach(delay => {
        setTimeout(() => {
            if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
                for (const widget of node.widgets || []) {
                    if (widget.pjContent?.offsetHeight > 0) {
                        widget.pjHeight = Math.ceil(widget.pjContent.offsetHeight);
                    }
                }
                const sz = node.computeSize();
                const targetW = Math.max(node.size[0] || 0, 420);
                node.setSize([targetW, sz[1] + 20]);
                if (node.graph && typeof node.graph.setDirty === "function") {
                    node.graph.setDirty(true, true);
                }
            }
        }, delay);
    });
}

// ==========================================
// 3. 通用数据清洗与基础汉化
// ==========================================
function localizeAndSanitizeCommon(node) {
    if (!node) return;

    if (node.inputs) {
        for (const input of node.inputs) {
            if (INPUT_LABELS[input.name]) {
                input.label = INPUT_LABELS[input.name];
            }
        }
    }

    if (node.outputs) {
        for (const output of node.outputs) {
            if (OUTPUT_LABELS[output.name]) {
                output.label = OUTPUT_LABELS[output.name];
            }
        }
    }

    if (node.widgets) {
        for (const w of node.widgets) {
            if (WIDGET_LABELS[w.name]) {
                w.label = WIDGET_LABELS[w.name];
            }

            if (w.name === "split_mode") {
                if (w.value === "auto_by_seconds") w.value = "按秒自动切分";
                else if (w.value === "fixed_segments") w.value = "固定分段数";
                else if (w.value === "custom" || w.value === "custom_cuts") w.value = "自定义切点";
                if (w.options) {
                    w.options.values = ["按秒自动切分", "固定分段数", "自定义切点"];
                }
            }

            if (w.name === "blend_mode") {
                if (w.value === "cosine") w.value = "余弦平滑(推荐)";
                else if (w.value === "linear") w.value = "线性过渡";
                else if (w.value === "hard" || w.value === "hard_cut") w.value = "硬切";
                else if (typeof w.value === "boolean" || !["余弦平滑(推荐)", "线性过渡", "硬切"].includes(w.value)) {
                    w.value = "余弦平滑(推荐)";
                }
                if (w.options) {
                    w.options.values = ["余弦平滑(推荐)", "线性过渡", "硬切"];
                }
            }

            if (w.name === "unload_model_after_sampling") {
                w.options = w.options || {};
                w.options.on = "开启(释放显存)";
                w.options.off = "保留显存";
                w.label_on = "开启(释放显存)";
                w.label_off = "保留显存";
                if (typeof w.value !== "boolean") {
                    w.value = true;
                }
            } else if (w.name === "keep_original_audio") {
                w.options = w.options || {};
                w.options.on = w.label_on = "保留输入音频";
                w.options.off = w.label_off = "使用二采音频";
                w.options.tooltip = "保留输入音频潜空间；不保证精修画面与原音频的口型一致。";
                if (typeof w.value !== "boolean") {
                    w.value = true;
                }
            } else if (w.name === "sync_ref_video") {
                w.options = w.options || {};
                w.options.on = "开启(按时间裁剪)";
                w.options.off = "关闭";
                w.label_on = "开启(按时间裁剪)";
                w.label_off = "关闭";
                w.options.tooltip = "按当前分段时间裁剪参考视频及音轨条件；不会把完整输出裁成独立片段，也不保证逐帧跟随参考。";
                if (typeof w.value !== "boolean") {
                    w.value = true;
                }
            } else if (w.name && w.name.endsWith("_lock")) {
                w.options = w.options || {};
                w.options.on = "已锁定";
                w.options.off = "未锁定";
                w.label_on = "已锁定";
                w.label_off = "未锁定";
                w.options.tooltip = "仅智能缓存模式使用：有有效缓存才可复用；输入、条件或采样配置变化后仍需重算。";
                // 彻底防御：若历史工作流因索引错位灌入了数字(如3)或字符串，强制清洗为 false (未锁定)
                if (typeof w.value !== "boolean") {
                    w.value = false;
                }
            }
        }
    }
}

// 溯源查找上游【分段规划器】节点（智能穿透 Reroute 路由节点）
function getUpstreamPlannerNode(node) {
    return getUpstreamInputNode(node, "segment_plan");
}

function getUpstreamInputNode(node, inputName) {
    if (!node.inputs || !app.graph) return null;
    const planInput = node.inputs.find(i => i.name === inputName || (inputName === "segment_plan" && (i.label === "分段规划总线" || i.type === "PJ_SEGMENT_PLAN")));
    if (!planInput || planInput.link == null) return null;

    let link = app.graph.links[planInput.link];
    let visited = new Set();
    while (link && link.origin_id != null) {
        if (visited.has(link.origin_id)) break;
        visited.add(link.origin_id);
        const origin = app.graph.getNodeById(link.origin_id);
        if (!origin) break;
        if (origin.type === "Reroute" || origin.type === "ReroutePrimitive") {
            const rInput = origin.inputs?.[0];
            if (rInput && rInput.link != null) {
                link = app.graph.links[rInput.link];
            } else {
                break;
            }
        } else {
            return origin;
        }
    }
    return null;
}

function hasSegmentPlan(node) {
    return !!node.inputs?.some(input =>
        (input.name === "segment_plan" || input.type === "PJ_SEGMENT_PLAN") && input.link != null);
}

function samplingSourceKey(node) {
    return JSON.stringify(["segment_plan", "latent_image", "repair_baseline", "source_result"].map(name => {
        const input = node.inputs?.find(value => value.name === name);
        let link = app.graph?.links[input?.link];
        const path = [input?.link ?? null];
        const visited = new Set();
        while (link && !visited.has(link.origin_id)) {
            visited.add(link.origin_id);
            path.push([link.origin_id, link.origin_slot]);
            const origin = app.graph.getNodeById(link.origin_id);
            if (origin?.type !== "Reroute" && origin?.type !== "ReroutePrimitive") break;
            link = app.graph.links[origin.inputs?.[0]?.link];
        }
        return path;
    }));
}

function currentSamplingSummary(node, planner) {
    const summary = node._pjSamplingSummary;
    if (!summary) return null;
    if (summary.sourceKey !== samplingSourceKey(node)) {
        delete node._pjSamplingSummary;
        return null;
    }
    const generation = node.widgets?.find(w => w.name === "generation_mode")?.value;
    const local = node.widgets?.find(w => w.name === "repair_action")?.value === "仅重绘目标段";
    const mode = !hasSegmentPlan(node) || (generation === "整段采样" && !local) ? "full" :
        generation === "22帧接力" ? "relay" : "segments";
    if (summary.mode !== mode || !["split_mode", "chunk_seconds", "overlap_seconds", "num_segments", "custom_cuts"]
        .every(name => summary.plan?.[name] === planner?.widgets?.find(w => w.name === name)?.value)) {
        summary.n_chunks = null;
    }
    return summary;
}

// 获取上游规划器的分段数；没有规划时只处理完整输入。
function resolveTargetSegmentCount(node) {
    if (!hasSegmentPlan(node)) return 1;
    // 1. 优先实时读取上游 PJ_SegmentPlanner 最新控件值
    const originNode = getUpstreamPlannerNode(node);
    const actual = currentSamplingSummary(node, originNode);
    if (actual?.n_chunks) return actual.n_chunks;
    if (originNode) {
        if (node.widgets?.find(w => w.name === "generation_mode")?.value === "22帧接力") {
            const preview = originNode.properties?.pj_preview_cache;
            if (!actual && preview?.relay_n_chunks) return preview.relay_n_chunks;
            const frames = preview?.total_frames || 345;
            const total = actual?.total_video_tokens ?? Math.max(2, Math.floor((frames - 5) / 17) * 5 + 2);
            const seconds = Number(originNode.widgets?.find(w => w.name === "chunk_seconds")?.value ?? 6);
            const window = Math.max(2, Math.floor((seconds * 24 - 5) / 17 + 0.5)) * 5 + 2;
            return 1 + Math.max(0, Math.ceil((total - window) / (window - 7)));
        }
        // 次级缓存读取：上游规划器已执行或有预览缓存时直接获取
        if (!actual && originNode.pj_cached_n_chunks) {
            return Math.min(20, Math.max(1, originNode.pj_cached_n_chunks));
        }
        if (!actual && originNode.properties?.["pj_preview_cache"]?.n_chunks) {
            return Math.min(20, Math.max(1, originNode.properties["pj_preview_cache"].n_chunks));
        }

        const mode = originNode.widgets?.find(w => w.name === "split_mode")?.value;
        if (mode === "固定分段数" || mode === "fixed_segments") {
            const upNumSeg = originNode.widgets?.find(w => w.name === "num_segments")?.value;
            if (upNumSeg) return Math.min(20, Math.max(1, parseInt(upNumSeg) || 3));
        } else if (mode === "自定义切点" || mode === "custom_cuts" || mode === "custom") {
            const cuts = originNode.widgets?.find(w => w.name === "custom_cuts")?.value || "";
            const count = cuts.split(",").filter(s => s.includes("-")).length;
            if (count > 0) return Math.min(20, count);
        } else {
            // 未执行时仅按向下对齐的约15秒参考长度估算，使用同一H3网格。
            const frames = originNode.properties?.pj_preview_cache?.total_frames ?? 345;
            const total = actual?.total_video_tokens ?? Math.max(2, Math.floor((frames - 5) / 17) * 5 + 2);
            const chunk = Number(originNode.widgets?.find(w => w.name === "chunk_seconds")?.value ?? 6);
            const ov = Number(originNode.widgets?.find(w => w.name === "overlap_seconds")?.value ?? 1.5);
            const groups = Math.min(Math.max(0, (total - 2) / 5), Math.max(1, Math.floor((chunk * 24 - 5) / 17 + 0.5)));
            const overlapGroups = Math.min(Math.max(0, groups - 1), Math.max(0, Math.floor((ov * 24 - 5) / 17 + 0.5)));
            const est = 1 + Math.max(0, Math.ceil((total - (groups * 5 + 2)) / Math.max(5, (groups - overlapGroups) * 5)));
            return Math.min(20, Math.max(1, est));
        }
    }

    return 3;
}

// 动态刷新采样器控件列表 (输入5变5段，输入15变15段)
function syncSamplerSegmentCount(node, isFirstPass) {
    const planned = hasSegmentPlan(node);
    const maxSeg = resolveTargetSegmentCount(node);
    const action = node.widgets?.find(w => w.name === (isFirstPass ? "repair_action" : "refine_action"));
    const local = action?.value === (isFirstPass ? "仅重绘目标段" : "仅重做目标段");
    const passive = action && (isFirstPass ? ["复用一采结果", "载入回修基准"].includes(action.value) :
        ["更新一采变更", "复用高清结果"].includes(action.value));

    // 仅保留当前规划的段号及工作流中已选的合法段号。
    const runModeWidget = node.widgets?.find(w => w.name === "run_mode");
    const generation = isFirstPass && node.widgets?.find(w => w.name === "generation_mode");
    if (generation) {
        if (!["分段编辑", "整段采样", "22帧接力"].includes(generation.value)) generation.value = "分段编辑";
        if (generation.value === "分段编辑" && String(runModeWidget?.value).includes("不分段")) generation.value = "整段采样";
    }
    if (runModeWidget) {
        const prefix = isFirstPass ? "仅重抽第" : "仅精修第";
        const defectAction = isFirstPass ? "按缺陷秒数重抽 (下方输入秒数)" : "按缺陷秒数精修 (下方输入秒数)";
        const allAction = isFirstPass ? "全部重新采样(刷新全部缓存)" : "全部精修(刷新全部缓存)";

        const newOptions = [
            ...(local ? ["全部目标段"] : ["智能缓存(仅重跑未锁定分段)",
                ...(isFirstPass && !generation ? ["整段全量采样(一采不分段)"] : []), allAction]),
            ...Array.from({ length: Math.min(20, maxSeg) }, (_, i) => `${prefix}${i + 1}段`),
            defectAction,
            "自定义指定分段(下方输入)"
        ];

        const savedSegment = Array.from({ length: 20 }, (_, i) => `${prefix}${i + 1}段`)
            .find(value => value === runModeWidget.value);
        if (savedSegment && !newOptions.includes(savedSegment)) {
            newOptions.splice(newOptions.indexOf(defectAction), 0, savedSegment);
        }

        runModeWidget.options = runModeWidget.options || {};
        runModeWidget.options.values = newOptions;
        runModeWidget.label = local ? "局部重做范围" : (isFirstPass ? "本次生成范围" : "精修范围（含后续段）");

        // 仅清洗后端不支持的值，不因预览段数变化重置目标。
        if (!newOptions.includes(runModeWidget.value)) {
            runModeWidget.value = local ? "全部目标段" : "智能缓存(仅重跑未锁定分段)";
        }
    }

    // 2. 控制 1~20 控件的动态显隐
    const runMode = String(runModeWidget?.value || "");
    const relay = planned && !local && !passive && generation?.value === "22帧接力";
    const full = !planned || (!local && (generation?.value === "整段采样" || (isFirstPass && !generation && runMode.includes("不分段"))));
    toggleWidget(node, generation, planned && !local && !passive);
    toggleWidget(node, runModeWidget, planned && !relay && !full && !passive);
    if (action) action.label = isFirstPass ? "一采操作" : "二采操作";
    const relayAction = node.widgets.find(w => w.name === "relay_action");
    const relayStart = node.widgets.find(w => w.name === "relay_start_segment");
    if (relayAction && !["全部生成", "从指定段重做"].includes(relayAction.value)) relayAction.value = "全部生成";
    if (relayStart) {
        if (!Number.isInteger(relayStart.value) || relayStart.value < 1) relayStart.value = 1;
        relayStart.options = relayStart.options || {};
        relayStart.options.max = Math.max(maxSeg, relayStart.value);
    }
    toggleWidget(node, relayAction, relay);
    toggleWidget(node, relayStart, relay && relayAction?.value === "从指定段重做");
    const customRerun = node.widgets?.find(w => w.name === "custom_rerun_segments");
    const targetTime = node.widgets?.find(w => w.name === "target_time_second");

    toggleWidget(node, customRerun, planned && !relay && !full && !passive && runMode.includes("自定义"));
    toggleWidget(node, targetTime, planned && !relay && !full && !passive && runMode.includes("缺陷秒数"));

    // 锁定仅影响智能缓存；种子偏移放入高级设置。
    for (let i = 1; i <= 20; i++) {
        const lockW = node.widgets?.find(w => w.name === `seg${i}_lock`);
        const offsetW = node.widgets?.find(w => w.name === `seg${i}_seed_offset`);

        const inRange = full ? false : (i <= maxSeg || lockW?.value === true || (isFirstPass && offsetW?.value !== i - 1));

        // 预览外已有的设置也需要可见，避免载入工作流后无法修改。
        toggleWidget(node, lockW, inRange && !relay && !local && !passive && runMode.includes("智能缓存"));
        if (!isFirstPass && lockW) {
            lockW.options = lockW.options || {};
            lockW.options.tooltip = "仅当前段及前段均可复用时生效；前段重做会重建全部后续段。";
        }

        // 一采独立种子偏移：在当前分段数内展示
        if (isFirstPass) {
            toggleWidget(node, offsetW, !passive && (full ? i === 1 : inRange) && !!node.properties?.pj_advanced_open);
        }
    }

    const boundary = isFirstPass && !!node.widgets.find(w => w.name === "boundary_constraints")?.value;
    toggleWidget(node, node.widgets.find(w => w.name === "boundary_constraints"), isFirstPass && !full && !relay && !local && !passive);
    const blend = node.widgets.find(w => w.name === "blend_mode");
    if (!isFirstPass && blend) blend.label = "音频重叠过渡";
    const showBlend = !full && !boundary && !relay && !local && !passive &&
        (isFirstPass || node.widgets.find(w => w.name === "keep_original_audio")?.value === false);
    for (const name of ["sync_ref_video", "blend_mode", "unload_model_after_sampling"]) {
        toggleWidget(node, node.widgets.find(w => w.name === name),
            !!node.properties?.pj_advanced_open && (name !== "blend_mode" || showBlend));
    }
    const repairLabels = {repair_denoise: "局部重绘降噪", repair_steps: "重绘步数", repair_scheduler: "重绘调度器",
        repair_boundary_tokens: "边界保护潜步", repair_request_id: "重绘操作编号", refine_request_id: "精修操作编号"};
    for (const [name, label] of Object.entries(repairLabels)) {
        const w = node.widgets.find(w => w.name === name);
        if (w) w.label = label;
        toggleWidget(node, w, local && (name === "repair_denoise" || !!node.properties?.pj_advanced_open));
    }
    updateSamplerControls(node, isFirstPass);
    autoResizeNode(node);
}

function updateSamplerControls(node, isFirstPass) {
    node.widgets = node.widgets.filter(w => !(w.type === "button" &&
        /读取规划|刷新分段|总线同步|已刷新|一采当前为整段采样/.test(w.name || "")));
    let button = node.widgets.find(w => w.name === "pj_advanced_toggle");
    if (!button) {
        button = node.addWidget("button", "pj_advanced_toggle", null, () => {
            node.properties.pj_advanced_open = !node.properties.pj_advanced_open;
            syncSamplerSegmentCount(node, isFirstPass);
            node.setDirtyCanvas?.(true, true);
        }, { serialize: false });
    }
    button.label = node.properties?.pj_advanced_open ? "收起高级设置" : "展开高级设置";
}

// ==========================================
// 4. 一采节点与二采节点显隐更新
// ==========================================
function updateFirstPassVisibility(node) {
    if (!node.widgets) return;
    localizeAndSanitizeCommon(node);

    // 彻底隐藏任何冗余切分参数（全部由上游规划器总线统一管理）
    const redundantWidgets = [
        "split_mode", "chunk_seconds", "overlap_seconds", "num_segments",
        "custom_cuts", "align_51_frames", "fps"
    ];
    for (const name of redundantWidgets) {
        const w = node.widgets.find(w => w.name === name);
        if (w) toggleWidget(node, w, false);
    }

    syncSamplerSegmentCount(node, true);
}

function updateSecondPassVisibility(node) {
    if (!node.widgets) return;
    localizeAndSanitizeCommon(node);

    // 彻底隐藏任何冗余切分参数（全部由上游规划器总线统一管理）
    const redundantWidgets = [
        "split_mode", "chunk_seconds", "overlap_seconds", "num_segments",
        "custom_cuts", "align_51_frames", "fps"
    ];
    for (const name of redundantWidgets) {
        const w = node.widgets.find(w => w.name === name);
        if (w) toggleWidget(node, w, false);
    }

    syncSamplerSegmentCount(node, false);
}

// 绑定采样器事件监听
function attachSamplerHandlers(node, isFirstPass) {
    if (!node.widgets) return;
    node.properties = node.properties || {};

    if (!node._pj_summary_attached) {
        node._pj_summary_attached = true;
        const onExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const summary = message?.pj_sampling_summary?.[0];
            if (Number.isInteger(summary?.total_video_tokens) && summary.total_video_tokens > 0) {
                this._pjSamplingSummary = {...summary, sourceKey: samplingSourceKey(this)};
                if (!Number.isInteger(summary.n_chunks) || summary.n_chunks < 1) this._pjSamplingSummary.n_chunks = null;
                syncSamplerSegmentCount(this, isFirstPass);
            }
            return result;
        };
    }

    node.color = isFirstPass ? "#134e4a" : "#1e293b";
    node.bgcolor = isFirstPass ? "#042f2e" : "#0f172a";
    if (isFirstPass) {
        if (!node.title || node.title.startsWith("PJ_FirstPass")) node.title = "PJ一采分段重抽采样器 (单段重抽/缓存)";
    } else {
        if (!node.title || node.title.startsWith("PJ_Segment") || node.title.startsWith("PJSegment")) node.title = "PJ二采分段精修采样器 (分段精修/局部重做)";
    }

    const watchNames = ["repair_action", "refine_action", "generation_mode", "relay_action", "relay_start_segment", "run_mode", "target_time_second", "custom_rerun_segments", "sync_ref_video", "boundary_constraints", "blend_mode", "keep_original_audio", "unload_model_after_sampling"];
    for (const name of watchNames) {
        const w = node.widgets.find(widget => widget.name === name);
        if (w && !w._pj_attached) {
            w._pj_attached = true;
            const orig = w.callback;
            w.callback = function () {
                const res = orig?.apply(this, arguments);
                if (this.name === "generation_mode" && this.value !== "整段采样") {
                    const mode = node.widgets.find(widget => widget.name === "run_mode");
                    if (mode && String(mode.value).includes("不分段")) mode.value = "全部重新采样(刷新全部缓存)";
                }
                const curVal = String(this.value || "");
                // 当用户明确选择【全部重新采样】或【全部精修】时，主动将下方所有分段解除锁定（重置为未锁定）
                if (curVal.includes("全部重新采样") || curVal.includes("全部精修")) {
                    for (let i = 1; i <= 20; i++) {
                        const lockW = node.widgets?.find(lw => lw.name === `seg${i}_lock`);
                        if (lockW) lockW.value = false;
                    }
                    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
                    if (app.canvas && app.canvas.setDirty) app.canvas.setDirty(true, true);
                }
                if (isFirstPass) updateFirstPassVisibility(node);
                else updateSecondPassVisibility(node);
                return res;
            };
        }
    }

    // 监听连接变化 (若连上上游分段规划器，自动同步)
    if (!node._pj_conn_attached) {
        node._pj_conn_attached = true;
        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            if (node._pjSamplingSummary?.sourceKey !== samplingSourceKey(node)) delete node._pjSamplingSummary;
            const res = origOnConnectionsChange?.apply(this, arguments);
            setTimeout(() => {
                if (isFirstPass) updateFirstPassVisibility(node);
                else updateSecondPassVisibility(node);
            }, 100);
            return res;
        };
    }

    setTimeout(() => {
        if (isFirstPass) updateFirstPassVisibility(node);
        else updateSecondPassVisibility(node);
    }, 50);
}

// ==========================================
// 5. PJ_SegmentPlanner 可视时间戳仪表盘与分段播放器
// ==========================================
function notifyDownstreamSamplers(plannerNode) {
    if (!plannerNode.outputs || !app.graph) return;
    const planOutput = plannerNode.outputs.find(o => o.type === "PJ_SEGMENT_PLAN" || (o.name && o.name.includes("分段规划")));
    if (!planOutput || !planOutput.links) return;

    function notifyTarget(targetId) {
        const targetNode = app.graph.getNodeById(targetId);
        if (!targetNode) return;
        if (targetNode.type === "Reroute" || targetNode.type === "ReroutePrimitive") {
            const outLinks = targetNode.outputs?.[0]?.links || [];
            for (const lid of outLinks) {
                const l = app.graph.links[lid];
                if (l && l.target_id != null) notifyTarget(l.target_id);
            }
        } else if (targetNode.type === "PJ_FirstPass_SegmentSampler") {
            updateFirstPassVisibility(targetNode);
        } else if (targetNode.type === "PJ_SegmentSampler" || targetNode.type === "PJSegmentSampler") {
            updateSecondPassVisibility(targetNode);
        }
    }

    for (const linkId of planOutput.links) {
        const link = app.graph.links[linkId];
        if (link && link.target_id != null) {
            notifyTarget(link.target_id);
        }
    }
}

function setupPlannerNode(node) {
    localizeAndSanitizeCommon(node);
    node.color = "#312e81"; // 现代科技靛蓝 (Indigo)
    node.bgcolor = "#1e1b4b";
    if (!node.title || node.title.startsWith("PJ_SegmentPlanner")) {
        node.title = "PJ分段规划器 (时序总控/可视播放器)";
    }

    const modeWidget = node.widgets?.find(w => w.name === "split_mode");
    const chunkSec = node.widgets?.find(w => w.name === "chunk_seconds");
    const overlapSec = node.widgets?.find(w => w.name === "overlap_seconds");
    const numSeg = node.widgets?.find(w => w.name === "num_segments");
    const customCuts = node.widgets?.find(w => w.name === "custom_cuts");
    if (customCuts) {
        customCuts.label = "处理时间范围(秒)";
        customCuts.options = customCuts.options || {};
        customCuts.options.tooltip = "例如5-10表示只处理此范围，输出仍保持完整长度；未覆盖区域保留输入，空潜空间不会自动补生成。";
    }

    function updatePlannerWidgets() {
        if (modeWidget) {
            const mode = modeWidget.value;
            if (mode === "按秒自动切分") {
                toggleWidget(node, chunkSec, true);
                toggleWidget(node, overlapSec, true);
                toggleWidget(node, numSeg, false);
                toggleWidget(node, customCuts, false);
            } else if (mode === "固定分段数") {
                toggleWidget(node, chunkSec, false);
                toggleWidget(node, overlapSec, true);
                toggleWidget(node, numSeg, true);
                toggleWidget(node, customCuts, false);
            } else if (mode === "自定义切点") {
                toggleWidget(node, chunkSec, false);
                toggleWidget(node, overlapSec, false);
                toggleWidget(node, numSeg, false);
                toggleWidget(node, customCuts, true);
            }
        }
        autoResizeNode(node);
    }

    const plannerWatchNames = [
        "fps",
        "split_mode",
        "chunk_seconds",
        "overlap_seconds",
        "num_segments",
        "custom_cuts"
    ];
    for (const name of plannerWatchNames) {
        const w = node.widgets?.find(widget => widget.name === name);
        if (w && !w._pj_attached) {
            w._pj_attached = true;
            const orig = w.callback;
            w.callback = function () {
                const res = orig?.apply(this, arguments);
                delete node.pj_cached_n_chunks;
                if (node.properties) delete node.properties.pj_preview_cache;
                updatePlannerWidgets();
                notifyDownstreamSamplers(node);
                return res;
            };
        }
    }

    updatePlannerWidgets();

    if (!node._pj_conn_attached) {
        node._pj_conn_attached = true;
        const origOnConn = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            const res = origOnConn?.apply(this, arguments);
            setTimeout(() => {
                notifyDownstreamSamplers(node);
            }, 100);
            return res;
        };
    }

    setTimeout(updatePlannerWidgets, 50);
}

// 渲染分段规划器的交互式播放器与分段卡片
function renderPlannerDashboard(node, data) {
    if (!data) return;
    node.pj_cached_n_chunks = data.n_chunks;

    let widget = node.widgets?.find(w => w.name === "pj_planner_dashboard");
    if (!widget) {
        const root = document.createElement("div");
        root.style.width = "100%";
        root.style.overflow = "hidden";
        const container = document.createElement("div");
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.gap = "8px";
        container.style.padding = "10px";
        container.style.background = "#0f172a";
        container.style.border = "1px solid #334155";
        container.style.borderRadius = "8px";
        container.style.boxSizing = "border-box";
        container.style.width = "100%";
        container.style.fontSize = "12px";
        container.style.color = "#f8fafc";
        container.style.overflow = "hidden";
        container.style.flexShrink = "0";
        root.appendChild(container);

        widget = node.addDOMWidget("pj_planner_dashboard", "dashboard", root, {
            serialize: false, hideOnZoom: false, margin: 0,
            getMinHeight: () => widget?.pjHeight || 100,
            getMaxHeight: () => widget?.pjHeight || 100,
            getHeight: () => widget?.pjHeight || 100,
        });
        widget.pjContent = container;
        widget.pjHeight = 100;
        widget.computeSize = width => [width, widget.pjHeight];
        const observer = new ResizeObserver(() => {
            const height = Math.ceil(container.offsetHeight);
            if (height > 0 && height !== widget.pjHeight) {
                widget.pjHeight = height;
                autoResizeNode(node);
            }
        });
        observer.observe(container);
        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
            observer.disconnect();
            return onRemoved?.apply(this, arguments);
        };
    }

    const container = widget.pjContent;
    container.innerHTML = "";

    // 1. 顶部时序统计横幅
    const isReference = data.has_media === false;
    const topBanner = document.createElement("div");
    topBanner.style.display = "flex";
    topBanner.style.justifyContent = "space-between";
    topBanner.style.alignItems = "center";
    topBanner.style.background = "#1e293b";
    topBanner.style.padding = "6px 10px";
    topBanner.style.borderRadius = "6px";
    topBanner.style.fontSize = "12px";
    topBanner.style.fontWeight = "bold";
    topBanner.innerHTML = `
        <span style="color: #38bdf8;">${isReference ? '📐 规则参考时长: ' : '🎬 媒体总时长: '}${data.effective_seconds}s (${data.total_frames}帧)</span>
        <span style="color: #34d399;">📐 规划切分: ${data.n_chunks} 个分段</span>
        <span style="color: #fbbf24;">⚡ ${data.fps} FPS</span>
    `;
    container.appendChild(topBanner);
    if (data.timing_note) {
        const note = document.createElement("div");
        note.style.color = "#94a3b8";
        note.textContent = data.timing_note;
        container.appendChild(note);
    }

    // 2. 视频播放器 (如果合成出了视频)
    let videoEl = null;
    let loopSegmentHandler = null;

    if (data.has_video && data.video_filename) {
        const videoWrapper = document.createElement("div");
        videoWrapper.style.position = "relative";
        videoWrapper.style.borderRadius = "6px";
        videoWrapper.style.overflow = "hidden";
        videoWrapper.style.background = "#000";
        videoWrapper.style.maxHeight = "220px";
        videoWrapper.style.display = "flex";
        videoWrapper.style.justifyContent = "center";

        const video = document.createElement("video");
        video.src = api.apiURL(`/view?filename=${data.video_filename}&type=temp`);
        video.style.width = "100%";
        video.style.maxHeight = "220px";
        video.style.objectFit = "contain";
        video.controls = true;
        video.autoplay = false;
        video.muted = false;

        videoWrapper.appendChild(video);
        container.appendChild(videoWrapper);
        videoEl = video;
    } else {
        const tipEl = document.createElement("div");
        tipEl.style.padding = "8px 10px";
        tipEl.style.background = isReference ? "rgba(59, 130, 246, 0.12)" : "rgba(16, 185, 129, 0.1)";
        tipEl.style.border = isReference ? "1px dashed #3b82f6" : "1px dashed #10b981";
        tipEl.style.borderRadius = "6px";
        tipEl.style.color = isReference ? "#93c5fd" : "#6ee7b7";
        tipEl.style.fontSize = "11px";
        tipEl.style.lineHeight = "1.5";
        if (isReference) {
            tipEl.innerHTML = `💡 <b>规则待命模式</b>：当前规划器未直连潜空间或视频，已按参考时长生成预览规则。实际采样时一采与二采将以其各自输入的潜空间长度自动动态切分！<br><span style="color:#60a5fa;">（若需100%物理潜步预览或启用播放器，可直接将潜空间或视频接入上方端口）</span>`;
        } else {
            tipEl.innerHTML = `💡 <b>提示</b>：连接上方 <code>images</code> 与 <code>audio</code> 端口后，此处将自动渲染内置试听播放器，支持精准截取任意分段一键试听！`;
        }
        container.appendChild(tipEl);
    }

    // 3. 分段列表与点播控制台
    const listTitle = document.createElement("div");
    listTitle.style.display = "flex";
    listTitle.style.justifyContent = "space-between";
    listTitle.style.alignItems = "center";
    listTitle.style.marginTop = "4px";
    listTitle.style.fontSize = "11px";
    listTitle.style.color = "#94a3b8";

    listTitle.innerHTML = `<span>📋 分段时序明细 (点击按钮精准播放对应分段):</span>`;

    if (videoEl) {
        const fullLoopBtn = document.createElement("button");
        fullLoopBtn.innerText = "🔁 全片循环";
        fullLoopBtn.style.background = "#475569";
        fullLoopBtn.style.color = "#f8fafc";
        fullLoopBtn.style.border = "none";
        fullLoopBtn.style.borderRadius = "4px";
        fullLoopBtn.style.padding = "2px 8px";
        fullLoopBtn.style.cursor = "pointer";
        fullLoopBtn.style.fontSize = "10px";
        fullLoopBtn.onclick = () => {
            if (loopSegmentHandler) {
                videoEl.removeEventListener("timeupdate", loopSegmentHandler);
                loopSegmentHandler = null;
            }
            videoEl.loop = true;
            videoEl.currentTime = 0;
            videoEl.play();
        };
        listTitle.appendChild(fullLoopBtn);
    }
    container.appendChild(listTitle);

    const segListContainer = document.createElement("div");
    segListContainer.style.display = "flex";
    segListContainer.style.flexDirection = "column";
    segListContainer.style.gap = "4px";
    segListContainer.style.paddingRight = "4px";

    if (data.chunks && Array.isArray(data.chunks)) {
        data.chunks.forEach((chunk) => {
            const card = document.createElement("div");
            card.style.display = "flex";
            card.style.justifyContent = "space-between";
            card.style.alignItems = "center";
            card.style.background = "#1e293b";
            card.style.border = "1px solid #334155";
            card.style.borderRadius = "5px";
            card.style.padding = "5px 8px";
            card.style.fontSize = "11px";

            const leftInfo = document.createElement("div");
            leftInfo.style.display = "flex";
            leftInfo.style.alignItems = "center";
            leftInfo.style.gap = "8px";

            const badge = document.createElement("span");
            badge.style.background = "#0284c7";
            badge.style.color = "#fff";
            badge.style.borderRadius = "3px";
            badge.style.padding = "1px 5px";
            badge.style.fontWeight = "bold";
            badge.style.fontSize = "10px";
            badge.innerText = `段 ${chunk.segment < 10 ? '0' + chunk.segment : chunk.segment}`;

            const timeText = document.createElement("span");
            timeText.style.color = "#e2e8f0";
            timeText.innerText = `${chunk.start_sec.toFixed(2)}s ~ ${chunk.end_sec.toFixed(2)}s (${chunk.duration_sec}s, 帧 ${chunk.start_frame}~${chunk.end_frame})`;
            if (chunk.segment > 1 && chunk.overlap_frames !== undefined) {
                timeText.innerText += `\n与前段重叠 ${chunk.overlap_seconds.toFixed(3)}s / ${chunk.overlap_frames}帧`;
            }

            leftInfo.appendChild(badge);
            leftInfo.appendChild(timeText);
            card.appendChild(leftInfo);

            if (videoEl) {
                const playBtn = document.createElement("button");
                playBtn.innerText = "▶ 播放本段";
                playBtn.style.background = "#059669";
                playBtn.style.color = "#ffffff";
                playBtn.style.border = "none";
                playBtn.style.borderRadius = "4px";
                playBtn.style.padding = "3px 8px";
                playBtn.style.cursor = "pointer";
                playBtn.style.fontSize = "10px";
                playBtn.style.fontWeight = "bold";
                playBtn.style.transition = "all 0.15s ease";

                playBtn.onclick = () => {
                    if (loopSegmentHandler) {
                        videoEl.removeEventListener("timeupdate", loopSegmentHandler);
                    }
                    videoEl.loop = false;
                    videoEl.currentTime = chunk.start_sec;
                    videoEl.play();

                    loopSegmentHandler = () => {
                        if (videoEl.currentTime >= chunk.end_sec || videoEl.currentTime < chunk.start_sec) {
                            videoEl.currentTime = chunk.start_sec;
                            videoEl.play();
                        }
                    };
                    videoEl.addEventListener("timeupdate", loopSegmentHandler);

                    // 动态视觉反馈
                    container.querySelectorAll(".pj-card-active").forEach(el => {
                        el.classList.remove("pj-card-active");
                        el.style.borderColor = "#334155";
                    });
                    card.classList.add("pj-card-active");
                    card.style.borderColor = "#10b981";
                };

                card.appendChild(playBtn);
            }

            segListContainer.appendChild(card);
        });
    }

    container.appendChild(segListContainer);
    notifyDownstreamSamplers(node);
    autoResizeNode(node);
}

// 检查并刷新画布所有节点
function scanAndLocalizeAllNodes() {
    if (!app.graph || !app.graph._nodes) return;
    for (const node of app.graph._nodes) {
        if (node.type === "PJ_FirstPass_SegmentSampler") {
            attachSamplerHandlers(node, true);
        } else if (node.type === "PJ_SegmentSampler" || node.type === "PJSegmentSampler") {
            attachSamplerHandlers(node, false);
        } else if (node.type === "PJ_SegmentPlanner") {
            setupPlannerNode(node);
            if (node.properties?.["pj_preview_cache"]) {
                renderPlannerDashboard(node, node.properties["pj_preview_cache"]);
            }
        }
    }
}

// ==========================================
// 6. 工作流旧版参数清洗器 (兼容历史 Workflow)
// ==========================================
function sanitizePlannerWidgets(info) {
    if (info?.properties?.pj_preview_cache?.timing_policy !== "nearest_h3" && info?.properties) {
        delete info.properties.pj_preview_cache;
    }
    if (info && info.widgets_values && Array.isArray(info.widgets_values)) {
        // Six controls; remove the obsolete grid toggle without shifting saved values.
        const modes = ["按秒自动切分", "固定分段数", "自定义切点", "auto_by_seconds", "fixed_segments", "custom", "custom_cuts"];
        if (typeof info.widgets_values[0] === "string" && modes.includes(info.widgets_values[0])) {
            const fpsVal = info.widgets_values[6] ?? 24.0;
            info.widgets_values = [fpsVal, ...info.widgets_values.slice(0, 5)];
            return;
        }
        if (typeof info.widgets_values[0] === "number") {
            while (info.widgets_values.length > 2 && !modes.includes(info.widgets_values[1])) {
                info.widgets_values.splice(1, 1);
            }
            if (modes.includes(info.widgets_values[1])) info.widgets_values = info.widgets_values.slice(0, 6);
        }
    }
}

function sanitizeSamplerWidgets(info, isFirstPass) {
    if (!info || !info.widgets_values || !Array.isArray(info.widgets_values)) return;

    const legacySplitModes = [
        "按秒自动切分", "固定分段数", "自定义切点",
        "auto_by_seconds", "fixed_segments", "custom", "custom_cuts"
    ];

    // 检测是否为旧版工作流（以旧切分模式字符串开头）
    if (typeof info.widgets_values[0] === "string" && legacySplitModes.includes(info.widgets_values[0])) {
        // 移除旧版前置的 6 个规划参数: [split_mode, chunk_seconds, overlap_seconds, num_segments, custom_cuts, align_51_frames]
        info.widgets_values.splice(0, 6);

        if (isFirstPass) {
            // 一采当前结构: [sync_ref_video(bool), run_mode(str), target_time_second(float), custom_rerun_segments(str), base_seed(int), ...]
            // 旧版在 run_mode(str) 后面直接是 base_seed(number)，缺少 target_time_second 与 custom_rerun_segments:
            if (info.widgets_values.length > 2 && typeof info.widgets_values[2] === "number" && typeof info.widgets_values[3] !== "string") {
                info.widgets_values.splice(2, 0, 0.0, "5");
            }
        } else {
            // 二采当前结构: [run_mode(str), target_time_second(float), custom_rerun_segments(str), seg1_lock(bool)...]
            // 旧版在 run_mode(str) 后面直接是 seg1_lock(boolean)，缺少 target_time_second 与 custom_rerun_segments:
            if (info.widgets_values.length > 1 && typeof info.widgets_values[1] === "boolean") {
                info.widgets_values.splice(1, 0, 0.0, "5");
            }
        }

        // 如果旧版末尾附带了旧 fps (例如 24.0)，将其剔除
        const lastIdx = info.widgets_values.length - 1;
        if (lastIdx > 0 && typeof info.widgets_values[lastIdx] === "number" && (info.widgets_values[lastIdx] === 24.0 || info.widgets_values[lastIdx] === 24)) {
            info.widgets_values.pop();
        }
    }
}

// ==========================================
// 7. 注册 ComfyUI 前端扩展
// ==========================================
app.registerExtension({
    name: "PJ.SegmentSampler.InteractiveSuite",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "PJ_SegmentPlanner") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                setupPlannerNode(this);
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
                if (message?.preview_data && message.preview_data[0]) {
                    this.properties["pj_preview_cache"] = message.preview_data[0];
                    renderPlannerDashboard(this, message.preview_data[0]);
                }
                return r;
            };

            const origConfigure = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                sanitizePlannerWidgets(info);
                const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
                setupPlannerNode(this);
                if (this.properties?.["pj_preview_cache"]) {
                    renderPlannerDashboard(this, this.properties["pj_preview_cache"]);
                }
                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                sanitizePlannerWidgets(info);
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                setupPlannerNode(this);
                if (this.properties?.["pj_preview_cache"]) {
                    renderPlannerDashboard(this, this.properties["pj_preview_cache"]);
                }
                return r;
            };
        } else if (["PJ_H3EditPrepare", "PJ_H3SegmentContext"].includes(nodeData.name)) {
            if (nodeData.name === "PJ_H3SegmentContext") {
                const configure = nodeType.prototype.configure;
                nodeType.prototype.configure = function (info) {
                    for (const input of info.inputs || []) {
                        if (input.name === "reference_images") input.name = "ref_images.ref_image_0";
                    }
                    return configure?.apply(this, arguments);
                };
            }
            for (const event of ["onNodeCreated", "onConfigure"]) {
                const original = nodeType.prototype[event];
                nodeType.prototype[event] = function () {
                    const result = original?.apply(this, arguments);
                    localizeAndSanitizeCommon(this);
                    this.color = "#134e4a";
                    this.bgcolor = "#042f2e";
                    return result;
                };
            }
        } else if (nodeData.name === "PJ_FirstPass_SegmentSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                attachSamplerHandlers(this, true);
                return r;
            };
            const origConfigure = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                sanitizeSamplerWidgets(info, true);
                const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
                attachSamplerHandlers(this, true);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                sanitizeSamplerWidgets(info, true);
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                attachSamplerHandlers(this, true);
                return r;
            };
        } else if (nodeData.name === "PJ_SegmentSampler" || nodeData.name === "PJSegmentSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const mode = this.widgets?.find(w => w.name === "run_mode");
                if (mode) mode.value = "全部精修(刷新全部缓存)";
                attachSamplerHandlers(this, false);
                return r;
            };
            const origConfigure = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                sanitizeSamplerWidgets(info, false);
                const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
                attachSamplerHandlers(this, false);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                sanitizeSamplerWidgets(info, false);
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                attachSamplerHandlers(this, false);
                return r;
            };
        }
    },
    async nodeCreated(node) {
        if (node.comfyClass === "PJ_SegmentPlanner") {
            setupPlannerNode(node);
        } else if (node.comfyClass === "PJ_FirstPass_SegmentSampler") {
            attachSamplerHandlers(node, true);
        } else if (node.comfyClass === "PJ_SegmentSampler" || node.comfyClass === "PJSegmentSampler") {
            attachSamplerHandlers(node, false);
        }
    },
    async afterConfigureGraph() {
        scanAndLocalizeAllNodes();
    },
    setup() {
        setTimeout(scanAndLocalizeAllNodes, 500);
        setTimeout(scanAndLocalizeAllNodes, 1500);
    }
});
