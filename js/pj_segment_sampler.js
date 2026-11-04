import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HIDDEN_TAG = "pj_hidden";

// ==========================================
// 1. 中文标签与映射字典 (完整汉化核心)
// ==========================================
const WIDGET_LABELS = {
    // 规划模式相关
    "split_mode": "切分规划模式",
    "chunk_seconds": "单段最大秒数",
    "overlap_seconds": "重叠过渡秒数",
    "num_segments": "固定分段数量",
    "custom_cuts": "自定义切点范围",
    "align_51_frames": "51帧网格吸附",
    "fps": "视频帧率",
    "duration_seconds": "视频总时长(秒)",

    // 一采专有功能
    "sync_ref_video": "同步参考视频",
    "base_seed": "基础随机种子",

    // 运行重抽与缓存锁
    "run_mode": "运行重抽模式",
    "target_time_second": "缺陷定位秒数",
    "custom_rerun_segments": "自定义重抽段号",

    // 算法与系统控制
    "blend_mode": "平滑过渡算法",
    "keep_original_audio": "保留一采原音频",
    "unload_model_after_sampling": "释放显存保卫VAE"
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
    "segment_plan": "分段规划总线"
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
}

function autoResizeNode(node) {
    [20, 100, 250].forEach(delay => {
        setTimeout(() => {
            if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
                const dashboard = node.widgets?.find(w => w.name === "pj_planner_dashboard");
                if (dashboard?.pjContent?.offsetHeight > 0) {
                    dashboard.pjHeight = Math.ceil(dashboard.pjContent.offsetHeight);
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
                w.options.on = "开启(原音保真)";
                w.options.off = "二采融合";
                w.label_on = "开启(原音保真)";
                w.label_off = "二采融合";
                if (typeof w.value !== "boolean") {
                    w.value = true;
                }
            } else if (w.name === "sync_ref_video") {
                w.options = w.options || {};
                w.options.on = "开启(时序同步)";
                w.options.off = "关闭";
                w.label_on = "开启(时序同步)";
                w.label_off = "关闭";
                if (typeof w.value !== "boolean") {
                    w.value = true;
                }
            } else if (w.name === "align_51_frames") {
                w.options = w.options || {};
                w.options.on = "开启51帧吸附";
                w.options.off = "关闭";
                w.label_on = "开启51帧吸附";
                w.label_off = "关闭";
                if (typeof w.value !== "boolean") {
                    w.value = false;
                }
            } else if (w.name && w.name.endsWith("_lock")) {
                w.options = w.options || {};
                w.options.on = "已锁定";
                w.options.off = "未锁定";
                w.label_on = "已锁定";
                w.label_off = "未锁定";
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
    if (!node.inputs || !app.graph) return null;
    const planInput = node.inputs.find(i => i.name === "segment_plan" || i.label === "分段规划总线" || i.type === "PJ_SEGMENT_PLAN");
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

// 检查当前采样器是否已连接上游【分段规划总线】
function isSegmentPlanConnected(node) {
    return getUpstreamPlannerNode(node) != null;
}

// 获取上游规划器或当前设置的目标分段数
function resolveTargetSegmentCount(node) {
    // 1. 优先实时读取上游 PJ_SegmentPlanner 最新控件值
    const originNode = getUpstreamPlannerNode(node);
    if (originNode) {
        // 次级缓存读取：上游规划器已执行或有预览缓存时直接获取
        if (originNode.pj_cached_n_chunks) {
            return Math.min(20, Math.max(1, originNode.pj_cached_n_chunks));
        }
        if (originNode.properties?.["pj_preview_cache"]?.n_chunks) {
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
            // 按秒自动切分模式：根据默认参考时长 15 秒与窗口步长实时估算
            let dur = 15.0;
            const chunk = parseFloat(originNode.widgets?.find(w => w.name === "chunk_seconds")?.value || 6);
            const ov = parseFloat(originNode.widgets?.find(w => w.name === "overlap_seconds")?.value || 1.5);
            const step = Math.max(0.5, chunk - ov);
            const est = Math.ceil(dur / step);
            return Math.min(20, Math.max(1, est));
        }
    }

    // 2. 读取本节点的 num_segments (单节点独立模式 / 历史工作流兼容)
    const numSegWidget = node.widgets?.find(w => w.name === "num_segments");
    if (numSegWidget) {
        const val = parseInt(numSegWidget.value);
        if (!isNaN(val) && val >= 1) {
            return Math.min(20, Math.max(1, val));
        }
    }

    return 3;
}

// 动态刷新采样器控件列表 (输入5变5段，输入15变15段)
function syncSamplerSegmentCount(node, isFirstPass) {
    const maxSeg = resolveTargetSegmentCount(node);

    // 1. 动态生成匹配当前分段数量的 run_mode 选项
    const runModeWidget = node.widgets?.find(w => w.name === "run_mode");
    if (runModeWidget) {
        const prefix = isFirstPass ? "仅重抽第" : "仅精修第";
        const defectAction = isFirstPass ? "按缺陷秒数重抽 (下方输入秒数)" : "按缺陷秒数精修 (下方输入秒数)";
        const allAction = isFirstPass ? "全部重新采样(刷新全部缓存)" : "全部精修(刷新全部缓存)";

        const newOptions = [
            "智能缓存(仅重跑未锁定分段)",
            ...(isFirstPass ? ["整段全量采样(一采不分段)"] : []),
            allAction,
            ...Array.from({ length: maxSeg }, (_, i) => `${prefix}${i + 1}段`),
            defectAction,
            "自定义指定分段(下方输入)"
        ];

        runModeWidget.options = runModeWidget.options || {};
        runModeWidget.options.values = newOptions;

        // 如果当前值是无效的布尔值或超出了新选项，安全重置
        if (!newOptions.includes(runModeWidget.value)) {
            runModeWidget.value = "智能缓存(仅重跑未锁定分段)";
        }
    }

    // 2. 控制 1~20 控件的动态显隐
    const runMode = String(runModeWidget?.value || "");
    const customRerun = node.widgets?.find(w => w.name === "custom_rerun_segments");
    const targetTime = node.widgets?.find(w => w.name === "target_time_second");

    toggleWidget(node, customRerun, runMode.includes("自定义"));
    toggleWidget(node, targetTime, runMode.includes("缺陷秒数"));

    // 关键修正：1~maxSeg 的锁定开关始终显示（整段不分段模式除外）
    for (let i = 1; i <= 20; i++) {
        const lockW = node.widgets?.find(w => w.name === `seg${i}_lock`);
        const offsetW = node.widgets?.find(w => w.name === `seg${i}_seed_offset`);

        const inRange = (isFirstPass && runMode.includes("不分段")) ? false : (i <= maxSeg);

        // 锁定开关：在当前分段数内始终展示，超出部分严格隐藏
        toggleWidget(node, lockW, inRange);

        // 一采独立种子偏移：在当前分段数内展示
        if (isFirstPass) {
            toggleWidget(node, offsetW, inRange);
        }
    }

    autoResizeNode(node);
}

// 为采样器注入便捷的“读取规划器参数 / 刷新分段适配”按钮 (单例且自动清除任何历史多余重复项)
function ensureSyncButton(node, isFirstPass) {
    if (!node.widgets) return;

    const isConnected = isSegmentPlanConnected(node);
    const maxSeg = resolveTargetSegmentCount(node);
    const runModeVal = String(node.widgets?.find(w => w.name === "run_mode")?.value || "");
    const targetBtnName = (isFirstPass && runModeVal.includes("不分段"))
        ? `⚡ 一采当前为整段采样模式 (共 1 整段)`
        : (isConnected
            ? `🔗 读取规划器参数 (当前共 ${maxSeg} 段)`
            : `🔄 读取规划参数 (当前共 ${maxSeg} 段)`);

    const handleButtonClick = (widget) => {
        syncSamplerSegmentCount(node, isFirstPass);
        const curSeg = resolveTargetSegmentCount(node);
        if (widget) {
            widget.name = `✅ 已读取规划参数：当前共 ${curSeg} 段！`;
            if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
            if (app.canvas && app.canvas.setDirty) app.canvas.setDirty(true, true);
            setTimeout(() => {
                const refreshedConnected = isSegmentPlanConnected(node);
                widget.name = refreshedConnected
                    ? `🔗 读取规划器参数 (当前共 ${curSeg} 段)`
                    : `🔄 读取规划参数 (当前共 ${curSeg} 段)`;
                if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
                if (app.canvas && app.canvas.setDirty) app.canvas.setDirty(true, true);
            }, 1200);
        }
        console.log(`[PJ分段采样] 🔄 用户点击读取规划器参数，当前已生效分段数: ${curSeg}`);
    };

    let keptOne = false;
    const cleanWidgets = [];
    for (const w of node.widgets) {
        if (w.type === "button" || (w.name && (w.name.includes("刷新") || w.name.includes("读取") || w.name.includes("总线同步") || w.name.includes("已刷新") || w.name.includes("已读取")))) {
            if (!keptOne) {
                // 保留且仅保留第一个，重置其名称与回调
                keptOne = true;
                w.name = targetBtnName;
                w.type = "button";
                w.callback = () => handleButtonClick(w);
                cleanWidgets.push(w);
            }
            // 其它多余的历史重复按钮 (#1, #2, ... #11 等) 全部剔除！
        } else {
            cleanWidgets.push(w);
        }
    }
    node.widgets = cleanWidgets;

    // 如果之前一个都没有，才新增一个
    if (!keptOne) {
        const newBtn = node.addWidget("button", targetBtnName, null, () => handleButtonClick(newBtn));
    }
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

    ensureSyncButton(node, true);
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

    ensureSyncButton(node, false);
    syncSamplerSegmentCount(node, false);
}

// 绑定采样器事件监听
function attachSamplerHandlers(node, isFirstPass) {
    if (!node.widgets) return;

    node.color = isFirstPass ? "#134e4a" : "#1e293b";
    node.bgcolor = isFirstPass ? "#042f2e" : "#0f172a";
    if (isFirstPass) {
        if (!node.title || node.title.startsWith("PJ_FirstPass")) node.title = "PJ一采分段重抽采样器 (单段重抽/缓存)";
    } else {
        if (!node.title || node.title.startsWith("PJ_Segment") || node.title.startsWith("PJSegment")) node.title = "PJ二采分段精修采样器 (高清无损/防爆显存)";
    }

    const watchNames = ["split_mode", "num_segments", "run_mode", "chunk_seconds"];
    for (const name of watchNames) {
        const w = node.widgets.find(widget => widget.name === name);
        if (w && !w._pj_attached) {
            w._pj_attached = true;
            const orig = w.callback;
            w.callback = function () {
                const res = orig?.apply(this, arguments);
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
        "custom_cuts",
        "align_51_frames"
    ];
    for (const name of plannerWatchNames) {
        const w = node.widgets?.find(widget => widget.name === name);
        if (w && !w._pj_attached) {
            w._pj_attached = true;
            const orig = w.callback;
            w.callback = function () {
                const res = orig?.apply(this, arguments);
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
    if (info && info.widgets_values && Array.isArray(info.widgets_values)) {
        // 兼容旧版 workflow 结构清洗:
        // 目标 7 个控件: [fps, split_mode, chunk_seconds, overlap_seconds, num_segments, custom_cuts, align_51_frames]
        const modes = ["按秒自动切分", "固定分段数", "自定义切点"];
        if (typeof info.widgets_values[0] === "string" && modes.includes(info.widgets_values[0])) {
            const fpsVal = info.widgets_values[6] ?? 24.0;
            info.widgets_values = [fpsVal, ...info.widgets_values.slice(0, 6)];
            return;
        }
        if (typeof info.widgets_values[0] === "number") {
            while (info.widgets_values.length > 2 && !modes.includes(info.widgets_values[1])) {
                info.widgets_values.splice(1, 1);
            }
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
