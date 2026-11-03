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
    "total_frames": "视频总帧数",

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
    "images": "视频图像",
    "audio": "音频原声",
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
            } else if (w.name === "keep_original_audio") {
                w.options = w.options || {};
                w.options.on = "开启(原音保真)";
                w.options.off = "二采融合";
                w.label_on = "开启(原音保真)";
                w.label_off = "二采融合";
            } else if (w.name === "sync_ref_video") {
                w.options = w.options || {};
                w.options.on = "开启(时序同步)";
                w.options.off = "关闭";
                w.label_on = "开启(时序同步)";
                w.label_off = "关闭";
            } else if (w.name === "align_51_frames") {
                w.options = w.options || {};
                w.options.on = "开启51帧吸附";
                w.options.off = "关闭";
                w.label_on = "开启51帧吸附";
                w.label_off = "关闭";
            } else if (w.name && w.name.endsWith("_lock")) {
                w.options = w.options || {};
                w.options.on = "已锁定";
                w.options.off = "未锁定";
                w.label_on = "已锁定";
                w.label_off = "未锁定";
            }
        }
    }
}

// 获取上游规划器或当前设置的目标分段数
function resolveTargetSegmentCount(node) {
    // 1. 优先检查是否连接了上游 PJ_SegmentPlanner
    if (node.inputs) {
        const planInput = node.inputs.find(i => i.name === "segment_plan");
        if (planInput && planInput.link != null && app.graph) {
            const link = app.graph.links[planInput.link];
            if (link) {
                const originNode = app.graph.getNodeById(link.origin_id);
                if (originNode && originNode.type === "PJ_SegmentPlanner") {
                    if (originNode.pj_cached_n_chunks) {
                        return Math.min(20, Math.max(1, originNode.pj_cached_n_chunks));
                    }
                    const upNumSeg = originNode.widgets?.find(w => w.name === "num_segments")?.value;
                    if (upNumSeg) {
                        return Math.min(20, Math.max(1, parseInt(upNumSeg) || 3));
                    }
                }
            }
        }
    }

    // 2. 读取本节点的 num_segments
    const numSegWidget = node.widgets?.find(w => w.name === "num_segments");
    if (numSegWidget) {
        const val = parseInt(numSegWidget.value);
        if (!isNaN(val) && val >= 1) {
            return Math.min(20, Math.max(1, val));
        }
    }

    return 5; // 默认平滑容纳 5 段
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

    for (let i = 1; i <= 20; i++) {
        const lockW = node.widgets?.find(w => w.name === `seg${i}_lock`);
        const offsetW = node.widgets?.find(w => w.name === `seg${i}_seed_offset`);

        const inRange = i <= maxSeg;

        if (runMode.includes("自定义") || runMode.includes("缺陷秒数")) {
            toggleWidget(node, lockW, false);
            if (isFirstPass) toggleWidget(node, offsetW, inRange);
        } else if (runMode.startsWith(isFirstPass ? "仅重抽第" : "仅精修第")) {
            toggleWidget(node, lockW, false);
            if (isFirstPass) {
                const targetSeg = parseInt(runMode.replace(/\D/g, "")) || 1;
                toggleWidget(node, offsetW, i === targetSeg);
            }
        } else if (runMode.includes("全部")) {
            toggleWidget(node, lockW, false);
            if (isFirstPass) toggleWidget(node, offsetW, inRange);
        } else {
            // 智能缓存模式
            toggleWidget(node, lockW, inRange);
            if (isFirstPass) toggleWidget(node, offsetW, inRange);
        }
    }

    autoResizeNode(node);
}

// 为采样器注入便捷的“刷新分段适配”按钮 (单例且自动清除任何历史多余重复项)
function ensureSyncButton(node, isFirstPass) {
    if (!node.widgets) return;

    let keptOne = false;
    const cleanWidgets = [];
    for (const w of node.widgets) {
        if (w.name && w.name.includes("刷新分段适配")) {
            if (!keptOne) {
                // 保留且仅保留第一个，重置其名称与回调
                keptOne = true;
                w.name = "🔄 刷新分段适配 (按当前分段数自适应)";
                w.callback = () => {
                    syncSamplerSegmentCount(node, isFirstPass);
                };
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
        node.addWidget("button", "🔄 刷新分段适配 (按当前分段数自适应)", null, () => {
            syncSamplerSegmentCount(node, isFirstPass);
        });
    }
}

// ==========================================
// 4. 一采节点与二采节点显隐更新
// ==========================================
function updateFirstPassVisibility(node) {
    if (!node.widgets) return;
    localizeAndSanitizeCommon(node);
    ensureSyncButton(node, true);

    const modeWidget = node.widgets.find(w => w.name === "split_mode");
    const chunkSec = node.widgets.find(w => w.name === "chunk_seconds");
    const overlapSec = node.widgets.find(w => w.name === "overlap_seconds");
    const numSeg = node.widgets.find(w => w.name === "num_segments");
    const customCuts = node.widgets.find(w => w.name === "custom_cuts");

    // 切分模式关联切换
    if (modeWidget) {
        const mode = modeWidget.value;
        if (mode === "按秒自动切分" || mode === "auto_by_seconds") {
            toggleWidget(node, chunkSec, true);
            toggleWidget(node, overlapSec, true);
            toggleWidget(node, numSeg, false);
            toggleWidget(node, customCuts, false);
        } else if (mode === "固定分段数" || mode === "fixed_segments") {
            toggleWidget(node, chunkSec, false);
            toggleWidget(node, overlapSec, true);
            toggleWidget(node, numSeg, true);
            toggleWidget(node, customCuts, false);
        } else if (mode === "自定义切点" || mode === "custom_cuts") {
            toggleWidget(node, chunkSec, false);
            toggleWidget(node, overlapSec, false);
            toggleWidget(node, numSeg, false);
            toggleWidget(node, customCuts, true);
        }
    }

    syncSamplerSegmentCount(node, true);
}

function updateSecondPassVisibility(node) {
    if (!node.widgets) return;
    localizeAndSanitizeCommon(node);
    ensureSyncButton(node, false);

    const modeWidget = node.widgets.find(w => w.name === "split_mode");
    const chunkSec = node.widgets.find(w => w.name === "chunk_seconds");
    const overlapSec = node.widgets.find(w => w.name === "overlap_seconds");
    const numSeg = node.widgets.find(w => w.name === "num_segments");
    const customCuts = node.widgets.find(w => w.name === "custom_cuts");

    if (modeWidget) {
        const mode = modeWidget.value;
        if (mode === "按秒自动切分" || mode === "auto_by_seconds") {
            toggleWidget(node, chunkSec, true);
            toggleWidget(node, overlapSec, true);
            toggleWidget(node, numSeg, false);
            toggleWidget(node, customCuts, false);
        } else if (mode === "固定分段数" || mode === "fixed_segments") {
            toggleWidget(node, chunkSec, false);
            toggleWidget(node, overlapSec, true);
            toggleWidget(node, numSeg, true);
            toggleWidget(node, customCuts, false);
        } else if (mode === "自定义切点" || mode === "custom_cuts") {
            toggleWidget(node, chunkSec, false);
            toggleWidget(node, overlapSec, false);
            toggleWidget(node, numSeg, false);
            toggleWidget(node, customCuts, true);
        }
    }

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

    if (modeWidget && !modeWidget._pj_attached) {
        modeWidget._pj_attached = true;
        const orig = modeWidget.callback;
        modeWidget.callback = function () {
            const res = orig?.apply(this, arguments);
            updatePlannerWidgets();
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

        widget = node.addDOMWidget("pj_planner_dashboard", "dashboard", container, {
            serialize: false, hideOnZoom: false
        });
    }

    const container = widget.element;
    container.innerHTML = "";

    // 1. 顶部时序统计横幅
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
        <span style="color: #38bdf8;">🎬 总时长: ${data.effective_seconds}s (${data.total_frames}帧)</span>
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
        tipEl.style.background = "rgba(59, 130, 246, 0.1)";
        tipEl.style.border = "1px dashed #3b82f6";
        tipEl.style.borderRadius = "6px";
        tipEl.style.color = "#93c5fd";
        tipEl.style.fontSize = "11px";
        tipEl.style.lineHeight = "1.5";
        tipEl.innerHTML = `💡 <b>提示</b>：连接上方 <code>images</code> 与 <code>audio</code> 端口后，此处将自动渲染内置试听播放器，支持精准截取任意分段一键试听！`;
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
    segListContainer.style.maxHeight = "200px";
    segListContainer.style.overflowY = "auto";
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
// 6. 注册 ComfyUI 前端扩展
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

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
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
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
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
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
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
