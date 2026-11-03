import { app } from "../../scripts/app.js";

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

    // 一采专有功能
    "sync_ref_video": "同步参考视频",
    "base_seed": "基础随机种子",
    "seg1_seed_offset": "第1段种子偏移",
    "seg2_seed_offset": "第2段种子偏移",
    "seg3_seed_offset": "第3段种子偏移",
    "seg4_seed_offset": "第4段种子偏移",

    // 运行重抽与缓存锁
    "run_mode": "运行重抽模式",
    "seg1_lock": "锁定第1段",
    "seg2_lock": "锁定第2段",
    "seg3_lock": "锁定第3段",
    "seg4_lock": "锁定第4段",

    // 算法与系统控制
    "blend_mode": "平滑过渡算法",
    "keep_original_audio": "保留一采原音频",
    "unload_model_after_sampling": "释放显存保卫VAE",
    "fps": "视频帧率"
};

const INPUT_LABELS = {
    "noise": "随机噪声",
    "guider": "引导器",
    "sampler": "采样算法",
    "sigmas": "调度步长",
    "latent_image": "输入潜空间"
};

const OUTPUT_LABELS = {
    "output": "完整潜空间",
    "denoised_output": "去噪潜空间"
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
                const targetW = Math.max(node.size[0] || 0, 380);
                node.setSize([targetW, sz[1] + 20]);
                if (node.graph && typeof node.graph.setDirty === "function") {
                    node.graph.setDirty(true, true);
                }
            }
        }, delay);
    });
}

// ==========================================
// 3. 全量中文本地化与旧版数据清洗
// ==========================================
function localizeAndSanitizeNode(node, isFirstPass) {
    if (!node) return;

    // 1. 配色与标题规范
    if (isFirstPass) {
        node.color = "#134e4a"; // 现代雅致青 (Teal)
        node.bgcolor = "#042f2e";
        if (!node.title || node.title === "PJ_FirstPass_SegmentSampler") {
            node.title = "PJ一采分段重抽采样器 (单段重抽/缓存)";
        }
    } else {
        node.color = "#1e293b"; // 深邃石青蓝 (Slate Navy)
        node.bgcolor = "#0f172a";
        if (!node.title || node.title === "PJ_SegmentSampler" || node.title === "PJ分段采样器 (二采无缝精修+自动释放显存)") {
            node.title = "PJ二采分段精修采样器 (高清无损/防爆显存)";
        }
    }

    // 2. 汉化输入插槽 (Input Sockets)
    if (node.inputs) {
        for (const input of node.inputs) {
            if (INPUT_LABELS[input.name]) {
                input.label = INPUT_LABELS[input.name];
            }
        }
    }

    // 3. 汉化输出插槽 (Output Sockets)
    if (node.outputs) {
        for (const output of node.outputs) {
            if (OUTPUT_LABELS[output.name]) {
                output.label = OUTPUT_LABELS[output.name];
            }
        }
    }

    // 4. 汉化控件名称 (左侧label) 及清洗英文/布尔值数据
    if (node.widgets) {
        for (const w of node.widgets) {
            // 设定控件左侧中文名
            if (WIDGET_LABELS[w.name]) {
                w.label = WIDGET_LABELS[w.name];
            }

            // 清洗与汉化 split_mode
            if (w.name === "split_mode") {
                if (w.value === "auto_by_seconds") w.value = "按秒自动切分";
                else if (w.value === "fixed_segments") w.value = "固定分段数";
                else if (w.value === "custom" || w.value === "custom_cuts") w.value = "自定义切点";
                if (w.options) {
                    w.options.values = ["按秒自动切分", "固定分段数", "自定义切点"];
                }
            }

            // 清洗与汉化 run_mode (防止旧工作流残留布尔值 true/false)
            if (w.name === "run_mode") {
                if (w.value === true || w.value === "true" || w.value === false || w.value === "false" || typeof w.value === "boolean") {
                    w.value = isFirstPass ? "智能缓存(仅重跑未锁定分段)" : "全部精修(刷新全部缓存)";
                }
                if (w.options) {
                    w.options.values = isFirstPass ? [
                        "智能缓存(仅重跑未锁定分段)",
                        "全部重新采样(刷新全部缓存)",
                        "仅重抽第1段",
                        "仅重抽第2段",
                        "仅重抽第3段",
                        "仅重抽第4段"
                    ] : [
                        "智能缓存(仅重跑未锁定分段)",
                        "全部精修(刷新全部缓存)",
                        "仅精修第1段",
                        "仅精修第2段",
                        "仅精修第3段",
                        "仅精修第4段"
                    ];
                }
            }

            // 清洗与汉化 blend_mode
            if (w.name === "blend_mode") {
                if (w.value === "cosine") w.value = "余弦平滑(推荐)";
                else if (w.value === "linear") w.value = "线性过渡";
                else if (w.value === "hard" || w.value === "hard_cut") w.value = "硬切";
                if (w.options) {
                    w.options.values = ["余弦平滑(推荐)", "线性过渡", "硬切"];
                }
            }

            // 彻底解决开关按钮 (Toggle) 文本过长导致的重叠压字 bug
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

// ==========================================
// 4. 一采节点动态交互与显隐控制
// ==========================================
function updateFirstPassVisibility(node) {
    if (!node.widgets) return;

    localizeAndSanitizeNode(node, true);

    const modeWidget = node.widgets.find(w => w.name === "split_mode");
    const runModeWidget = node.widgets.find(w => w.name === "run_mode");

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

    if (runModeWidget) {
        const runMode = runModeWidget.value;
        const seg1Lock = node.widgets.find(w => w.name === "seg1_lock");
        const seg1Offset = node.widgets.find(w => w.name === "seg1_seed_offset");
        const seg2Lock = node.widgets.find(w => w.name === "seg2_lock");
        const seg2Offset = node.widgets.find(w => w.name === "seg2_seed_offset");
        const seg3Lock = node.widgets.find(w => w.name === "seg3_lock");
        const seg3Offset = node.widgets.find(w => w.name === "seg3_seed_offset");
        const seg4Lock = node.widgets.find(w => w.name === "seg4_lock");
        const seg4Offset = node.widgets.find(w => w.name === "seg4_seed_offset");

        if (runMode === "仅重抽第1段") {
            toggleWidget(node, seg1Lock, false);
            toggleWidget(node, seg1Offset, true);
            toggleWidget(node, seg2Lock, false);
            toggleWidget(node, seg2Offset, false);
            toggleWidget(node, seg3Lock, false);
            toggleWidget(node, seg3Offset, false);
            toggleWidget(node, seg4Lock, false);
            toggleWidget(node, seg4Offset, false);
        } else if (runMode === "仅重抽第2段") {
            toggleWidget(node, seg1Lock, false);
            toggleWidget(node, seg1Offset, false);
            toggleWidget(node, seg2Lock, false);
            toggleWidget(node, seg2Offset, true);
            toggleWidget(node, seg3Lock, false);
            toggleWidget(node, seg3Offset, false);
            toggleWidget(node, seg4Lock, false);
            toggleWidget(node, seg4Offset, false);
        } else if (runMode === "仅重抽第3段") {
            toggleWidget(node, seg1Lock, false);
            toggleWidget(node, seg1Offset, false);
            toggleWidget(node, seg2Lock, false);
            toggleWidget(node, seg2Offset, false);
            toggleWidget(node, seg3Lock, false);
            toggleWidget(node, seg3Offset, true);
            toggleWidget(node, seg4Lock, false);
            toggleWidget(node, seg4Offset, false);
        } else if (runMode === "全部重新采样(刷新全部缓存)") {
            toggleWidget(node, seg1Lock, false);
            toggleWidget(node, seg1Offset, true);
            toggleWidget(node, seg2Lock, false);
            toggleWidget(node, seg2Offset, true);
            toggleWidget(node, seg3Lock, false);
            toggleWidget(node, seg3Offset, true);
            toggleWidget(node, seg4Lock, false);
            toggleWidget(node, seg4Offset, true);
        } else {
            // 智能缓存模式
            toggleWidget(node, seg1Lock, true);
            toggleWidget(node, seg1Offset, true);
            toggleWidget(node, seg2Lock, true);
            toggleWidget(node, seg2Offset, true);
            toggleWidget(node, seg3Lock, true);
            toggleWidget(node, seg3Offset, true);
            toggleWidget(node, seg4Lock, true);
            toggleWidget(node, seg4Offset, true);
        }
    }

    autoResizeNode(node);
}

// ==========================================
// 5. 二采节点动态交互与显隐控制
// ==========================================
function updateSecondPassVisibility(node) {
    if (!node.widgets) return;

    localizeAndSanitizeNode(node, false);

    const modeWidget = node.widgets.find(w => w.name === "split_mode");
    const runModeWidget = node.widgets.find(w => w.name === "run_mode");

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

    if (runModeWidget) {
        const runMode = runModeWidget.value;
        const seg1Lock = node.widgets.find(w => w.name === "seg1_lock");
        const seg2Lock = node.widgets.find(w => w.name === "seg2_lock");
        const seg3Lock = node.widgets.find(w => w.name === "seg3_lock");
        const seg4Lock = node.widgets.find(w => w.name === "seg4_lock");

        if (runMode === "全部精修(刷新全部缓存)") {
            toggleWidget(node, seg1Lock, false);
            toggleWidget(node, seg2Lock, false);
            toggleWidget(node, seg3Lock, false);
            toggleWidget(node, seg4Lock, false);
        } else if (runMode === "仅精修第1段" || runMode === "仅精修第2段" || runMode === "仅精修第3段" || runMode === "仅精修第4段") {
            toggleWidget(node, seg1Lock, false);
            toggleWidget(node, seg2Lock, false);
            toggleWidget(node, seg3Lock, false);
            toggleWidget(node, seg4Lock, false);
        } else {
            // 智能缓存模式
            toggleWidget(node, seg1Lock, true);
            toggleWidget(node, seg2Lock, true);
            toggleWidget(node, seg3Lock, true);
            toggleWidget(node, seg4Lock, true);
        }
    }

    autoResizeNode(node);
}

// ==========================================
// 6. 事件绑定与生命周期接入
// ==========================================
function attachFirstPassHandlers(node) {
    if (!node.widgets) return;
    localizeAndSanitizeNode(node, true);

    const modeWidget = node.widgets.find(w => w.name === "split_mode");
    if (modeWidget && !modeWidget._pj_attached) {
        modeWidget._pj_attached = true;
        const origCallback = modeWidget.callback;
        modeWidget.callback = function () {
            const res = origCallback?.apply(this, arguments);
            updateFirstPassVisibility(node);
            return res;
        };
    }

    const runModeWidget = node.widgets.find(w => w.name === "run_mode");
    if (runModeWidget && !runModeWidget._pj_attached) {
        runModeWidget._pj_attached = true;
        const origCallback = runModeWidget.callback;
        runModeWidget.callback = function () {
            const res = origCallback?.apply(this, arguments);
            updateFirstPassVisibility(node);
            return res;
        };
    }

    setTimeout(() => {
        updateFirstPassVisibility(node);
    }, 50);
}

function attachSecondPassHandlers(node) {
    if (!node.widgets) return;
    localizeAndSanitizeNode(node, false);

    const modeWidget = node.widgets.find(w => w.name === "split_mode");
    if (modeWidget && !modeWidget._pj_attached) {
        modeWidget._pj_attached = true;
        const origCallback = modeWidget.callback;
        modeWidget.callback = function () {
            const res = origCallback?.apply(this, arguments);
            updateSecondPassVisibility(node);
            return res;
        };
    }

    const runModeWidget = node.widgets.find(w => w.name === "run_mode");
    if (runModeWidget && !runModeWidget._pj_attached) {
        runModeWidget._pj_attached = true;
        const origCallback = runModeWidget.callback;
        runModeWidget.callback = function () {
            const res = origCallback?.apply(this, arguments);
            updateSecondPassVisibility(node);
            return res;
        };
    }

    setTimeout(() => {
        updateSecondPassVisibility(node);
    }, 50);
}

// 检查并刷新画布上所有目标节点
function scanAndLocalizeAllNodes() {
    if (!app.graph || !app.graph._nodes) return;
    for (const node of app.graph._nodes) {
        if (node.type === "PJ_FirstPass_SegmentSampler") {
            attachFirstPassHandlers(node);
        } else if (node.type === "PJ_SegmentSampler" || node.type === "PJSegmentSampler") {
            attachSecondPassHandlers(node);
        }
    }
}

// 注册 ComfyUI 前端扩展
app.registerExtension({
    name: "PJ.SegmentSampler.FullChineseLocalization",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "PJ_FirstPass_SegmentSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                attachFirstPassHandlers(this);
                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                attachFirstPassHandlers(this);
                return r;
            };
        } else if (nodeData.name === "PJ_SegmentSampler" || nodeData.name === "PJSegmentSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                attachSecondPassHandlers(this);
                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                attachSecondPassHandlers(this);
                return r;
            };
        }
    },
    async nodeCreated(node) {
        if (node.comfyClass === "PJ_FirstPass_SegmentSampler") {
            attachFirstPassHandlers(node);
        } else if (node.comfyClass === "PJ_SegmentSampler" || node.comfyClass === "PJSegmentSampler") {
            attachSecondPassHandlers(node);
        }
    },
    async afterConfigureGraph() {
        scanAndLocalizeAllNodes();
    },
    setup() {
        // 画布初次就绪后延时全面排查一次，彻底纠正所有已打开的旧画布
        setTimeout(scanAndLocalizeAllNodes, 500);
        setTimeout(scanAndLocalizeAllNodes, 1500);
    }
});
