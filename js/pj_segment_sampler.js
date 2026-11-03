import { app } from "../../scripts/app.js";

const HIDDEN_TAG = "pj_hidden";

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

function updateFirstPassVisibility(node) {
    if (!node.widgets) return;

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

    // Dynamic slot control based on run_mode
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
            // 智能缓存模式: show all locks and offsets
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

function updateSecondPassVisibility(node) {
    if (!node.widgets) return;

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
        } else {
            toggleWidget(node, seg1Lock, true);
            toggleWidget(node, seg2Lock, true);
            toggleWidget(node, seg3Lock, true);
            toggleWidget(node, seg4Lock, true);
        }
    }

    autoResizeNode(node);
}

function autoResizeNode(node) {
    [20, 100, 250].forEach(delay => {
        setTimeout(() => {
            if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
                const sz = node.computeSize();
                node.setSize([Math.max(node.size[0], 340), sz[1] + 20]);
                if (node.graph && typeof node.graph.setDirty === "function") {
                    node.graph.setDirty(true, true);
                }
            }
        }, delay);
    });
}

function attachFirstPassHandlers(node) {
    if (!node.widgets) return;
    node.color = "#134e4a"; // Modern Teal
    node.bgcolor = "#042f2e";

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
    node.color = "#1e293b"; // Modern Slate Navy
    node.bgcolor = "#0f172a";

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

app.registerExtension({
    name: "PJ.SegmentSampler.DynamicWidgetToggle",
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
    }
});
