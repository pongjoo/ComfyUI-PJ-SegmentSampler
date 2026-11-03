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

function updateWidgetVisibility(node) {
    if (!node.widgets) return;
    const modeWidget = node.widgets.find(w => w.name === "split_mode");
    if (!modeWidget) return;

    const mode = modeWidget.value;
    const chunkSecWidget = node.widgets.find(w => w.name === "chunk_seconds");
    const overlapSecWidget = node.widgets.find(w => w.name === "overlap_seconds");
    const numSegWidget = node.widgets.find(w => w.name === "num_segments");
    const customCutsWidget = node.widgets.find(w => w.name === "custom_cuts");

    if (mode === "auto_by_seconds") {
        toggleWidget(node, chunkSecWidget, true);
        toggleWidget(node, overlapSecWidget, true);
        toggleWidget(node, numSegWidget, false);
        toggleWidget(node, customCutsWidget, false);
    } else if (mode === "fixed_segments") {
        toggleWidget(node, chunkSecWidget, false);
        toggleWidget(node, overlapSecWidget, true);
        toggleWidget(node, numSegWidget, true);
        toggleWidget(node, customCutsWidget, false);
    } else if (mode === "custom_cuts") {
        toggleWidget(node, chunkSecWidget, false);
        toggleWidget(node, overlapSecWidget, false);
        toggleWidget(node, numSegWidget, false);
        toggleWidget(node, customCutsWidget, true);
    }

    [30, 100, 250].forEach(delay => {
        setTimeout(() => {
            if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
                const sz = node.computeSize();
                node.setSize([node.size[0], sz[1] + 16]);
                if (node.graph && typeof node.graph.setDirty === "function") {
                    node.graph.setDirty(true, true);
                }
            }
        }, delay);
    });
}

function attachHandler(node) {
    if (!node.widgets) return;
    const modeWidget = node.widgets.find(w => w.name === "split_mode");
    if (modeWidget && !modeWidget._pj_attached) {
        modeWidget._pj_attached = true;
        const origCallback = modeWidget.callback;
        modeWidget.callback = function (val) {
            const res = origCallback?.apply(this, arguments);
            updateWidgetVisibility(node);
            return res;
        };
    }
    setTimeout(() => {
        updateWidgetVisibility(node);
    }, 50);
}

app.registerExtension({
    name: "PJ.SegmentSampler.DynamicWidgetToggle",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "PJ_SegmentSampler" || nodeData.name === "PJSegmentSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                attachHandler(this);
                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                attachHandler(this);
                return r;
            };
        }
    },
    async nodeCreated(node) {
        if (node.comfyClass === "PJ_SegmentSampler" || node.comfyClass === "PJSegmentSampler") {
            attachHandler(node);
        }
    },
    async loadedGraphNode(node) {
        if (node.comfyClass === "PJ_SegmentSampler" || node.comfyClass === "PJSegmentSampler") {
            attachHandler(node);
        }
    }
});
