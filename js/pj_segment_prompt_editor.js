function parseSegmentPrompts(text) {
    let prompts;
    try {
        prompts = JSON.parse(text || "{}");
    } catch (error) {
        throw new Error(`JSON 格式错误：${error.message}`);
    }
    if (!prompts || Array.isArray(prompts) || typeof prompts !== "object") {
        throw new Error('请使用段号组成的 JSON 对象，例如 {"2":"人物转身"}。');
    }
    const seen = new Set();
    for (const [key, entry] of Object.entries(prompts)) {
        if (!/^[0-9]+$/.test(key) || BigInt(key) < 1n || seen.has(String(BigInt(key)))) {
            throw new Error("段号需要从 1 开始，不能重复。");
        }
        seen.add(String(BigInt(key)));
        if (typeof entry === "string") continue;
        if (!entry || Array.isArray(entry) || typeof entry !== "object" ||
            Object.keys(entry).some(name => !["prompt", "transition", "audio_transition", "dialogue"].includes(name)) ||
            typeof (entry.prompt === undefined ? "" : entry.prompt) !== "string" ||
            !["continue", "cut"].includes(entry.transition === undefined ? "continue" : entry.transition) ||
            !["continue", "reset"].includes(entry.audio_transition === undefined ? "continue" : entry.audio_transition)) {
            throw new Error(`第 ${key} 段需要提示词文本，或包含 prompt、transition、audio_transition、dialogue 的设置对象。`);
        }
        if ("dialogue" in entry && (!Array.isArray(entry.dialogue) || entry.dialogue.some(line =>
            !line || Object.keys(line).sort().join(",") !== "end,speaker,start,text" ||
            typeof line.speaker !== "string" || typeof line.text !== "string" ||
            !Number.isFinite(line.start) || !Number.isFinite(line.end)))) {
            throw new Error(`第 ${key} 段 dialogue 需要台词数组：speaker、text 为文本，start、end 为成片秒数。`);
        }
    }
    return prompts;
}

function segmentKeys(prompts) {
    return Object.keys(prompts).sort((a, b) => BigInt(a) < BigInt(b) ? -1 : BigInt(a) > BigInt(b) ? 1 : 0);
}

function promptInputConnected(node) {
    return node.inputs?.some(input =>
        (input.name === "segment_prompts" || input.widget?.name === "segment_prompts") && input.link != null);
}

function setPromptText(node, text) {
    if (promptInputConnected(node)) return false;
    const widget = node.widgets.find(w => w.name === "segment_prompts");
    if (widget.value === text) return true;
    node.graph?.beforeChange?.();
    widget.value = text;
    if (widget.inputEl) widget.inputEl.value = text;
    widget.callback?.(text);
    node.graph?.afterChange?.();
    node.setDirtyCanvas?.(true, true);
    return true;
}

function editorStyles() {
    if (document.getElementById("pj-prompt-editor-style")) return;
    const style = document.createElement("style");
    style.id = "pj-prompt-editor-style";
    style.textContent = `
        .pj-prompt-editor { box-sizing: border-box; width: min(820px, calc(100vw - 32px));
            max-height: calc(100dvh - 32px); padding: 20px; border: 1px solid var(--border-color, #666);
            border-radius: 8px; color: var(--input-text, #ddd); background: var(--comfy-menu-bg, #252525);
            font: 14px sans-serif; overflow: auto; }
        .pj-prompt-editor::backdrop { background: #0008; }
        .pj-prompt-editor * { box-sizing: border-box; }
        .pj-prompt-editor [hidden] { display: none !important; }
        .pj-prompt-editor header, .pj-prompt-editor nav, .pj-prompt-editor .pj-prompt-controls {
            display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
        .pj-prompt-editor header { justify-content: space-between; }
        .pj-prompt-editor h2 { margin: 0; font-size: 18px; }
        .pj-prompt-editor button, .pj-prompt-editor select { font: inherit; min-height: 32px;
            padding: 4px 12px; color: var(--input-text, #ddd); background: var(--comfy-input-bg, #333);
            border: 1px solid var(--border-color, #666); border-radius: 4px; cursor: pointer; }
        .pj-prompt-editor button[aria-pressed="true"] { border-color: var(--input-text, #ddd); font-weight: bold; }
        .pj-prompt-editor :focus-visible { outline: 2px solid var(--input-text, #ddd); outline-offset: 2px; }
        .pj-prompt-editor textarea { display: block; width: 100%; height: clamp(180px, 48dvh, 480px);
            resize: vertical; padding: 12px; color: var(--input-text, #ddd); background: var(--comfy-input-bg, #171717);
            border: 1px solid var(--border-color, #666); border-radius: 4px; line-height: 1.55; font-size: 14px; }
        .pj-prompt-editor label { display: block; margin-bottom: 8px; }
        .pj-prompt-editor .pj-dialogue-line { display: grid; grid-template-columns: 1fr 2fr 80px 80px auto; gap: 6px; margin: 8px 0; }
        .pj-prompt-editor .pj-dialogue-line button { align-self: center; }
        .pj-prompt-editor input { width: 100%; min-width: 0; padding: 6px; font: inherit;
            color: var(--input-text, #ddd); background: var(--comfy-input-bg, #333); border: 1px solid var(--border-color, #666); }
        .pj-prompt-editor footer { margin-top: 12px; font-size: 12px; }
        .pj-prompt-editor [role="alert"] { margin-top: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
        .pj-prompt-editor button:disabled, .pj-prompt-editor textarea:disabled { opacity: .55; cursor: default; }
        @media (max-width: 600px) { .pj-prompt-editor { padding: 12px; }
            .pj-prompt-editor .pj-dialogue-line { grid-template-columns: 1fr 1fr; } }
    `;
    document.head.appendChild(style);
}

function openPromptEditor(node) {
    if (node._pjPromptEditor) {
        node._pjPromptEditor.refresh();
        node._pjPromptEditor.dialog.focus();
        return;
    }
    editorStyles();
    const element = (tag, text, parent) => {
        const result = document.createElement(tag);
        if (text) result.textContent = text;
        parent?.appendChild(result);
        return result;
    };
    const button = (text, parent, action) => {
        const result = element("button", text, parent);
        result.type = "button";
        result.addEventListener("click", action);
        return result;
    };
    const read = () => node.widgets.find(w => w.name === "segment_prompts").value || "";
    const dialog = element("dialog");
    dialog.className = "pj-prompt-editor";
    dialog.setAttribute("aria-label", "分段提示词编辑器");
    const header = element("header", "", dialog);
    element("h2", "分段提示词", header);
    button("关闭", header, () => dialog.close());
    const notice = element("p", "外部输入已连接，请在上游编辑提示词。", dialog);
    const body = element("div", "", dialog);
    const nav = element("nav", "", body);
    nav.setAttribute("aria-label", "提示词显示方式");
    const rawButton = button("完整 JSON", nav, () => switchView("json"));
    const segmentButton = button("分段编辑", nav, () => switchView("segments"));
    const rawPanel = element("label", "各段补充提示词（JSON）", body);
    const raw = element("textarea", "", rawPanel);
    raw.setAttribute("aria-label", "各段补充提示词 JSON");
    raw.placeholder = '{\n  "1": "第一段提示词",\n  "2": "第二段提示词"\n}';
    raw.spellcheck = false;
    const segments = element("div", "", body);
    const controls = element("div", "", segments);
    controls.className = "pj-prompt-controls";
    const select = element("select", "", controls);
    select.setAttribute("aria-label", "当前分段");
    button("添加分段", controls, () => {
        const prompts = parseSegmentPrompts(read());
        const keys = segmentKeys(prompts);
        selected = keys.length ? String(BigInt(keys.at(-1)) + 1n) : "1";
        prompts[selected] = "";
        setPromptText(node, JSON.stringify(prompts, null, 2));
        refresh();
        text.focus();
    });
    const label = element("label", "本段补充提示词", segments);
    const text = element("textarea", "", label);
    text.setAttribute("aria-label", "本段补充提示词");
    text.spellcheck = false;
    text.style.height = "180px";
    const settings = element("div", "", segments);
    settings.className = "pj-prompt-controls";
    const choose = (title, choices) => {
        const label = element("label", title, settings);
        const control = element("select", "", label);
        control.setAttribute("aria-label", title);
        for (const [value, title] of choices) {
            const option = element("option", title, control); option.value = value;
        }
        return control;
    };
    const transition = choose("镜头衔接", [["continue", "连续镜头"], ["cut", "切换镜头"]]);
    transition.title = "重新生成时生效；局部重绘沿用已有镜头边界。";
    const audioTransition = choose("声音衔接", [["continue", "承接前段"], ["reset", "本段重新生成"]]);
    const speech = choose("本段对白", [["prompt", "按提示词"], ["none", "无对白"], ["lines", "指定台词"]]);
    const dialogue = element("div", "", segments);
    element("p", "对白时间填写成片中的秒数。", dialogue);
    const lines = element("div", "", dialogue);
    button("添加台词", dialogue, () => {
        editEntry(entry => { entry.dialogue ||= []; entry.dialogue.push({speaker: "", text: "", start: 0, end: 1}); });
        refresh();
    });
    const error = element("div", "", body);
    error.setAttribute("role", "alert");
    element("footer", "修改自动同步到节点；全局提示词保持独立。", body);
    let view = node.properties?.pj_prompt_view === "segments" ? "segments" : "json";
    let selected = "";

    function editEntry(change) {
        const prompts = parseSegmentPrompts(read());
        if (!selected || !(selected in prompts) || promptInputConnected(node)) return;
        let entry = prompts[selected];
        if (typeof entry === "string") entry = {prompt: entry};
        change(entry);
        prompts[selected] = entry;
        setPromptText(node, JSON.stringify(prompts, null, 2));
    }

    function renderDialogue(entry) {
        lines.replaceChildren();
        (entry.dialogue || []).forEach((line, index) => {
            const row = element("div", "", lines); row.className = "pj-dialogue-line";
            for (const [field, title] of [["speaker", "说话者"], ["text", "台词"], ["start", "开始秒数"], ["end", "结束秒数"]]) {
                const label = element("label", title, row);
                const input = element("input", "", label);
                input.setAttribute("aria-label", `第 ${index + 1} 句${title}`);
                input.value = line[field];
                if (field === "speaker") input.placeholder = "<Subject 1> (S1)";
                if (field === "text") input.placeholder = "[Chinese] 你好。";
                const numeric = field === "start" || field === "end";
                input.type = numeric ? "number" : "text";
                if (numeric) input.step = "0.1";
                input.addEventListener("input", () => {
                    if (numeric && !Number.isFinite(input.valueAsNumber)) return;
                    editEntry(entry => { entry.dialogue[index][field] = numeric ? input.valueAsNumber : input.value; });
                });
            }
            button("删除", row, () => { editEntry(entry => entry.dialogue.splice(index, 1)); refresh(); });
        });
    }

    function showView() {
        rawPanel.hidden = view !== "json";
        segments.hidden = view !== "segments";
        rawButton.setAttribute("aria-pressed", String(view === "json"));
        segmentButton.setAttribute("aria-pressed", String(view === "segments"));
    }

    function refresh() {
        const connected = promptInputConnected(node);
        notice.hidden = !connected;
        body.hidden = connected;
        if (connected) return;
        raw.value = read();
        error.textContent = "";
        if (view === "segments") {
            try {
                const prompts = parseSegmentPrompts(read());
                const keys = segmentKeys(prompts);
                if (!keys.includes(selected)) selected = keys[0] || "";
                select.replaceChildren();
                for (const key of keys) {
                    const option = element("option", `第 ${key} 段`, select);
                    option.value = key;
                }
                if (!keys.length) element("option", "暂无分段", select);
                select.value = selected;
                select.disabled = !keys.length;
                const value = prompts[selected] ?? "";
                const entry = typeof value === "string" ? {prompt: value} : value;
                text.value = entry.prompt || "";
                text.disabled = !keys.length;
                settings.hidden = !keys.length;
                transition.value = entry.transition || "continue";
                audioTransition.value = entry.audio_transition || "continue";
                speech.value = !("dialogue" in entry) ? "prompt" : entry.dialogue.length ? "lines" : "none";
                dialogue.hidden = speech.value !== "lines" || !keys.length;
                renderDialogue(entry);
            } catch (err) {
                view = "json";
                error.textContent = err.message;
            }
        }
        showView();
    }

    function switchView(next) {
        view = next;
        refresh();
        node.properties ||= {};
        node.properties.pj_prompt_view = view;
    }

    raw.addEventListener("input", () => {
        setPromptText(node, raw.value);
        error.textContent = "";
    });
    select.addEventListener("change", () => {
        selected = select.value;
        refresh();
    });
    text.addEventListener("input", () => {
        const prompts = parseSegmentPrompts(read());
        if (typeof prompts[selected] === "string") prompts[selected] = text.value;
        else prompts[selected].prompt = text.value;
        setPromptText(node, JSON.stringify(prompts, null, 2));
    });
    transition.addEventListener("change", () => editEntry(entry => { entry.transition = transition.value; }));
    audioTransition.addEventListener("change", () => editEntry(entry => { entry.audio_transition = audioTransition.value; }));
    speech.addEventListener("change", () => {
        editEntry(entry => {
            if (speech.value === "prompt") delete entry.dialogue;
            else entry.dialogue = speech.value === "none" ? [] : [{speaker: "", text: "", start: 0, end: 1}];
        });
        refresh();
    });
    dialog.addEventListener("close", () => {
        dialog.remove();
        delete node._pjPromptEditor;
    }, {once: true});
    node._pjPromptEditor = {dialog, refresh};
    document.body.appendChild(dialog);
    refresh();
    dialog.showModal();
}

export function setupSegmentPromptEditor(node) {
    if (!node.widgets?.some(w => w.name === "segment_prompts")) return;
    if (!node.widgets.some(w => w.name === "pj_prompt_editor")) {
        const control = node.addWidget("button", "编辑分段提示词", null, () => openPromptEditor(node), {serialize: false});
        control.name = "pj_prompt_editor";
        control.label = "编辑分段提示词";
        control.serialize = false;
        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
            this._pjPromptEditor?.dialog.close();
            return onRemoved?.apply(this, arguments);
        };
        const onConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            this._pjPromptEditor?.refresh();
            return result;
        };
    }
    node._pjPromptEditor?.refresh();
}
