export function modeTabs(node, name, label, choices, selected, onSelect) {
    if (!node.addDOMWidget) return null;
    let widget = node.widgets.find(w => w.name === name);
    if (!widget) {
        const root = document.createElement("div");
        root.setAttribute("role", "tablist");
        root.setAttribute("aria-label", label);
        Object.assign(root.style, {display: "grid", gridAutoRows: "28px", alignContent: "start", gap: "4px",
            width: "100%", padding: "4px 0", boxSizing: "border-box"});
        widget = node.addDOMWidget(name, "pj_tabs", root, {
            serialize: false, hideOnZoom: false, margin: 0,
            getMinHeight: () => widget._pjHeight || 36,
            getMaxHeight: () => widget._pjHeight || 36,
            getHeight: () => widget._pjHeight || 36,
        });
        widget.serialize = false;
        widget._pjRoot = root;
        widget.computeSize = width => [width, widget._pjHeight || 36];
        // Only the nonserialized tab bar moves; backend widgets retain their order.
        node.widgets.splice(node.widgets.indexOf(widget), 1);
        const position = node.widgets.filter(w => w.type === "pj_tabs").length;
        node.widgets.splice(position, 0, widget);
    }
    widget._pjSelect = onSelect;
    const columns = choices.length > 4 ? 3 : choices.length;
    widget._pjRoot.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
    widget._pjHeight = Math.ceil(choices.length / columns) * 32 + 4;
    const signature = JSON.stringify(choices);
    if (widget._pjChoices !== signature) {
        widget._pjChoices = signature;
        widget._pjRoot.replaceChildren();
        for (const [value, text] of choices) {
            const button = document.createElement("button");
            button.type = "button";
            button.setAttribute("role", "tab");
            button.dataset.value = value;
            button.textContent = text;
            button.title = text;
            Object.assign(button.style, {minWidth: "0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                boxSizing: "border-box", padding: "4px 6px", border: "1px solid var(--border-color, #666)",
                borderRadius: "4px", background: "var(--comfy-input-bg, #222)", color: "var(--input-text, #ddd)",
                font: "12px sans-serif", cursor: "pointer"});
            button.addEventListener("click", () => {
                node.graph?.beforeChange?.();
                widget._pjSelect(value);
                node.graph?.afterChange?.();
            });
            button.addEventListener("keydown", event => {
                const buttons = Array.from(widget._pjRoot.children);
                let index = buttons.indexOf(button);
                if (event.key === "ArrowRight") index = (index + 1) % buttons.length;
                else if (event.key === "ArrowLeft") index = (index + buttons.length - 1) % buttons.length;
                else if (event.key === "Home") index = 0;
                else if (event.key === "End") index = buttons.length - 1;
                else return;
                event.preventDefault();
                event.stopPropagation();
                buttons[index].focus();
                buttons[index].click();
            });
            widget._pjRoot.appendChild(button);
        }
    }
    for (const button of widget._pjRoot.children) {
        const active = button.dataset.value === selected;
        button.setAttribute("aria-selected", String(active));
        button.tabIndex = active ? 0 : -1;
        button.style.fontWeight = active ? "bold" : "normal";
        button.style.background = active ? "var(--comfy-menu-bg, #353535)" : "var(--comfy-input-bg, #222)";
        button.style.borderColor = active ? "var(--input-text, #ddd)" : "var(--border-color, #666)";
    }
    return widget;
}
