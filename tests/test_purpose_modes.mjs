import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import vm from 'node:vm';

const source = ['pj_segment_prompt_editor.js', 'pj_mode_tabs.js', 'pj_segment_sampler.js']
    .map(name => readFileSync(new URL(`../js/${name}`, import.meta.url), 'utf8')
        .replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '')).join('\n');

// The event host checks serialization and data edits; browser checks own real layout.
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.attrs = {}; this.dataset = {}; this.events = {}; this.value = ''; }
    appendChild(child) { this.children.push(child); child.parent = this; }
    replaceChildren() { this.children = []; }
    setAttribute(key, value) { this.attrs[key] = value; }
    addEventListener(name, fn) { (this.events[name] ||= []).push(fn); }
    fire(name, event = {}) { for (const fn of this.events[name] || []) fn(event); }
    click() { this.fire('click'); }
    focus() {}
    showModal() {}
    close() { this.fire('close'); }
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
}
function environment() {
    const document = {head: new Element('head'), body: new Element('body'),
        createElement: tag => new Element(tag), getElementById: () => null};
    const ctx = vm.createContext({document, app: {registerExtension() {}}, setTimeout() {},
        ResizeObserver: class {observe() {} disconnect() {}}});
    vm.runInContext(source, ctx);
    return ctx;
}
function node(values) {
    return {properties: {}, inputs: [], size: [440, 300],
        widgets: Object.entries(values).map(([name, value]) => ({name, value, type: 'control', options: {}})),
        addWidget(type, name, value, callback, options) { const w = {type, name, value, callback, options}; this.widgets.push(w); return w; },
        addDOMWidget(name, type, element, options) { const w = {name, type, element, options}; this.widgets.push(w); return w; },
        setSize() {}, computeSize() { return [440, 300]; }};
}
const widget = (node, name) => node.widgets.find(w => w.name === name);
function descendants(element) { return [element, ...element.children.flatMap(descendants)]; }
function field(ctx, label) { return descendants(ctx.document.body).find(el => el.attrs['aria-label'] === label); }
function button(ctx, text) { return descendants(ctx.document.body).find(el => el.tag === 'button' && el.textContent === text); }
function open(ctx, target) { ctx.target = target; vm.runInContext('setupSegmentPromptEditor(target); openPromptEditor(target)', ctx); }

test('one paste round-trips sparse segments, metadata and text without rewriting on view switches', () => {
    const ctx = environment();
    const raw = '{ "2": "鞋底", "08": {"prompt":"转身", "transition":"cut", "dialogue":[]} }';
    const target = node({prompt: 'global unchanged', segment_prompts: raw});
    open(ctx, target);
    button(ctx, '分段编辑').click();
    assert.equal(widget(target, 'segment_prompts').value, raw);
    const select = field(ctx, '当前分段'); select.value = '08'; select.fire('change');
    assert.equal(field(ctx, '镜头衔接').value, 'cut');
    const text = field(ctx, '本段补充提示词'); text.value = '改变动作'; text.fire('input');
    button(ctx, '完整 JSON').click();
    const saved = JSON.parse(field(ctx, '各段补充提示词 JSON').value);
    assert.deepEqual(saved, {'2': '鞋底', '08': {prompt: '改变动作', transition: 'cut', dialogue: []}});
    assert.equal(widget(target, 'prompt').value, 'global unchanged');
    button(ctx, '关闭').click(); open(ctx, target);
    assert.equal(field(ctx, '各段补充提示词 JSON').value, widget(target, 'segment_prompts').value);
    assert.equal(target.widgets.filter(w => w.name === 'pj_prompt_editor').length, 1);
    assert.equal(widget(target, 'pj_prompt_editor').serialize, false);
});

test('invalid JSON stays intact and cannot be silently replaced by segmented view', () => {
    const ctx = environment(); const target = node({segment_prompts: '{"2":'});
    open(ctx, target); button(ctx, '分段编辑').click();
    assert.equal(widget(target, 'segment_prompts').value, '{"2":');
    assert.equal(field(ctx, '各段补充提示词 JSON').parent.hidden, false);
    assert.match(descendants(ctx.document.body).find(el => el.attrs.role === 'alert').textContent, /JSON/);
});

test('external prompt connection prevents stale local edits', () => {
    const ctx = environment(); const target = node({segment_prompts: '{"1":"local"}'});
    open(ctx, target);
    target.inputs = [{name: 'segment_prompts', link: 17}]; target.onConnectionsChange();
    const before = widget(target, 'segment_prompts').value;
    const raw = field(ctx, '各段补充提示词 JSON'); raw.value = '{}'; raw.fire('input');
    assert.equal(widget(target, 'segment_prompts').value, before);
    assert.equal(raw.parent.parent.hidden, true);
    target.inputs[0].link = null; target.onConnectionsChange();
    assert.equal(raw.value, before);
});

test('empty script can add a segment and structured dialogue without altering unrelated entries', () => {
    const ctx = environment(); const target = node({segment_prompts: '{}'}); open(ctx, target);
    button(ctx, '分段编辑').click(); button(ctx, '添加分段').click();
    const scene = field(ctx, '镜头衔接'); scene.value = 'cut'; scene.fire('change');
    const speech = field(ctx, '本段对白'); speech.value = 'lines'; speech.fire('change');
    const line = field(ctx, '第 1 句台词'); line.value = '[Chinese] 你好'; line.fire('input');
    const parsed = JSON.parse(widget(target, 'segment_prompts').value);
    assert.equal(parsed['1'].transition, 'cut');
    assert.equal(parsed['1'].dialogue[0].text, '[Chinese] 你好');
    assert.equal(parsed['1'].dialogue[0].start, 0);
});

test('backend-invalid structured fields and ambiguous segment numbers are rejected in the editor', () => {
    const ctx = environment();
    for (const value of ['[]', '{"1":3}', '{"1":"x","01":"y"}', '{"1":{"typo":true}}', '{"1":{"dialogue":[{}]}}']) {
        ctx.raw = value;
        assert.throws(() => vm.runInContext('parseSegmentPrompts(raw)', ctx));
    }
});

test('deleting a segment keeps sparse ids, other settings and global prompt through reopening', () => {
    const ctx = environment();
    const retained = {prompt: '转身', transition: 'cut', audio_transition: 'reset', dialogue: []};
    const target = node({prompt: 'global unchanged', segment_prompts: JSON.stringify({'2': '鞋底', '4': '', '08': retained})});
    open(ctx, target); button(ctx, '分段编辑').click();
    const select = field(ctx, '当前分段'); select.value = '4'; select.fire('change');
    button(ctx, '删除本段配置').click();
    assert.deepEqual(JSON.parse(widget(target, 'segment_prompts').value), {'2': '鞋底', '08': retained});
    assert.equal(select.value, '08');
    assert.equal(widget(target, 'prompt').value, 'global unchanged');
    button(ctx, '完整 JSON').click();
    assert.deepEqual(JSON.parse(field(ctx, '各段补充提示词 JSON').value), {'2': '鞋底', '08': retained});
    button(ctx, '关闭').click(); open(ctx, target); button(ctx, '分段编辑').click();
    assert.deepEqual(field(ctx, '当前分段').children.map(option => option.value), ['2', '08']);
});

test('deleting all segment settings leaves an empty script that can add again and respects external input', () => {
    const ctx = environment(); const target = node({segment_prompts: '{"3":{"prompt":"x","dialogue":[]}}'});
    open(ctx, target); button(ctx, '分段编辑').click();
    target.inputs = [{name: 'segment_prompts', link: 17}]; target.onConnectionsChange();
    const before = widget(target, 'segment_prompts').value;
    button(ctx, '删除本段配置').click();
    assert.equal(widget(target, 'segment_prompts').value, before);
    target.inputs[0].link = null; target.onConnectionsChange();
    button(ctx, '删除本段配置').click();
    assert.equal(widget(target, 'segment_prompts').value, '{}');
    assert.equal(button(ctx, '删除本段配置').disabled, true);
    assert.equal(field(ctx, '本段补充提示词').disabled, true);
    button(ctx, '添加分段').click();
    assert.deepEqual(JSON.parse(widget(target, 'segment_prompts').value), {'1': ''});
    assert.equal(button(ctx, '删除本段配置').disabled, false);
});

test('purpose tabs preserve native widget order, wire ids and hidden parameter values', () => {
    const ctx = environment();
    const target = node({generation_mode: '整段采样', repair_action: '原采样方式', run_mode: '整段全量采样(一采不分段)',
        base_seed: 123, repair_denoise: 0.42, repair_boundary_tokens: 7, relay_action: '全部生成', relay_start_segment: 2,
        blend_mode: '硬切', sync_ref_video: false, seg1_seed_offset: 99});
    target.inputs = [{name: 'segment_plan', link: 123}];
    const order = target.widgets.map(w => w.name); ctx.target = target;
    vm.runInContext('syncSamplerSegmentCount(target, true)', ctx);
    const tabs = widget(target, 'pj_purpose_tabs');
    tabs._pjRoot.children.find(button => button.dataset.value === '22帧接力').click();
    assert.equal(widget(target, 'generation_mode').value, '22帧接力');
    assert.equal(widget(target, 'repair_denoise').hidden, true);
    tabs._pjRoot.children.find(button => button.dataset.value === '仅重绘目标段').click();
    assert.equal(widget(target, 'repair_denoise').hidden, false);
    assert.equal(widget(target, 'relay_action').hidden, true);
    assert.equal(widget(target, 'repair_denoise').value, 0.42);
    assert.equal(widget(target, 'seg1_seed_offset').value, 99);
    assert.deepEqual(target.inputs, [{name: 'segment_plan', link: 123}]);
    assert.deepEqual(target.widgets.filter(w => w.serialize !== false).map(w => w.name), order);
});

test('context tabs retain audio and reference settings independently', () => {
    const ctx = environment();
    const target = node({prompt: 'global', segment_prompts: '{}', reference_mode: '原视频编辑（按时间裁剪）', audio_mode: '参考声音生成'});
    ctx.target = target; vm.runInContext('setupContextModes(target)', ctx);
    widget(target, 'pj_audio_mode_tabs')._pjRoot.children.find(b => b.dataset.value === '保留指定音轨').click();
    assert.equal(widget(target, 'audio_mode').value, '保留指定音轨');
    assert.equal(widget(target, 'reference_mode').value, '原视频编辑（按时间裁剪）');
    assert.equal(widget(target, 'segment_prompts').value, '{}');
    vm.runInContext('setupContextModes(target)', ctx);
    assert.equal(target.widgets.filter(w => w.type === 'pj_tabs').length, 2);
});
