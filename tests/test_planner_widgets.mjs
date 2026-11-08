import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const editorSource = readFileSync(new URL('../js/pj_segment_prompt_editor.js', import.meta.url), 'utf8')
    .replace(/^export /gm, '');
const tabsSource = readFileSync(new URL('../js/pj_mode_tabs.js', import.meta.url), 'utf8').replace(/^export /gm, '');
const source = editorSource + '\n' + tabsSource + '\n' + readFileSync(new URL('../js/pj_segment_sampler.js', import.meta.url), 'utf8')
    .replace(/^import .*;\r?\n/gm, '');
const context = vm.createContext({ app: { registerExtension() {} } });
vm.runInContext(source, context);

function migrate(info) {
    context.info = info;
    vm.runInContext('sanitizePlannerWidgets(info)', context);
    return JSON.parse(JSON.stringify(info));
}

test('current six controls and old seven controls keep the same reference values', () => {
    const values = [24, '按秒自动切分', 6, 0.5, 3, '0-6, 5.5-12'];
    for (const grid of [[], [false], [true]]) {
        const result = migrate({ widgets_values: [...values, ...grid] });
        assert.deepEqual(result.widgets_values, values);
        assert.deepEqual(migrate(result), result);
    }
});

test('old mode-first and duration controls migrate without shifting values', () => {
    for (const mode of ['按秒自动切分', 'auto_by_seconds']) {
        const expected = [30, mode, 6.5, 0.5, 4, '0-6'];
        assert.deepEqual(migrate({ widgets_values: [mode, 6.5, 0.5, 4, '0-6', true, 30] }).widgets_values, expected);
        assert.deepEqual(migrate({ widgets_values: [30, 15, mode, 6.5, 0.5, 4, '0-6', true] }).widgets_values, expected);
    }
});

test('old preview is discarded, current backend preview survives', () => {
    const old = migrate({ properties: { pj_preview_cache: { n_chunks: 4 }, unrelated: true } });
    assert.deepEqual(old.properties, { unrelated: true });
    const current = { properties: { pj_preview_cache: { timing_policy: 'nearest_h3', n_chunks: 3 } } };
    assert.deepEqual(migrate(current), current);
});

let uiExtension;
const uiContext = vm.createContext({ app: { registerExtension(value) { uiExtension = value; } }, setTimeout() {} });
vm.runInContext(source, uiContext);
vm.runInContext('updateSamplerControls = () => {};', uiContext);
function sampler(mode, first = false) {
    const values = {run_mode: mode, target_time_second: 5, custom_rerun_segments: '2',
        sync_ref_video: false, boundary_constraints: false, blend_mode: '硬切', keep_original_audio: false, unload_model_after_sampling: false};
    for (let i = 1; i <= 20; i++) {
        values[`seg${i}_lock`] = i === 8;
        if (first) values[`seg${i}_seed_offset`] = i - 1;
    }
    return {properties: {}, inputs: [{name: "segment_plan", link: 1}], widgets: Object.entries(values).map(([name, value]) =>
        ({name, value, type: 'number', options: {}}))};
}
function refresh(node, first = false) {
    uiContext.node = node;
    uiContext.first = first;
    vm.runInContext('syncSamplerSegmentCount(node, first)', uiContext);
}
const widget = (node, name) => node.widgets.find(w => w.name === name);
const visible = (node, name) => widget(node, name).type !== 'pj_hidden';

function overlapPlanner(seconds = 0.5) {
    const values = {fps: 30, split_mode: '按秒自动切分', chunk_seconds: 6, overlap_seconds: seconds,
        num_segments: 3, custom_cuts: '0-6, 5.5-12'};
    return {properties: {}, inputs: [], outputs: [], widgets: Object.entries(values).map(([name, value]) =>
        ({name, value, type: 'number', options: name === 'overlap_seconds' ? {max: 10} : {}})),
        addWidget(type, name, value, callback, options) {
            if (type === 'combo') assert.ok(Array.isArray(options?.values), 'native combo requires values at creation');
            const w = {type, name, value, callback, options}; this.widgets.push(w); return w;
        }};
}

function refreshPlanner(node) {
    uiContext.node = node;
    vm.runInContext('setupPlannerNode(node)', uiContext);
}

test('overlap frame menu reads legacy seconds without changing saved values or their order', () => {
    for (const [seconds, label] of [[0, '0 帧（不重叠）'], [0.5, '5 帧'], [0.92, '22 帧'], [1.5, '39 帧'], [10, '243 帧']]) {
        const node = overlapPlanner(seconds);
        const original = node.widgets.map(w => [w.name, w.value]);
        refreshPlanner(node);
        assert.equal(widget(node, 'pj_overlap_frames').value, label);
        assert.equal(visible(node, 'overlap_seconds'), false);
        assert.equal(visible(node, 'pj_overlap_frames'), true);
        assert.deepEqual(node.widgets.filter(w => w.serialize !== false).map(w => [w.name, w.value]), original);
        refreshPlanner(node);
        assert.equal(node.widgets.filter(w => w.name === 'pj_overlap_frames').length, 1);
    }
});

test('every overlap menu choice round-trips through the backend seconds field on the H3 clock', () => {
    const node = overlapPlanner();
    widget(node, 'overlap_seconds').callback = function () { this.value = Math.round(this.value * 10) / 10; };
    refreshPlanner(node);
    const frames = widget(node, 'pj_overlap_frames');
    assert.deepEqual(Array.from(frames.options.values).slice(0, 5), ['0 帧（不重叠）', '5 帧', '22 帧', '39 帧', '56 帧']);
    for (const choice of frames.options.values) {
        node.properties.pj_preview_cache = {n_chunks: 9};
        frames.callback(choice);
        assert.equal(frames.value, choice);
        assert.equal(widget(node, 'fps').value, 30);
        assert.ok(widget(node, 'overlap_seconds').value <= 10);
        assert.equal(node.properties.pj_preview_cache, undefined);
        const saved = node.widgets.filter(w => w.serialize !== false).map(w => w.value);
        const restored = overlapPlanner(saved[3]); refreshPlanner(restored);
        assert.equal(widget(restored, 'pj_overlap_frames').value, choice);
        assert.equal(saved.length, 6);
    }
});

test('custom cuts hide overlap and switching planner tabs retains the frame selection', () => {
    const node = overlapPlanner(); refreshPlanner(node);
    widget(node, 'pj_overlap_frames').callback('22 帧');
    const mode = widget(node, 'split_mode');
    mode.value = '自定义切点'; mode.callback();
    assert.equal(visible(node, 'pj_overlap_frames'), false);
    assert.equal(visible(node, 'overlap_seconds'), false);
    mode.value = '固定分段数'; mode.callback();
    assert.equal(visible(node, 'pj_overlap_frames'), true);
    assert.equal(widget(node, 'pj_overlap_frames').value, '22 帧');
});

test('legacy wired overlap keeps its seconds contract and hides the local frame selector', () => {
    const node = overlapPlanner();
    const input = {name: 'overlap_seconds', widget: {name: 'overlap_seconds'}, link: 7};
    node.inputs.push(input); refreshPlanner(node);
    assert.equal(input.link, 7);
    assert.match(input.label, /秒数/);
    assert.equal(visible(node, 'overlap_seconds'), true);
    assert.equal(visible(node, 'pj_overlap_frames'), false);
    assert.equal(widget(node, 'overlap_seconds').value, 0.5);
    input.link = null; refreshPlanner(node);
    assert.equal(visible(node, 'overlap_seconds'), false);
    assert.equal(visible(node, 'pj_overlap_frames'), true);
});

test('second pass exposes blending only for regenerated audio', () => {
    const node = sampler('全部精修(刷新全部缓存)');
    node.properties.pj_advanced_open = true;
    widget(node, 'keep_original_audio').value = true;
    refresh(node);
    assert.equal(visible(node, 'blend_mode'), false);
    assert.equal(widget(node, 'run_mode').label, '精修范围（含后续段）');
    widget(node, 'keep_original_audio').value = false;
    refresh(node);
    assert.equal(visible(node, 'blend_mode'), true);
    assert.equal(widget(node, 'blend_mode').label, '音频重叠过渡');
    assert.equal(widget(node, 'blend_mode').value, '硬切');
    uiContext.node = node;
    vm.runInContext('attachSamplerHandlers(node, false)', uiContext);
    widget(node, 'keep_original_audio').value = true;
    widget(node, 'keep_original_audio').callback();
    assert.equal(visible(node, 'blend_mode'), false);
});

test('hidden widget sockets do not draw and expanding restores the original renderer', () => {
    const node = sampler('全部精修(刷新全部缓存)');
    let draws = 0;
    const input = {name: 'blend_mode', widget: {name: 'blend_mode'}, link: null, alwaysVisible: true,
        draw(context, options) {assert.equal(this, input); assert.equal(context, 'canvas'); draws++; return options;}};
    node.inputs.push(input);
    const inputs = [...node.inputs];
    const values = node.widgets.map(w => w.value);
    refresh(node);
    input.draw('canvas', 'hidden');
    assert.equal(draws, 0);
    assert.equal(widget(node, 'blend_mode').hidden, true);
    const wrapped = input.draw;
    refresh(node);
    assert.equal(input.draw, wrapped);
    node.properties.pj_advanced_open = true;
    refresh(node);
    assert.equal(input.draw('canvas', 'visible'), 'visible');
    assert.equal(draws, 1);
    assert.equal(widget(node, 'blend_mode').hidden, false);
    assert.deepEqual(node.inputs, inputs);
    assert.deepEqual(node.widgets.map(w => w.value), values);
});

test('connected widget inputs keep their rows when folding or disconnecting the planner', () => {
    const node = sampler('智能缓存(仅重跑未锁定分段)', true);
    node.inputs[0].link = null;
    const input = {name: 'seg8_seed_offset', widget: {name: 'seg8_seed_offset'}, link: 0, draw() {return 'drawn';}};
    node.inputs.push(input);
    refresh(node, true);
    assert.equal(visible(node, 'seg8_seed_offset'), true);
    assert.equal(widget(node, 'seg8_seed_offset').hidden, false);
    assert.equal(input.draw(), 'drawn');
    assert.equal(input.link, 0);
    input.link = null;
    refresh(node, true);
    assert.equal(visible(node, 'seg8_seed_offset'), false);
    assert.equal(input.draw(), undefined);
    node.isInputConnected = index => index === 1;
    refresh(node, true);
    assert.equal(visible(node, 'seg8_seed_offset'), true);
    assert.equal(input.draw(), 'drawn');
});

test('socket renderer follows replaced widgets after workflow configuration', () => {
    const node = sampler('全部精修(刷新全部缓存)');
    const input = {name: 'blend_mode', widget: {name: 'blend_mode'}, link: null, draw() {return 'drawn';}};
    node.inputs.push(input);
    refresh(node);
    const index = node.widgets.findIndex(w => w.name === 'blend_mode');
    node.widgets[index] = {name: 'blend_mode', type: 'combo', value: '硬切'};
    node.properties.pj_advanced_open = true;
    refresh(node);
    assert.equal(input.draw(), 'drawn');
    node.properties.pj_advanced_open = false;
    refresh(node);
    assert.equal(input.draw(), undefined);
    assert.equal(node.inputs[1], input);
});

test('preview count cannot reset a saved target beyond the preview', () => {
    const node = sampler('仅精修第18段');
    refresh(node);
    assert.equal(widget(node, 'run_mode').value, '仅精修第18段');
    assert.ok(widget(node, 'run_mode').options.values.includes('仅精修第18段'));
    assert.ok(!widget(node, 'run_mode').options.values.includes('仅精修第20段'));
    assert.equal(visible(node, 'seg1_lock'), false);
    assert.equal(widget(node, 'seg8_lock').value, true);
});

test('smart locks and advanced settings reflect applicable mode without changing values', () => {
    const node = sampler('智能缓存(仅重跑未锁定分段)', true);
    refresh(node, true);
    assert.equal(visible(node, 'seg1_lock'), true);
    assert.equal(visible(node, 'seg8_lock'), true);
    assert.equal(visible(node, 'sync_ref_video'), false);
    assert.equal(visible(node, 'seg1_seed_offset'), false);
    node.properties.pj_advanced_open = true;
    refresh(node, true);
    assert.equal(visible(node, 'sync_ref_video'), true);
    assert.equal(visible(node, 'seg1_seed_offset'), true);
    assert.equal(widget(node, 'sync_ref_video').value, false);
    widget(node, 'run_mode').value = '整段全量采样(一采不分段)';
    refresh(node, true);
    for (const name of ['blend_mode', 'seg1_lock', 'seg2_seed_offset'])
        assert.equal(visible(node, name), false);
    assert.equal(visible(node, 'seg1_seed_offset'), true);
    assert.equal(visible(node, 'sync_ref_video'), true);
    assert.equal(visible(node, 'unload_model_after_sampling'), true);
});

test('target fields follow the selected operation', () => {
    const node = sampler('按缺陷秒数精修 (下方输入秒数)');
    refresh(node);
    assert.equal(visible(node, 'target_time_second'), true);
    assert.equal(visible(node, 'custom_rerun_segments'), false);
    widget(node, 'run_mode').value = '自定义指定分段(下方输入)';
    refresh(node);
    assert.equal(visible(node, 'target_time_second'), false);
    assert.equal(visible(node, 'custom_rerun_segments'), true);
});

test('selecting all refinement clears locks while preserving advanced values', () => {
    const node = sampler('智能缓存(仅重跑未锁定分段)');
    uiContext.node = node;
    vm.runInContext('attachSamplerHandlers(node, false)', uiContext);
    const mode = widget(node, 'run_mode');
    mode.value = '全部精修(刷新全部缓存)';
    mode.callback();
    assert.equal(widget(node, 'seg8_lock').value, false);
    assert.equal(visible(node, 'seg1_lock'), false);
    assert.equal(widget(node, 'blend_mode').value, '硬切');
    assert.equal(widget(node, 'unload_model_after_sampling').value, false);
});

test('native second pass selector preserves saved modes and lock visibility', () => {
    const cases = [['全部精修(刷新全部缓存)', '常规分段采样'],
        ['智能缓存(仅重跑未锁定分段)', '按锁定状态采样'],
        ['仅精修第18段', '局部重做'], ['自定义指定分段(下方输入)', '局部重做']];
    for (const [mode, task] of cases) {
        const node = sampler(mode);
        refresh(node);
        assert.equal(widget(node, 'run_mode').value, mode);
        assert.equal(visible(node, 'run_mode'), true);
        assert.equal(visible(node, 'seg1_lock'), task === '按锁定状态采样');
    }
});

test('new second pass nodes default to normal sampling; configure restores saved modes', async () => {
    let extension;
    const creationContext = vm.createContext({app: {registerExtension(value) {extension=value;}}, setTimeout() {}});
    vm.runInContext(source, creationContext);
    vm.runInContext('updateSamplerControls = () => {};', creationContext);
    for (const name of ['PJ_SegmentSampler', 'PJSegmentSampler']) {
        class Node {
            constructor() { Object.assign(this, sampler('智能缓存(仅重跑未锁定分段)')); }
            configure(info) { this.widgets[0].value = info.widgets_values[0]; }
        }
        await extension.beforeRegisterNodeDef(Node, {name});
        const node = new Node();
        node.onNodeCreated();
        assert.equal(widget(node, 'run_mode').value, '全部精修(刷新全部缓存)');
        node.configure({widgets_values: ['仅精修第18段']});
        assert.equal(widget(node, 'run_mode').value, '仅精修第18段');
        node.configure({widgets_values: ['智能缓存(仅重跑未锁定分段)']});
        assert.equal(widget(node, 'seg8_lock').value, true);
    }
});


test('boundary constraints stay visible, replace blending, and do not reset saved controls', () => {
    const node = sampler('仅重抽第2段', true);
    node.properties.pj_advanced_open = true;
    refresh(node, true);
    assert.equal(visible(node, 'boundary_constraints'), true);
    assert.equal(visible(node, 'blend_mode'), true);
    widget(node, 'boundary_constraints').value = true;
    refresh(node, true);
    assert.equal(visible(node, 'blend_mode'), false);
    assert.equal(widget(node, 'blend_mode').value, '硬切');
    widget(node, 'run_mode').value = '整段全量采样(一采不分段)';
    refresh(node, true);
    assert.equal(visible(node, 'boundary_constraints'), false);
    assert.equal(widget(node, 'boundary_constraints').value, true);
});


test('samplers use one native advanced button and no explanation panel', () => {
    for (const first of [true, false]) {
        const node = sampler('全部精修(刷新全部缓存)', first);
        node.addWidget = (type, name, value, callback, options) => {
            const control = {type, name, value, callback, options};
            node.widgets.push(control);
            return control;
        };
        context.node = node;
        context.first = first;
        vm.runInContext('updateSamplerControls(node, first); updateSamplerControls(node, first);', context);
        const buttons = node.widgets.filter(w => w.type === 'button');
        assert.equal(buttons.length, 1);
        assert.equal(buttons[0].label, '展开高级设置');
        assert.equal(buttons[0].options.serialize, false);
        // Run the real button callback without browser timers.
        context.setTimeout = () => {};
        buttons[0].callback();
        assert.equal(node.properties.pj_advanced_open, true);
        assert.equal(buttons[0].label, '收起高级设置');
        assert.equal(node.widgets.some(w => w.name === 'pj_sampler_guide'), false);
    }
});

test('edit preparer exposes Chinese input labels without extra sampling widgets', async () => {
    let extension;
    const ctx = vm.createContext({app:{registerExtension(value) {extension=value;}}, setTimeout() {}});
    vm.runInContext(source, ctx);
    class Node {
        constructor() {this.inputs = ['av_latent','source_latent','edit_mask'].map(name=>({name}));}
    }
    await extension.beforeRegisterNodeDef(Node, {name:'PJ_H3EditPrepare'});
    const node = new Node();
    node.onNodeCreated();
    assert.deepEqual(node.inputs.map(i=>i.label), ['H3目标音视频潜空间','原视频编码潜空间','空间编辑遮罩(白改黑留)']);
    node.onConfigure();
    assert.equal(node.inputs.length,3);
});


test('disconnected planners hide all segment controls and preserve saved values', () => {
    for (const first of [true, false]) {
        const node = sampler(first ? '仅重抽第8段' : '按缺陷秒数精修 (下方输入秒数)', first);
        node.properties.pj_advanced_open = true;
        node.inputs[0].link = null;
        refresh(node, first);
        for (const name of ['run_mode', 'target_time_second', 'custom_rerun_segments',
                            'seg1_lock', 'seg8_lock', 'boundary_constraints', 'blend_mode'])
            assert.equal(visible(node, name), false, name);
        assert.equal(widget(node, 'seg8_lock').value, true);
        if (first) {
            assert.equal(visible(node, 'seg1_seed_offset'), true);
            assert.equal(visible(node, 'seg2_seed_offset'), false);
        }
        node.inputs[0].link = 1;
        refresh(node, first);
        assert.equal(visible(node, 'run_mode'), true);
        assert.equal(widget(node, 'run_mode').value, first ? '仅重抽第8段' : '按缺陷秒数精修 (下方输入秒数)');
    }
});

test('both selectors follow upstream planner counts through reroutes', () => {
    const planner = {type: 'PJ_SegmentPlanner', properties: {pj_preview_cache: {n_chunks: 3}}};
    uiContext.app.graph = {links: {1: {origin_id: 10}, 2: {origin_id: 20}},
        getNodeById(id) {return id === 10 ? {type: 'Reroute', inputs: [{link: 2}]} : planner;}};
    try {
        for (const first of [true, false]) {
            const prefix = first ? '仅重抽第' : '仅精修第';
            const node = sampler(first ? '全部重新采样(刷新全部缓存)' : '全部精修(刷新全部缓存)', first);
            for (const count of [3, 6, 2]) {
                planner.properties.pj_preview_cache.n_chunks = count;
                refresh(node, first);
                const choices = widget(node, 'run_mode').options.values.filter(value => value.startsWith(prefix));
                assert.deepEqual(Array.from(choices), Array.from({length: count}, (_, i) => `${prefix}${i + 1}段`));
            }
        }
    } finally {delete uiContext.app.graph;}
});


function relayWidgets(node, generation = '22帧接力') {
    for (const [name, value] of Object.entries({generation_mode: generation,
            relay_action: '全部生成', relay_start_segment: 1}))
        node.widgets.push({name, value, type: 'number', options: {}});
    return node;
}

test('relay exposes generation and restart controls without locks or blending', () => {
    const node = relayWidgets(sampler('智能缓存(仅重跑未锁定分段)', true));
    node.properties.pj_advanced_open = true;
    refresh(node, true);
    for (const name of ['generation_mode', 'relay_action', 'seg1_seed_offset'])
        assert.equal(visible(node, name), true);
    for (const name of ['run_mode', 'relay_start_segment', 'seg1_lock', 'seg8_lock', 'blend_mode', 'boundary_constraints'])
        assert.equal(visible(node, name), false, name);
    widget(node, 'relay_action').value = '从指定段重做';
    refresh(node, true);
    assert.equal(visible(node, 'relay_start_segment'), true);
    node.inputs[0].link = null;
    refresh(node, true);
    for (const name of ['generation_mode', 'relay_action', 'relay_start_segment', 'run_mode'])
        assert.equal(visible(node, name), false);
    node.inputs[0].link = 1;
    widget(node, 'generation_mode').value = '分段编辑';
    refresh(node, true);
    assert.equal(visible(node, 'run_mode'), true);
    assert.equal(visible(node, 'seg8_lock'), true);
    assert.equal(widget(node, 'seg8_lock').value, true);
});

test('legacy full-length selection migrates and can switch to relay or editing', () => {
    const node = relayWidgets(sampler('整段全量采样(一采不分段)', true), '分段编辑');
    refresh(node, true);
    assert.equal(widget(node, 'generation_mode').value, '整段采样');
    assert.equal(visible(node, 'run_mode'), false);
    uiContext.node = node;
    vm.runInContext('attachSamplerHandlers(node, true)', uiContext);
    for (const value of ['22帧接力', '分段编辑']) {
        const generation = widget(node, 'generation_mode');
        generation.value = value;
        generation.callback();
        assert.equal(generation.value, value);
        assert.equal(visible(node, 'run_mode'), value === '分段编辑');
    }
});

test('relay count uses relay preview, not normal planning count', () => {
    const planner = {type: 'PJ_SegmentPlanner', properties: {pj_preview_cache: {n_chunks: 4, relay_n_chunks: 3, total_frames: 362}},
        widgets: [{name: 'chunk_seconds', value: 6}]};
    uiContext.app.graph = {links: {1: {origin_id: 10}}, getNodeById() {return planner;}};
    try {
        const node = relayWidgets(sampler('智能缓存(仅重跑未锁定分段)', true));
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 3);
        delete planner.properties.pj_preview_cache.relay_n_chunks;
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 3);
        planner.properties.pj_preview_cache = {n_chunks: 7, relay_n_chunks: 6, total_frames: 719};
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 6);
        widget(node, 'relay_start_segment').value = 9;
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').value, 9);
        assert.equal(widget(node, 'relay_start_segment').options.max, 9);
    } finally {delete uiContext.app.graph;}
});

async function executedSampler(first = true, generation = '22帧接力') {
    const rules = {split_mode: '按秒自动切分', chunk_seconds: 6, overlap_seconds: 0.5,
        num_segments: 3, custom_cuts: '0-6, 4.5-10.5, 9-15'};
    const planner = {id: 10, type: 'PJ_SegmentPlanner',
        widgets: Object.entries(rules).map(([name, value]) => ({name, value})),
        properties: {pj_preview_cache: {n_chunks: 3, relay_n_chunks: 3, total_frames: 345}}};
    const inputs = {20: {id: 20, type: 'EmptyLatent'}, 30: {id: 30, type: 'OfficialBaseline'}};
    uiContext.app.graph = {links: {1: {origin_id: 10}, 2: {origin_id: 20}, 3: {origin_id: 30}},
        getNodeById(id) { return id === 10 ? planner : inputs[id]; }};
    class Node {
        constructor() {
            Object.assign(this, sampler(first ? '仅重抽第2段' : '仅精修第2段', first));
            this.inputs.push({name: 'latent_image', link: 2}, {name: first ? 'repair_baseline' : 'source_result', link: 3});
            if (first) relayWidgets(this, generation);
            this.widgets.push({name: first ? 'repair_action' : 'refine_action',
                value: first ? '原采样方式' : '常规精修', type: 'combo', options: {}});
        }
        onExecuted() { this.originalExecuted = true; return 'original'; }
    }
    await uiExtension.beforeRegisterNodeDef(Node, {name: first ? 'PJ_FirstPass_SegmentSampler' : 'PJ_SegmentSampler'});
    const node = new Node();
    node.onNodeCreated();
    refresh(node, first);
    const send = (changes = {}) => node.onExecuted({pj_sampling_summary: [{n_chunks: 7,
        mode: first && generation === '22帧接力' ? 'relay' : 'segments', plan: {...rules},
        total_video_tokens: 252, ...changes}]});
    return {node, planner, rules, send};
}

test('executed relay uses seven real windows instead of the rule-only preview', async () => {
    const {node, planner, send} = await executedSampler();
    try {
        assert.equal(widget(node, 'relay_start_segment').options.max, 3);
        assert.equal(send(), 'original');
        assert.equal(node.originalExecuted, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 7);
        assert.equal(planner.properties.pj_preview_cache.relay_n_chunks, 3);
        widget(node, 'repair_action').value = '仅重绘目标段';
        refresh(node, true);
        assert.equal(widget(node, 'run_mode').options.values.filter(value => value.startsWith('仅重抽第')).length, 7);
        assert.equal(node.properties.pj_sampling_summary, undefined);
    } finally { delete uiContext.app.graph; }
});

test('executed second pass owns its real count and preserves a saved target', async () => {
    const {node, planner, send} = await executedSampler(false);
    try {
        widget(node, 'run_mode').value = '仅精修第18段';
        send();
        const choices = widget(node, 'run_mode').options.values;
        assert.equal(choices.includes('仅精修第7段'), true);
        assert.equal(choices.includes('仅精修第8段'), false);
        assert.equal(widget(node, 'run_mode').value, '仅精修第18段');
        assert.equal(planner.properties.pj_preview_cache.n_chunks, 3);
    } finally { delete uiContext.app.graph; }
});

test('rule changes discard the executed count but retain the real media length', async () => {
    const {node, planner, send} = await executedSampler();
    try {
        send({n_chunks: 9});
        assert.equal(widget(node, 'relay_start_segment').options.max, 9);
        widget(planner, 'chunk_seconds').value = 8;
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 5);
        widget(planner, 'chunk_seconds').value = 6;
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 6);
        widget(node, 'relay_start_segment').value = 12;
        widget(planner, 'chunk_seconds').value = 8;
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').value, 12);
        assert.equal(widget(node, 'relay_start_segment').options.max, 12);
    } finally { delete uiContext.app.graph; }
});

test('input reconnection discards both actual count and media length', async () => {
    for (const inputName of ['segment_plan', 'latent_image', 'repair_baseline', 'source_result']) {
        const first = inputName !== 'source_result';
        const {node, send} = await executedSampler(first);
        try {
            send();
            uiContext.node = node;
            assert.equal(vm.runInContext('resolveTargetSegmentCount(node)', uiContext), 7);
            const input = node.inputs.find(value => value.name === inputName);
            const link = input.link;
            input.link = null;
            node.onConnectionsChange();
            input.link = link;
            refresh(node, first);
            assert.equal(vm.runInContext('resolveTargetSegmentCount(node)', uiContext), 3, inputName);
        } finally { delete uiContext.app.graph; }
    }
});

test('output reconnection preserves the actual count and media length', async () => {
    for (const first of [true, false]) {
        const {node, send} = await executedSampler(first);
        try {
            send();
            node.outputs = [{name: 'output', links: [4]}];
            for (const connected of [false, true]) {
                node.outputs[0].links = connected ? [5] : [];
                node.onConnectionsChange(2, 0, connected);
                refresh(node, first);
                assert.equal(vm.runInContext('resolveTargetSegmentCount(node)', uiContext), 7);
            }
        } finally { delete uiContext.app.graph; }
    }
});

test('changing generation mode keeps media length but does not revive the old count', async () => {
    const {node, send} = await executedSampler();
    try {
        send({n_chunks: 9});
        assert.equal(widget(node, 'relay_start_segment').options.max, 9);
        widget(node, 'generation_mode').value = '分段编辑';
        refresh(node, true);
        const choices = widget(node, 'run_mode').options.values;
        assert.equal(choices.includes('仅重抽第7段'), true);
        assert.equal(choices.includes('仅重抽第8段'), false);
        widget(node, 'generation_mode').value = '22帧接力';
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 6);
    } finally { delete uiContext.app.graph; }
});

test('changing a latent source behind the same link discards the actual summary', async () => {
    const {node, send} = await executedSampler();
    try {
        send();
        assert.equal(widget(node, 'relay_start_segment').options.max, 7);
        uiContext.app.graph.links[2].origin_id = 30;
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 3);
    } finally { delete uiContext.app.graph; }
});

test('imported whole results provide real length before the first local repair', async () => {
    const {node, send} = await executedSampler(true, '整段采样');
    try {
        widget(node, 'repair_action').value = '载入回修基准';
        send({n_chunks: null, mode: 'full', plan: null});
        widget(node, 'repair_action').value = '仅重绘目标段';
        refresh(node, true);
        const choices = widget(node, 'run_mode').options.values;
        assert.equal(choices.includes('仅重抽第7段'), true);
        assert.equal(choices.includes('仅重抽第8段'), false);
        widget(node, 'generation_mode').value = '22帧接力';
        widget(node, 'repair_action').value = '原采样方式';
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 6);
    } finally { delete uiContext.app.graph; }
});

test('an old mode or rules cannot replace the current planning count', async () => {
    const {node, send} = await executedSampler();
    try {
        send({mode: 'segments', n_chunks: 10});
        assert.equal(widget(node, 'relay_start_segment').options.max, 6);
        send({n_chunks: 10, plan: {chunk_seconds: 8}});
        assert.equal(widget(node, 'relay_start_segment').options.max, 6);
    } finally { delete uiContext.app.graph; }
});

test('relay generation budgets before overlaps while repair covers the delivered result', async () => {
    const {node, planner, send} = await executedSampler();
    try {
        planner.properties.pj_preview_cache = {total_frames: 1076};
        widget(planner, 'chunk_seconds').value = 15;
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 3);
        send({n_chunks: 3, plan: {chunk_seconds: 15}, total_video_tokens: 307, input_video_tokens: 317});
        widget(planner, 'chunk_seconds').value = 6;
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 8);
        widget(node, 'repair_action').value = '仅重绘目标段';
        refresh(node, true);
        const choices = widget(node, 'run_mode').options.values;
        assert.equal(choices.includes('仅重抽第9段'), true);
        assert.equal(choices.includes('仅重抽第10段'), false);
        send({mode: 'relay_repair', n_chunks: 9, total_video_tokens: 307, input_video_tokens: 317});
        widget(node, 'repair_action').value = '原采样方式';
        refresh(node, true);
        assert.equal(widget(node, 'relay_start_segment').options.max, 8);
    } finally { delete uiContext.app.graph; }
});

test('actual counts above twenty preserve custom and time selection', async () => {
    const {node, send} = await executedSampler(false);
    try {
        widget(node, 'refine_action').value = '仅重做目标段';
        widget(node, 'run_mode').value = '自定义指定分段(下方输入)';
        widget(node, 'custom_rerun_segments').value = '25';
        send({n_chunks: 30, total_video_tokens: 1202});
        const choices = widget(node, 'run_mode').options.values;
        assert.equal(choices.includes('仅精修第20段'), true);
        assert.equal(choices.includes('仅精修第21段'), false);
        assert.equal(choices.includes('按缺陷秒数精修 (下方输入秒数)'), true);
        assert.equal(visible(node, 'custom_rerun_segments'), true);
        assert.equal(widget(node, 'custom_rerun_segments').value, '25');
    } finally { delete uiContext.app.graph; }
});

test('local repair exposes its own denoise and target independently of original generation', () => {
    const node = relayWidgets(sampler('仅重抽第2段', true), '整段采样');
    for (const [name, value] of Object.entries({repair_action: '仅重绘目标段', repair_denoise: 0.4,
            repair_steps: 6, repair_scheduler: 'simple', repair_request_id: 0, repair_boundary_tokens: 2}))
        node.widgets.push({name, value, type: 'number', options: {}});
    refresh(node, true);
    assert.equal(visible(node, 'generation_mode'), false);
    assert.equal(visible(node, 'run_mode'), true);
    assert.equal(widget(node, 'run_mode').label, '局部重做范围');
    assert.equal(visible(node, 'repair_denoise'), true);
    assert.equal(visible(node, 'repair_steps'), false);
    assert.equal(visible(node, 'seg8_lock'), false);
    node.properties.pj_advanced_open = true;
    refresh(node, true);
    assert.equal(visible(node, 'repair_steps'), true);
    assert.equal(visible(node, 'repair_boundary_tokens'), true);
});

test('loading and reusing results hide sampling targets without destroying saved values', () => {
    for (const action of ['载入回修基准', '复用一采结果']) {
        const node = relayWidgets(sampler('仅重抽第2段', true));
        node.widgets.push({name: 'repair_action', value: action, type: 'combo', options: {}});
        node.properties.pj_advanced_open = true;
        refresh(node, true);
        for (const name of ['generation_mode', 'run_mode', 'seg1_seed_offset', 'seg8_lock'])
            assert.equal(visible(node, name), false, name);
        assert.equal(widget(node, 'run_mode').value, '仅重抽第2段');
    }
    for (const action of ['更新一采变更', '复用高清结果']) {
        const node = sampler('仅精修第2段');
        node.widgets.push({name: 'refine_action', value: action, type: 'combo', options: {}});
        refresh(node);
        assert.equal(visible(node, 'run_mode'), false);
        assert.equal(visible(node, 'seg8_lock'), false);
    }
});

test('local selection removes cache actions and preserves explicit targets', () => {
    for (const first of [true, false]) {
        const node = sampler('智能缓存(仅重跑未锁定分段)', first);
        node.widgets.push({name: first ? 'repair_action' : 'refine_action',
            value: first ? '仅重绘目标段' : '仅重做目标段', type: 'combo', options: {}});
        refresh(node, first);
        assert.equal(widget(node, 'run_mode').value, '全部目标段');
        assert.equal(widget(node, 'run_mode').options.values.some(value => value.includes('缓存')), false);
        widget(node, 'run_mode').value = first ? '仅重抽第2段' : '仅精修第2段';
        refresh(node, first);
        assert.match(widget(node, 'run_mode').value, /第2段$/);
    }
});

test('unexecuted planner estimates with floor-aligned duration and accepts zero overlap', () => {
    const planner = {type: 'PJ_SegmentPlanner', properties: {}, widgets: [
        {name: 'split_mode', value: '按秒自动切分'}, {name: 'chunk_seconds', value: 6},
        {name: 'overlap_seconds', value: 0}]};
    uiContext.app.graph = {links: {1: {origin_id: 10}}, getNodeById() {return planner;}};
    try {
        const node = sampler('全部精修(刷新全部缓存)');
        refresh(node);
        assert.equal(widget(node, 'run_mode').options.values.filter(v => v.startsWith('仅精修第')).length, 3);
        planner.widgets[2].value = 1.5;
        refresh(node);
        assert.equal(widget(node, 'run_mode').options.values.filter(v => v.startsWith('仅精修第')).length, 3);
    } finally {delete uiContext.app.graph;}
});

test('H3 context uses concise native Chinese controls', async () => {
    let extension;
    const ctx = vm.createContext({app:{registerExtension(value) {extension=value;}}, setTimeout() {}});
    vm.runInContext(source, ctx);
    class Node {
        constructor() {
            this.inputs = ['clip', 'vae', 'reference_video', 'audio_vae'].map(name => ({name}));
            this.widgets = ['prompt', 'segment_prompts', 'reference_mode', 'motion_context'].map(name => ({name}));
        }
        addWidget(type, name, value, callback, options) {
            const widget = {type, name, value, callback, options};
            this.widgets.push(widget);
            return widget;
        }
    }
    await extension.beforeRegisterNodeDef(Node, {name:'PJ_H3SegmentContext'});
    const node = new Node();
    node.onNodeCreated();
    assert.deepEqual(node.inputs.map(i => i.label), ['文本编码器', '视频VAE', '视频参考', '音频VAE']);
    assert.deepEqual(node.widgets.map(i => i.label), ['全局提示词', '各段补充提示词', '参考用途', '前段运动引导', '编辑分段提示词']);
    assert.equal(node.widgets[4].options.serialize, false);
    node.onConfigure();
    assert.equal(node.widgets.length, 5);
});

test('H3 context migrates all old media wires to native dynamic ports', async () => {
    let extension;
    const ctx = vm.createContext({app:{registerExtension(value) {extension=value;}}});
    vm.runInContext(source, ctx);
    class Node {
        configure(info) {this.loaded = info;}
    }
    await extension.beforeRegisterNodeDef(Node, {name:'PJ_H3SegmentContext'});
    const node = new Node();
    const info = {inputs:[{name:'clip',link:1},{name:'reference_images',type:'IMAGE',link:7},
        {name:'reference_video',link:8}, {name:'reference_video_audio',link:9},
        {name:'reference_audio',link:10}], widgets_values:['prompt','{}','原视频编辑（按时间裁剪）',true]};
    node.configure(info);
    assert.equal(node.loaded.inputs[1].name, 'ref_images.ref_image_0');
    assert.equal(node.loaded.inputs[1].link, 7);
    assert.equal(node.loaded.inputs[2].name, 'ref_videos.ref_video_0');
    assert.equal(node.loaded.inputs[3].name, 'ref_video_audios.ref_video_audio_0');
    assert.equal(node.loaded.inputs[4].name, 'ref_audios.ref_audio_0');
    assert.deepEqual(node.loaded.inputs.slice(2).map(input => input.link), [8, 9, 10]);
    assert.deepEqual(node.loaded.widgets_values, ['prompt','{}','原视频编辑（按时间裁剪）',true]);
    node.configure(info);
    assert.equal(node.loaded.inputs.length, 5);
});

test('H3 preparation nodes preserve saved custom colors on configure', async () => {
    let extension;
    const ctx = vm.createContext({app:{registerExtension(value) {extension=value;}}, setTimeout() {}});
    vm.runInContext(source, ctx);
    for (const name of ['PJ_H3SegmentContext', 'PJ_H3EditPrepare']) {
        class Node {constructor() {this.inputs = []; this.widgets = [];}}
        await extension.beforeRegisterNodeDef(Node, {name});
        const node = new Node();
        node.onNodeCreated();
        assert.equal(node.bgcolor, '#042f2e');
        node.color = '#733333';
        node.bgcolor = '#542525';
        node.onConfigure({color: node.color, bgcolor: node.bgcolor});
        assert.equal(node.color, '#733333');
        assert.equal(node.bgcolor, '#542525');
    }
});
