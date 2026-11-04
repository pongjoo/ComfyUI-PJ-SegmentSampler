import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/pj_segment_sampler.js', import.meta.url), 'utf8')
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

const uiContext = vm.createContext({ app: { registerExtension() {} }, setTimeout() {} });
vm.runInContext(source, uiContext);
vm.runInContext('updateSamplerControls = () => {};', uiContext);
function sampler(mode, first = false) {
    const values = {run_mode: mode, target_time_second: 5, custom_rerun_segments: '2',
        sync_ref_video: false, boundary_constraints: false, blend_mode: '硬切', unload_model_after_sampling: false};
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
    const ctx = vm.createContext({app:{registerExtension(value) {extension=value;}}});
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
