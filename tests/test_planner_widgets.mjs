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
vm.runInContext('updateSamplerGuide = () => {};', uiContext);
function sampler(mode, first = false) {
    const values = {run_mode: mode, target_time_second: 5, custom_rerun_segments: '2',
        sync_ref_video: false, blend_mode: '硬切', unload_model_after_sampling: false};
    for (let i = 1; i <= 20; i++) {
        values[`seg${i}_lock`] = i === 8;
        if (first) values[`seg${i}_seed_offset`] = i - 1;
    }
    return {properties: {}, widgets: Object.entries(values).map(([name, value]) =>
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
    assert.ok(widget(node, 'run_mode').options.values.includes('仅精修第20段'));
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

test('target fields follow the selected operation and guides do not promise cache hits', () => {
    const node = sampler('按缺陷秒数精修 (下方输入秒数)');
    refresh(node);
    assert.equal(visible(node, 'target_time_second'), true);
    assert.equal(visible(node, 'custom_rerun_segments'), false);
    const guide = vm.runInContext('samplerGuideText(node, false)', uiContext);
    assert.match(guide.action, /无缓存或缓存失效时仍需计算/);
    assert.match(guide.planning, /未连接/);
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

test('second pass tasks map saved backend modes without changing workflow values', () => {
    const cases = [['全部精修(刷新全部缓存)', '常规分段采样'],
        ['智能缓存(仅重跑未锁定分段)', '按锁定状态采样'],
        ['仅精修第18段', '局部重做'], ['自定义指定分段(下方输入)', '局部重做']];
    for (const [mode, task] of cases) {
        const node = sampler(mode);
        refresh(node);
        assert.equal(vm.runInContext('secondPassOperation(node.widgets[0].value)', uiContext), task);
        assert.equal(widget(node, 'run_mode').value, mode);
        assert.equal(visible(node, 'run_mode'), false);
        assert.equal(visible(node, 'seg1_lock'), task === '按锁定状态采样');
    }
    const node = sampler('全部精修(刷新全部缓存)');
    refresh(node);
    assert.match(vm.runInContext('samplerGuideText(node, false).action', uiContext), /逐段采样一次/);
});

test('new second pass nodes default to normal sampling; configure restores saved modes', async () => {
    let extension;
    const creationContext = vm.createContext({app: {registerExtension(value) {extension=value;}}, setTimeout() {}});
    vm.runInContext(source, creationContext);
    vm.runInContext('updateSamplerGuide = () => {};', creationContext);
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
