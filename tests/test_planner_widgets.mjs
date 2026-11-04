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
