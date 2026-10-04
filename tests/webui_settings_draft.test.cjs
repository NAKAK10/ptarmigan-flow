// Dependency-free DOM fixture: real app.js listeners/rendering, mocked native bridge.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const { test } = require('node:test');

function fixture() {
  const nodes = new Map();
  const keys = new Map();
  const element = (id, value = '') => ({
    id, value, dataset: {}, listeners: {}, textContent: '',
    addEventListener(type, listener) { this.listeners[type] = listener; },
    querySelector() { return null; },
    contains() { return false; },
  });
  let html = '';
  function card(markup) {
    const token = markup.match(/data-select-model="([^"]+)"/)[1];
    const node = element(token);
    node.dataset.selectModel = token;
    Object.defineProperty(node, 'outerHTML', {
      set(value) { card(value); },
      get() { return markup; },
    });
    nodes.set(token, node);
  }
  const app = {
    set innerHTML(value) {
      html = value;
      nodes.clear();
      for (const match of value.matchAll(/<input id="([^"]+)" value="([^"]*)"/g)) {
        nodes.set(match[1], element(match[1], match[2]));
      }
      for (const match of value.matchAll(/<select id="([^"]+)">([\s\S]*?)<\/select>/g)) {
        const selected = match[2].match(/<option value="([^"]*)" selected>/);
        nodes.set(match[1], element(match[1], selected?.[1] || ''));
      }
      const hotkey = value.match(/id="hotkey"[^>]*data-hotkey-value="([^"]+)"/);
      if (hotkey) {
        const node = element('hotkey');
        node.dataset.hotkeyValue = hotkey[1];
        nodes.set('hotkey', node);
      }
      for (const match of value.matchAll(/<div class="model-card[^>]+>/g)) card(match[0]);
    },
    querySelector() { return null; },
    querySelectorAll(selector) {
      if (selector === 'select[id], input[id]') {
        return [...nodes.values()].filter(node => !node.dataset.selectModel && node.id !== 'hotkey');
      }
      if (selector === '[data-select-model]') {
        return [...nodes.values()].filter(node => node.dataset.selectModel);
      }
      return [];
    },
  };
  const window = {
    webkit: { messageHandlers: { bridge: { postMessage() {} } } },
    addEventListener(type, listener) { keys.set(type, listener); },
    removeEventListener(type) { keys.delete(type); },
  };
  const context = vm.createContext({
    window, document: { getElementById: id => id === 'app' ? app : nodes.get(id) },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/ptarmigan_flow/webui/app.js'), 'utf8'), context);
  const run = script => vm.runInContext(script, context);
  const initial = {
    setup_required: false, daemon_running: false, strings: {},
    settings: {
      model: 'moonshine:tiny', language: 'en', hotkey: 'right_cmd',
      output_mode: 'direct_typing',
      llm_correction: { mode: 'always', provider: 'ollama', model: 'old', base_url: 'http://old' },
    },
    models: ['moonshine:tiny', 'moonshine:base'].map(token => ({ token, label: token, downloaded: false })),
  };
  run(`state = ${JSON.stringify(initial)}; render();`);
  return {
    run, nodes, window, initial, html: () => html,
    payload: () => JSON.parse(run('JSON.stringify(settingsPayload())')),
    edit(id, value, type = 'input') {
      const node = nodes.get(id);
      node.value = value;
      node.listeners[type]?.();
    },
    push(event, snapshot) { window.app.dispatch({ event, payload: snapshot }); },
    select(key = null) {
      const node = nodes.get('moonshine:base');
      if (key) node.listeners.keydown({ target: node, currentTarget: node, key, preventDefault() {} });
      else node.listeners.click();
    },
    hotkey() {
      run('beginHotkeyCaptureUi(document.getElementById("hotkey"), {}, {}, {})');
      keys.get('keydown')({ code: 'AltLeft', preventDefault() {} });
    },
  };
}

for (const key of [null, 'Enter', ' ']) {
  test(`model selection (${key || 'click'}) preserves every edited field`, () => {
    const f = fixture();
    f.edit('language', 'ja', 'change');
    f.edit('output_mode', 'clipboard_paste', 'change');
    f.edit('llm_mode', 'ask', 'change');
    f.edit('llm_provider', 'draft-provider');
    f.edit('llm_model', 'draft-model');
    f.edit('llm_base_url', '');
    f.hotkey();
    const expected = { ...f.payload(), model: 'moonshine:base' };
    f.select(key);
    assert.deepEqual(f.payload(), expected);
    assert.equal(f.nodes.get('llm_base_url').value, '');
    assert.match(f.nodes.get('moonshine:base').outerHTML, /model-card selected/);
    assert.equal(f.run('state.settings.model'), 'moonshine:tiny');
    assert.equal(f.run('state.settings.llm_correction.provider'), 'ollama');
  });
}

test('native snapshots retain dirty fields but refresh clean fields and model availability', () => {
  const f = fixture();
  f.edit('language', 'ja');
  f.select();
  const snapshot = structuredClone(f.initial);
  snapshot.daemon_running = true;
  snapshot.settings.llm_correction.provider = 'external-provider';
  snapshot.models[1].downloaded = true;
  for (const event of ['daemonState', 'permissionsChanged']) {
    f.push(event, snapshot);
    assert.equal(f.payload().language, 'ja');
    assert.equal(f.payload().model, 'moonshine:base');
    assert.equal(f.payload().llm_correction.provider, 'external-provider');
    assert.match(f.html(), /settings_model_downloaded_badge/);
    assert.match(f.html(), /brand-status running/);
  }
  f.push('routeChanged', { route: 'dictionary' });
  f.push('routeChanged', { route: 'settings' });
  assert.equal(f.payload().language, 'ja');
});

test('save sends the displayed draft and acknowledged edits allow subsequent external updates', async () => {
  const f = fixture();
  f.edit('language', 'ja');
  f.edit('llm_provider', '');
  f.select();
  const expected = f.payload();
  f.run(`bridge = async (action, payload) => {
    if (action === 'saveSettings') {
      if (JSON.stringify(payload) !== ${JSON.stringify(JSON.stringify(expected))}) throw Error('wrong payload');
      state = { ...state, settings: payload };
      return { saved: true };
    }
    return state;
  };`);
  await f.run('saveSettings()');
  assert.deepEqual(f.payload(), expected);
  assert.equal(f.run('Object.keys(settingsDraft).length'), 0);
  const snapshot = structuredClone(f.initial);
  snapshot.settings.language = 'zh';
  f.push('daemonState', snapshot);
  assert.equal(f.payload().language, 'zh');
  assert.equal(f.payload().model, 'moonshine:tiny');
});

test('download progress patches only the card and keeps the draft selection', () => {
  const f = fixture();
  f.edit('language', 'ja');
  f.select();
  const input = f.nodes.get('language');
  f.push('downloadProgress', { model: 'moonshine:base', type: 'progress', fraction: 0.42 });
  assert.equal(f.nodes.get('language'), input);
  assert.equal(f.payload().language, 'ja');
  assert.match(f.nodes.get('moonshine:base').outerHTML, /model-card selected/);
  assert.match(f.nodes.get('moonshine:base').outerHTML, /42%/);
});
