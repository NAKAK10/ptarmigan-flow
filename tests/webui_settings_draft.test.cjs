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
    let selected = /model-card selected/.test(markup);
    node.classList = { toggle(name, value) { if (name === 'selected') selected = value; } };
    node.focus = () => { context.document.activeElement = node; };
    const status = { innerHTML: markup.match(/aria-atomic="true">([\s\S]*?)<\/div>/)[1] };
    let button = null;
    const buttonMarkup = markup.match(/<button[^>]+data-download-model[^>]*>([^<]*)<\/button>/);
    if (buttonMarkup) {
      button = element(token);
      button.dataset.downloadModel = token;
      button.textContent = buttonMarkup[1];
      button.disabled = /disabled/.test(buttonMarkup[0]);
    }
    let progress = markup.match(/<div class="model-progress-row">[\s\S]*?\n      <\/div>/)?.[0];
    const progressNode = value => ({
      markup: value,
      isEqualNode(other) { return value === other.markup; },
      replaceWith(other) { progress = other.markup; },
      remove() { progress = null; },
    });
    node.querySelector = selector => {
      if (selector === '[role="status"]') return status;
      if (selector === '[data-download-model]') {
        if (button) button.remove = () => { button = null; };
        return button;
      }
      if (selector === '.model-action') return { append(value) { button = value; } };
      if (selector === '.model-progress-row') return progress ? progressNode(progress) : null;
      return null;
    };
    node.append = value => { progress = value.markup; };
    Object.defineProperty(node, 'outerHTML', {
      get() {
        return `<div class="model-card ${selected ? 'selected' : ''}">${status.innerHTML}${progress || ''}</div>`;
      },
    });
    return node;
  }
  const brand = { innerHTML: '', running: false, classList: { toggle(name, value) { brand.running = value; } } };
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
      for (const match of value.matchAll(/<div class="model-card[\s\S]*?\n    <\/div>/g)) {
        const node = card(match[0]);
        nodes.set(node.dataset.selectModel, node);
      }
    },
    querySelector(selector) { return selector === '.brand-status' ? brand : null; },
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
    window, document: {
      getElementById: id => id === 'app' ? app : nodes.get(id),
      createElement() {
        return { content: {}, set innerHTML(value) { this.content.firstElementChild = card(value); } };
      },
    },
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
    run, nodes, window, initial,
    html: () => html + [...nodes.values()].filter(node => node.dataset.selectModel).map(node => node.outerHTML).join('') + (brand.running ? 'brand-status running' : ''),
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
      return { saved: true, settings: payload };
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

test('normalized save acknowledges sent LLM drafts and allows external updates', async () => {
  const f = fixture();
  f.edit('llm_provider', ' openai ');
  f.edit('llm_model', ' demo-model ');
  f.edit('llm_base_url', ' http://localhost:12345 ');
  f.select();
  f.run(`bridge = async (action, payload) => {
    if (action === 'saveSettings') {
      const settings = { ...payload, llm_correction: { ...payload.llm_correction } };
      for (const field of ['provider', 'model', 'base_url']) {
        settings.llm_correction[field] = settings.llm_correction[field].trim();
      }
      state = { ...state, settings };
      return { saved: true, settings };
    }
    return state;
  };`);
  await f.run('saveSettings()');
  assert.equal(f.run('Object.keys(settingsDraft).length'), 0);
  assert.deepEqual(f.payload().llm_correction, {
    mode: 'always', provider: 'openai', model: 'demo-model', base_url: 'http://localhost:12345',
  });
  const snapshot = structuredClone(f.initial);
  snapshot.settings.llm_correction = {
    mode: 'ask', provider: 'external-provider', model: 'external-model', base_url: 'http://external',
  };
  f.push('daemonState', snapshot);
  assert.deepEqual(f.payload().llm_correction, snapshot.settings.llm_correction);
});

test('save acknowledgement does not discard edits made while saving', async () => {
  const f = fixture();
  f.edit('llm_provider', ' openai ');
  f.edit('llm_model', ' demo-model ');
  f.run(`bridge = (action, payload) => {
    if (action === 'saveSettings') {
      const settings = { ...payload, llm_correction: {
        ...payload.llm_correction, provider: 'openai', model: 'demo-model',
      } };
      state = { ...state, settings };
      return new Promise(resolve => { finishSave = () => resolve({ saved: true, settings }); });
    }
    return Promise.resolve(state);
  };`);
  const saving = f.run('saveSettings()');
  f.edit('llm_provider', 'new-unsaved-provider');
  f.run('finishSave()');
  await saving;
  assert.equal(f.payload().llm_correction.provider, 'new-unsaved-provider');
  assert.equal(f.payload().llm_correction.model, 'demo-model');
  assert.equal(f.run('JSON.stringify(settingsDraft)'), '{"llm_provider":"new-unsaved-provider"}');
});

test('failed save leaves the sent draft intact', async () => {
  for (const response of ["return { saved: false, errors: ['llm_provider'] };", "throw Error('save failed');"]) {
    const f = fixture();
    f.edit('llm_provider', ' openai ');
    f.run(`bridge = async () => { ${response} };`);
    await f.run('saveSettings()');
    assert.equal(f.payload().llm_correction.provider, ' openai ');
    assert.equal(f.run('settingsDraft.llm_provider'), ' openai ');
  }
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

for (const alreadyDownloaded of [false, true]) {
  test(`${alreadyDownloaded ? 'already-downloaded' : 'done'} sync preserves draft selection, mounted inputs/cards and clean external settings`, async () => {
    const f = fixture();
    f.edit('language', 'ja');
    f.edit('llm_provider', 'draft-provider');
    f.edit('llm_base_url', '');
    f.select('Enter');
    const input = f.nodes.get('language');
    const selectedCard = f.nodes.get('moonshine:base');
    const downloadingCard = f.nodes.get('moonshine:tiny');
    const liveStatus = downloadingCard.querySelector('[role="status"]');
    selectedCard.focus();
    const snapshot = structuredClone(f.initial);
    snapshot.daemon_running = true;
    snapshot.models[0].downloaded = true;
    snapshot.settings.output_mode = 'clipboard_paste';
    snapshot.settings.hotkey = 'left_shift';
    snapshot.settings.llm_correction = {
      mode: 'ask', provider: 'external-provider', model: 'external-model', base_url: 'http://external',
    };
    f.run(`bridge = async action => action === 'getState' ? ${JSON.stringify(snapshot)} :
      ${JSON.stringify(alreadyDownloaded ? { started: false, already_downloaded: true } : { started: true })};`);
    const start = f.run(`handleDownloadModelClick({ stopPropagation() {}, currentTarget: {
      dataset: { downloadModel: 'moonshine:tiny' },
    } })`);
    assert.equal(f.run('activeDownloadToken'), 'moonshine:tiny');
    assert.equal(selectedCard.querySelector('[data-download-model]').disabled, true);
    assert.equal(f.nodes.get('moonshine:base'), selectedCard);
    if (!alreadyDownloaded) {
      await start;
      f.push('downloadProgress', { model: 'moonshine:tiny', type: 'progress', fraction: 0.42 });
      assert.match(downloadingCard.outerHTML, /42%/);
      assert.match(selectedCard.outerHTML, /model-card selected/);
      f.push('downloadProgress', { model: 'moonshine:tiny', type: 'done' });
      await new Promise(resolve => setImmediate(resolve));
    } else {
      await start;
    }
    assert.equal(f.run('activeDownloadToken'), null);
    assert.equal(selectedCard.querySelector('[data-download-model]').disabled, false);
    assert.equal(f.run('state.models[0].downloaded'), true);
    f.push('daemonState', snapshot);
    assert.deepEqual(f.payload(), {
      model: 'moonshine:base', language: 'ja', hotkey: 'left_shift', output_mode: 'clipboard_paste',
      llm_correction: { mode: 'ask', provider: 'draft-provider', model: 'external-model', base_url: '' },
    });
    assert.equal(f.run('state.settings.model'), 'moonshine:tiny');
    assert.equal(f.nodes.get('language'), input);
    assert.equal(f.nodes.get('moonshine:base'), selectedCard);
    assert.equal(f.nodes.get('moonshine:tiny'), downloadingCard);
    assert.equal(downloadingCard.querySelector('[role="status"]'), liveStatus);
    assert.equal(f.run('document.activeElement.dataset.selectModel'), 'moonshine:base');
    assert.match(selectedCard.outerHTML, /model-card selected/);
    assert.match(downloadingCard.outerHTML, /settings_model_downloaded_badge/);
    assert.match(f.html(), /brand-status running/);
  });
}

test('daemon snapshot acknowledges matching drafts without losing a captured hotkey', () => {
  const f = fixture();
  f.edit('language', 'ja');
  f.hotkey();
  const hotkey = f.nodes.get('hotkey');
  const snapshot = structuredClone(f.initial);
  snapshot.settings.language = 'ja';
  snapshot.settings.hotkey = 'left_shift';
  f.push('daemonState', snapshot);
  assert.equal(f.payload().hotkey, 'left_alt');
  assert.equal(f.nodes.get('hotkey'), hotkey);
  assert.equal(f.run('settingsDraft.language'), undefined);
  snapshot.settings.language = 'zh';
  f.push('daemonState', snapshot);
  assert.equal(f.payload().language, 'zh');
});
