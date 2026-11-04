// Dependency-free DOM fixture: real app.js listeners/rendering, mocked native bridge.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const { test } = require('node:test');

function fixture({ glossary = [], mode = 'always', initialRoute = 'settings' } = {}) {
  const nodes = new Map();
  const keys = new Map();
  const element = (id, value = '') => ({
    id, value, dataset: {}, listeners: {}, textContent: '', className: '', attrs: {},
    selectionStart: 0, selectionEnd: 0,
    focus() { context.document.activeElement = this; },
    setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; },
    setAttribute(name, value) { this.attrs[name] = value; },
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
      // Replacing a real DOM subtree drops focus from its old input nodes.
      context.document.activeElement = null;
      nodes.clear();
      for (const scope of ['settings', 'glossary', 'dictionary']) {
        const status = value.match(new RegExp(`id="${scope}-error"[^>]*>([^<]*)`));
        if (status) {
          const node = element(`${scope}-error`);
          node.textContent = status[1];
          nodes.set(node.id, node);
        }
        const save = value.match(new RegExp(`(<button[^>]*data-action="save-${scope}"[^>]*>)([^<]*)`));
        if (save) {
          const node = element(`save-${scope}`);
          node.disabled = save[1].includes('disabled');
          node.attrs['aria-busy'] = save[1].match(/aria-busy="([^"]+)"/)[1];
          node.attrs['aria-describedby'] = save[1].match(/aria-describedby="([^"]*)"/)[1];
          node.textContent = save[2];
          node.className = save[1].match(/class="([^"]*)"/)[1];
          nodes.set(node.id, node);
        }
        const unsaved = value.match(new RegExp(`id="${scope}-unsaved" class="([^"]*)">([^<]*)`));
        if (unsaved) {
          const node = element(`${scope}-unsaved`);
          node.className = unsaved[1];
          node.textContent = unsaved[2];
          nodes.set(node.id, node);
        }
      }
      for (const match of value.matchAll(/<button[^>]*data-route="([^"]+)"[^>]*>([^<]*)/g)) {
        const node = element(`nav-${match[1]}`);
        node.dataset.route = match[1];
        node.textContent = match[2];
        nodes.set(node.id, node);
      }
      for (const match of value.matchAll(/<input id="([^"]+)" value="([^"]*)"/g)) {
        nodes.set(match[1], element(match[1], match[2]));
      }
      for (const match of value.matchAll(/<textarea id="([^"]+)"[^>]*>([\s\S]*?)<\/textarea>/g)) {
        // HTML textarea parsing discards the first newline after the start tag.
        const text = match[2].replace(/^\n/, '').replaceAll('&quot;', '"').replaceAll('&lt;', '<')
          .replaceAll('&gt;', '>').replaceAll('&amp;', '&');
        nodes.set(match[1], element(match[1], text));
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
      for (const match of value.matchAll(/<div class="dictionary-row[^\"]*" data-dictionary-row="(\d+)" data-section="([^"]+)">([\s\S]*?)\n    <\/div>/g)) {
        const row = element(`dict-row-${match[1]}`);
        row.dataset = { dictionaryRow: match[1], section: match[2] };
        row.classList = { remove() {} };
        const key = element(`dict-key-${match[1]}`, match[3].match(/data-dictionary-key value="([^"]*)"/)[1]);
        const values = nodes.get(`dict-values-${match[1]}`);
        const remove = element(`dict-delete-${match[1]}`);
        row.querySelector = selector => ({
          '[data-dictionary-key]': key, '[data-dictionary-values]': values,
          '[data-delete-row]': remove,
        })[selector] || null;
        for (const node of [row, key, remove]) nodes.set(node.id, node);
      }
      for (const match of value.matchAll(/data-add-dictionary="([^"]+)"/g)) {
        const node = element(`dict-add-${match[1]}`);
        node.dataset.addDictionary = match[1];
        nodes.set(node.id, node);
      }
    },
    querySelector(selector) {
      if (selector === '.brand-status') return brand;
      const action = selector.match(/data-action=['"]([^'"]+)/);
      if (action) return nodes.get(action[1]);
      const row = selector.match(/data-dictionary-row="(\d+)"/);
      if (row) return nodes.get(`dict-row-${row[1]}`);
      return null;
    },
    querySelectorAll(selector) {
      if (selector === 'select[id], input[id]') {
        return [...nodes.values()].filter(node => ['language', 'output_mode', 'llm_mode', 'llm_provider', 'llm_model', 'llm_base_url'].includes(node.id));
      }
      if (selector === '[data-route]') {
        return [...nodes.values()].filter(node => node.dataset.route);
      }
      if (selector === '[data-select-model]') {
        return [...nodes.values()].filter(node => node.dataset.selectModel);
      }
      if (selector === '[data-dictionary-row]') {
        return [...nodes.values()].filter(node => node.dataset.dictionaryRow);
      }
      if (selector === '[data-add-dictionary]') {
        return [...nodes.values()].filter(node => node.dataset.addDictionary);
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
      activeElement: null,
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
      llm_correction: { mode, provider: 'ollama', model: 'old', base_url: 'http://old', glossary },
    },
    models: ['moonshine:tiny', 'moonshine:base'].map(token => ({ token, label: token, downloaded: false })),
    dictionary: { exact: { Stored: ['stored'] }, regex: {} },
  };
  run(`state = ${JSON.stringify(initial)}; route = ${JSON.stringify(initialRoute)}; render();`);
  return {
    run, nodes, window, initial,
    html: () => html + [...nodes.values()].filter(node => node.dataset.selectModel).map(node => node.outerHTML).join('') + (brand.running ? 'brand-status running' : ''),
    payload: () => JSON.parse(run('JSON.stringify(settingsPayload())')),
    glossaryPayload: () => JSON.parse(run('JSON.stringify(glossaryPayload())')),
    nav(target) { nodes.get(`nav-${target}`).listeners.click(); },
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
    hotkey(code = 'AltLeft') {
      run('beginHotkeyCaptureUi(document.getElementById("hotkey"), {}, {}, {})');
      keys.get('keydown')({ code, preventDefault() {} });
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

test('pending save and download completion retain new edits, acknowledge normalized drafts and wait for refresh', async () => {
  const f = fixture();
  f.edit('llm_provider', ' openai ');
  f.edit('llm_model', ' demo-model ');
  f.edit('llm_base_url', ' http://localhost:12345 ');
  f.select();
  f.run(`
    let saveCalls = 0;
    let refreshCalls = 0;
    let savedSettings;
    bridge = (action, payload) => {
      if (action === 'saveSettings') {
        saveCalls++;
        savedSettings = { ...payload, llm_correction: { ...payload.llm_correction } };
        for (const field of ['provider', 'model', 'base_url']) {
          savedSettings.llm_correction[field] = savedSettings.llm_correction[field].trim();
        }
        return new Promise(resolve => { finishSave = () => resolve({ saved: true, settings: savedSettings }); });
      }
      if (action === 'downloadModel') return Promise.resolve({ started: true });
      refreshCalls++;
      return new Promise(resolve => {
        const finish = () => resolve({ ...state, settings: savedSettings });
        if (refreshCalls === 1) finishSaveRefresh = finish;
        else finishRefresh = finish;
      });
    };
  `);
  const saving = f.run('saveSettings()');
  const input = f.nodes.get('llm_provider');
  const pending = () => {
    assert.equal(f.nodes.get('save-settings').disabled, true);
    assert.equal(f.nodes.get('save-settings').attrs['aria-busy'], 'true');
    assert.equal(f.nodes.get('settings-error').textContent, 'save_in_progress');
  };
  pending();
  await f.run('saveSettings()');
  assert.equal(f.run('saveCalls'), 1);
  await f.run(`handleDownloadModelClick({ stopPropagation() {}, currentTarget: {
    dataset: { downloadModel: 'moonshine:tiny' },
  } })`);
  f.edit('llm_provider', 'new-unsaved-provider');
  f.push('downloadProgress', { model: 'moonshine:tiny', type: 'progress', fraction: 0.42 });
  const snapshot = structuredClone(f.initial);
  snapshot.models[0].downloaded = true;
  snapshot.settings.output_mode = 'clipboard_paste';
  f.push('daemonState', snapshot);
  assert.equal(f.nodes.get('llm_provider'), input);
  pending();
  f.run('finishSave()');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.run('refreshCalls'), 1);
  assert.equal(f.run('settingsDraft.llm_provider'), 'new-unsaved-provider');
  assert.equal(f.run('settingsDraft.llm_model'), ' demo-model '); // Retained until refresh succeeds.
  pending();
  f.edit('llm_model', 'new-unsaved-model'); // Also preserve edits during getState.
  f.push('downloadProgress', { model: 'moonshine:tiny', type: 'done' });
  assert.equal(f.run('activeDownloadToken'), null);
  assert.equal(f.run('refreshCalls'), 2);
  f.run('finishRefresh()'); // Download availability refresh must not report save success.
  await new Promise(resolve => setImmediate(resolve));
  pending();
  await f.run('saveSettings()');
  assert.equal(f.run('saveCalls'), 1);
  // Resolve the save's earlier getState separately from the download refresh.
  f.push('routeChanged', { route: 'settings' });
  pending();
  f.run('finishSaveRefresh()');
  await saving;
  assert.equal(f.nodes.get('save-settings').disabled, false);
  assert.equal(f.nodes.get('settings-error').textContent, 'settings_saved_message');
  assert.equal(f.payload().llm_correction.provider, 'new-unsaved-provider');
  assert.equal(f.payload().llm_correction.model, 'new-unsaved-model');
  assert.equal(f.payload().llm_correction.base_url, 'http://localhost:12345');
  snapshot.settings.llm_correction.base_url = 'http://external';
  f.push('daemonState', snapshot);
  assert.equal(f.payload().llm_correction.base_url, 'http://external');
  assert.equal(f.payload().llm_correction.provider, 'new-unsaved-provider');
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

test('matching native snapshots preserve unsaved settings and a captured hotkey', () => {
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
  assert.equal(f.run('settingsDraft.language'), 'ja');
  snapshot.settings.language = 'zh';
  f.push('daemonState', snapshot);
  assert.equal(f.payload().language, 'ja');
});

test('glossary loads preferred spellings with an accessible editable field even when LLM is off', () => {
  const glossary = ['PtarmiganFlow', '雷鳥', '<term>&"'];
  const f = fixture({ glossary, mode: 'never', initialRoute: 'glossary' });
  assert.equal(f.nodes.get('llm_glossary').value, glossary.join('\n'));
  assert.deepEqual(f.glossaryPayload().llm_correction.glossary, glossary);
  assert.match(f.html(), /<label for="llm_glossary">settings_llm_glossary_label<\/label>/);
  assert.match(f.html(), /<textarea id="llm_glossary"[^>]*aria-describedby="llm_glossary_help"/);
  assert.match(f.html(), /id="llm_glossary_help" class="form-help">settings_llm_glossary_help/);
  assert.doesNotMatch(f.html().match(/<textarea id="llm_glossary"[^>]*>/)[0], /disabled|readonly/);
  assert.match(f.html(), /&lt;term&gt;&amp;&quot;/);
  f.edit('llm_glossary', 'Added');
  assert.deepEqual(f.glossaryPayload().llm_correction.glossary, ['Added']);
  const clean = fixture({ initialRoute: 'glossary' });
  const snapshot = structuredClone(clean.initial);
  snapshot.settings.llm_correction.glossary = ['External', '外部'];
  clean.push('daemonState', snapshot);
  assert.equal(clean.nodes.get('llm_glossary').value, 'External\n外部');
});

test('glossary edits add, rename, remove and explicitly clear terms in first-occurrence order', () => {
  const f = fixture({ glossary: ['Original', 'Remove me'], initialRoute: 'glossary' });
  f.edit('llm_glossary', 'Original\nRemove me\n Added ');
  assert.deepEqual(f.glossaryPayload().llm_correction.glossary, ['Original', 'Remove me', 'Added']);
  f.edit('llm_glossary', ' Renamed \r\n\nAdded\rRenamed\nadded\n 日本語 \n\t\nAdded ');
  assert.deepEqual(f.glossaryPayload().llm_correction.glossary, ['Renamed', 'Added', 'added', '日本語']);
  f.edit('llm_glossary', 'Added', 'change');
  assert.deepEqual(f.glossaryPayload().llm_correction.glossary, ['Added']);
  for (const empty of ['', ' \n\t\r\n']) {
    f.edit('llm_glossary', empty);
    assert.deepEqual(f.glossaryPayload().llm_correction.glossary, []);
    assert.ok(Object.hasOwn(f.glossaryPayload().llm_correction, 'glossary'));
  }
  assert.deepEqual(f.initial.settings.llm_correction.glossary, ['Original', 'Remove me']);
});

test('raw glossary blank lines survive matching and differing pushes, rerenders and navigation', () => {
  const f = fixture({ glossary: ['Original'], initialRoute: 'glossary' });
  const text = '\n\n Original \n\n\n';
  f.edit('llm_glossary', text);
  const input = f.nodes.get('llm_glossary');
  const snapshot = structuredClone(f.initial);
  // Even a value-equal native array must not acknowledge an unsaved raw edit.
  f.push('daemonState', snapshot);
  assert.equal(f.nodes.get('llm_glossary'), input);
  assert.equal(input.value, text);
  snapshot.settings.llm_correction.glossary = ['External'];
  snapshot.settings.llm_correction.provider = 'external-provider';
  for (const event of ['daemonState', 'permissionsChanged']) {
    f.push(event, snapshot);
    assert.equal(f.nodes.get('llm_glossary').value, text);
    assert.equal(f.payload().llm_correction.provider, 'external-provider');
  }
  f.push('routeChanged', { route: 'dictionary' });
  assert.deepEqual(f.glossaryPayload().llm_correction.glossary, ['Original']);
  f.push('daemonState', snapshot);
  f.push('routeChanged', { route: 'glossary' });
  assert.equal(f.nodes.get('llm_glossary').value, text);
});

for (const failure of ['validation', 'transport', 'refresh']) {
  test(`glossary ${failure} failure preserves raw text and supports a successful retry including clear`, async () => {
    const f = fixture({ glossary: ['Original'], initialRoute: 'glossary' });
    const text = failure === 'validation' ? ' \n\n' : ' Added \n\nAdded\nSecond\n';
    const expected = failure === 'validation' ? [] : ['Added', 'Second'];
    f.edit('llm_glossary', text);
    f.run(`
      state.strings.settings_validation_error = 'Invalid: {fields}';
      let saves = [];
      let savedSettings;
      let fail = true;
      bridge = async (action, payload) => {
        if (action === 'saveSettings') {
          saves.push(payload);
          if (fail && ${JSON.stringify(failure)} === 'validation') {
            return { saved: false, errors: ['llm_correction.glossary'] };
          }
          if (fail && ${JSON.stringify(failure)} === 'transport') throw Error('write failed');
          savedSettings = JSON.parse(JSON.stringify(payload));
          return { saved: true, settings: savedSettings };
        }
        if (fail) throw Error('refresh failed');
        return { ...state, settings: savedSettings };
      };
    `);
    await f.nodes.get('save-glossary').listeners.click();
    assert.equal(f.nodes.get('save-glossary').disabled, false);
    assert.equal(f.nodes.get('save-glossary').textContent, 'save_retry_button');
    assert.equal(f.run('glossaryMessage.kind'), 'error');
    assert.match(f.nodes.get('glossary-error').textContent,
      failure === 'validation' ? /settings_llm_glossary_label/ : /failed/);
    f.push('permissionsChanged', structuredClone(f.initial));
    assert.equal(f.nodes.get('llm_glossary').value, text);
    f.run('fail = false');
    await f.nodes.get('save-glossary').listeners.click();
    assert.equal(f.run('saves.length'), 2);
    assert.deepEqual(JSON.parse(f.run('JSON.stringify(saves.map(p => p.llm_correction.glossary))')),
      [expected, expected]);
    assert.equal(f.nodes.get('llm_glossary').value, expected.join('\n'));
    assert.equal(f.run('glossaryDraftText'), null);
    assert.equal(f.run('Object.keys(settingsDraft).length'), 0);
    assert.equal(f.run('glossaryMessage.kind'), 'success');
  });
}

for (const stage of ['save', 'refresh']) {
  for (const newer of [' Newer \n\n', ' Sent \n\nSent\n', ' \n\n']) {
    test(`newer glossary edits survive stale ACK/pushes during ${stage}: ${JSON.stringify(newer)}`, async () => {
      const f = fixture({ glossary: ['Original'], initialRoute: 'glossary' });
      f.edit('llm_glossary', 'Sent');
      f.run(`
        let savedSettings;
        let saveCalls = 0;
        bridge = (action, payload) => {
          if (action === 'saveSettings') {
            saveCalls++;
            savedSettings = JSON.parse(JSON.stringify(payload));
            return new Promise(resolve => { finishSave = () => resolve({ saved: true, settings: savedSettings }); });
          }
          return new Promise(resolve => { finishRefresh = () => resolve({ ...state, settings: savedSettings }); });
        };
      `);
      const saving = f.run('saveGlossary()');
      if (stage === 'refresh') {
        f.run('finishSave()');
        await new Promise(resolve => setImmediate(resolve));
      }
      f.edit('llm_glossary', newer);
      const expected = f.glossaryPayload().llm_correction.glossary;
      const snapshot = structuredClone(f.initial);
      snapshot.settings.llm_correction.glossary = expected;
      f.push('daemonState', snapshot); // Matching push must not clear the newer draft.
      snapshot.settings.llm_correction.glossary = ['Original'];
      f.push('permissionsChanged', snapshot);
      assert.equal(f.nodes.get('llm_glossary').value, newer);
      if (stage === 'save') {
        f.run('finishSave()');
        await new Promise(resolve => setImmediate(resolve));
      }
      f.push('routeChanged', { route: 'dictionary' });
      f.push('routeChanged', { route: 'glossary' });
      assert.equal(f.nodes.get('save-glossary').disabled, true);
      await f.run('saveGlossary()');
      assert.equal(f.run('saveCalls'), 1);
      f.run('finishRefresh()');
      await saving;
      f.push('daemonState', snapshot);
      assert.equal(f.nodes.get('llm_glossary').value, newer);
      assert.deepEqual(f.glossaryPayload().llm_correction.glossary, expected);
      assert.equal(f.run('glossarySaving'), false);
    });
  }
}

function editScreen(f, scope, value) {
  const id = scope === 'settings' ? 'llm_base_url' : scope === 'glossary' ? 'llm_glossary' :
    `dict-values-${f.run('dictDraft.exact[0].id')}`;
  f.edit(id, value);
}

function screenValue(f, scope) {
  const id = scope === 'settings' ? 'llm_base_url' : scope === 'glossary' ? 'llm_glossary' :
    `dict-values-${f.run('dictDraft.exact[0].id')}`;
  return f.nodes.get(id).value;
}

function assertSaveChanged(f, scope, changed, pending = false) {
  const button = f.nodes.get(`save-${scope}`);
  assert.equal(f.run(`saveHasChanges('${scope}')`), changed);
  assert.equal(button.className.includes('has-changes'), changed && !pending);
  assert.equal(f.nodes.get(`${scope}-unsaved`).className.includes('hidden'), !changed || pending);
  assert.equal(button.attrs['aria-describedby'], changed && !pending ? `${scope}-unsaved` : '');
  assert.equal(button.disabled, pending);
}

function savedScreenSnapshot(f, scope, value) {
  const snapshot = structuredClone(f.initial);
  if (scope === 'settings') snapshot.settings.llm_correction.base_url = value;
  else if (scope === 'glossary') snapshot.settings.llm_correction.glossary = value ? [value] : [];
  else snapshot.dictionary.exact.Stored = value ? [value] : [];
  return snapshot;
}

for (const scope of ['settings', 'dictionary', 'glossary']) {
  test(`${scope} has exactly one top header save with live feedback before the editor`, () => {
    const f = fixture({ glossary: ['Stored'], initialRoute: scope });
    const header = f.html().match(/<header class="panel full screen-save-header"[\s\S]*?<\/header>/)[0];
    assert.ok(header.includes(`data-save-header="${scope}"`));
    assert.match(header, /<h2>[^<]+<\/h2>[\s\S]*class="screen-save-actions"/);
    assert.ok(header.includes(`data-action="save-${scope}"`));
    assert.equal(f.html().match(new RegExp(`data-action="save-${scope}"`, 'g')).length, 1);
    assert.ok(header.includes(`id="${scope}-error"`));
    assert.match(header, /role="status" aria-live="polite" aria-atomic="true"/);
    assert.match(f.html(), /<section class="grid">\s*<header /);
    assert.ok(!f.html().includes('action-bar'));
    assertSaveChanged(f, scope, false);
  });

  test(`${scope} save indication follows semantic edit/revert/clear without autosaving`, () => {
    const f = fixture({ glossary: ['Stored'], initialRoute: scope });
    const original = screenValue(f, scope);
    f.run('let writes = 0; bridge = async () => { writes++; };');
    editScreen(f, scope, 'http://edited');
    assertSaveChanged(f, scope, true);
    f.nav(scope === 'settings' ? 'glossary' : 'settings');
    f.nav(scope);
    assertSaveChanged(f, scope, true);
    editScreen(f, scope, original);
    assertSaveChanged(f, scope, false);
    editScreen(f, scope, '');
    assertSaveChanged(f, scope, true);
    editScreen(f, scope, ` ${original} \n`);
    assertSaveChanged(f, scope, false); // Presentation whitespace has no saved effect.
    f.nav(scope === 'settings' ? 'glossary' : 'settings');
    f.nav(scope);
    assert.equal(screenValue(f, scope), ` ${original} \n`);
    assertSaveChanged(f, scope, false);
    editScreen(f, scope, 'http://edited');
    for (const event of ['daemonState', 'permissionsChanged']) {
      f.push(event, structuredClone(f.initial));
      assert.equal(screenValue(f, scope), 'http://edited');
      assertSaveChanged(f, scope, true);
    }
    assert.equal(f.run('writes'), 0);
  });

  for (const newerStage of [null, 'save', 'refresh']) {
    test(`${scope} pending save suppresses animation and ${newerStage ? `retains newer ${newerStage} edits` : 'cleans acknowledged edits'}`, async () => {
      const f = fixture({ glossary: ['Stored'], initialRoute: scope });
      editScreen(f, scope, 'http://sent');
      const saved = savedScreenSnapshot(f, scope, 'http://sent');
      f.run(`
        let requests = [];
        const savedSnapshot = ${JSON.stringify(saved)};
        bridge = (action, payload) => {
          requests.push({ action, payload });
          if (action !== 'getState') return new Promise(resolve => {
            finishSave = () => resolve({ saved: true, settings: savedSnapshot.settings });
          });
          return new Promise(resolve => { finishRefresh = () => resolve(savedSnapshot); });
        };
      `);
      const saving = f.nodes.get(`save-${scope}`).listeners.click();
      assertSaveChanged(f, scope, true, true);
      await f.nodes.get(`save-${scope}`).listeners.click();
      assert.equal(f.run('requests.length'), 1);
      if (newerStage === 'save') editScreen(f, scope, 'http://newer');
      f.run('finishSave()');
      await new Promise(resolve => setImmediate(resolve));
      assertSaveChanged(f, scope, true, true);
      if (newerStage === 'refresh') editScreen(f, scope, 'http://newer');
      f.push('permissionsChanged', structuredClone(f.initial));
      assertSaveChanged(f, scope, true, true);
      f.run('finishRefresh()');
      await saving;
      assert.equal(screenValue(f, scope), newerStage ? 'http://newer' : 'http://sent');
      assertSaveChanged(f, scope, Boolean(newerStage));
      assert.equal(f.nodes.get(`${scope}-error`).textContent, `${scope}_saved_message`);
      f.nav(scope === 'settings' ? 'dictionary' : 'settings');
      f.nav(scope);
      assertSaveChanged(f, scope, Boolean(newerStage));
      editScreen(f, scope, 'http://sent');
      assertSaveChanged(f, scope, false);
    });
  }

  for (const failure of ['validation', 'transport', 'refresh']) {
    test(`${scope} ${failure} failure keeps the changed indication and retry clears it`, async () => {
      const f = fixture({ glossary: ['Stored'], initialRoute: scope });
      editScreen(f, scope, 'http://sent');
      const saved = savedScreenSnapshot(f, scope, 'http://sent');
      const errors = scope === 'dictionary' ? [{ section: 'exact', key: 'Stored', message: 'invalid rule' }] :
        [scope === 'settings' ? 'llm_correction.base_url' : 'llm_correction.glossary'];
      f.run(`
        let fail = true;
        const savedSnapshot = ${JSON.stringify(saved)};
        bridge = async action => {
          if (action !== 'getState') {
            if (fail && ${JSON.stringify(failure)} === 'transport') throw Error('write failed');
            if (fail && ${JSON.stringify(failure)} === 'validation') return { saved: false, errors: ${JSON.stringify(errors)} };
            return { saved: true, settings: savedSnapshot.settings };
          }
          if (fail) throw Error('refresh failed');
          return savedSnapshot;
        };
      `);
      await f.nodes.get(`save-${scope}`).listeners.click();
      assertSaveChanged(f, scope, true);
      assert.equal(f.nodes.get(`save-${scope}`).textContent, 'save_retry_button');
      f.push('permissionsChanged', structuredClone(f.initial));
      assert.equal(screenValue(f, scope), 'http://sent');
      assertSaveChanged(f, scope, true);
      f.run('fail = false');
      await f.nodes.get(`save-${scope}`).listeners.click();
      assertSaveChanged(f, scope, false);
    });
  }
}

test('model, hotkey and select changes update the same settings save indication', () => {
  const f = fixture();
  f.select();
  assertSaveChanged(f, 'settings', true);
  f.nodes.get('moonshine:tiny').listeners.click();
  assertSaveChanged(f, 'settings', false);
  f.hotkey();
  assertSaveChanged(f, 'settings', true);
  f.hotkey('MetaRight');
  assertSaveChanged(f, 'settings', false);
  for (const [id, edited, original] of [
    ['language', 'ja', 'en'], ['output_mode', 'clipboard_paste', 'direct_typing'],
    ['llm_mode', 'never', 'always'],
  ]) {
    f.edit(id, edited, 'change');
    assertSaveChanged(f, 'settings', true);
    f.edit(id, original, 'change');
    assertSaveChanged(f, 'settings', false);
  }
});

test('dictionary CRUD has semantic dirty state while preserving raw rows across routes', () => {
  const f = fixture({ initialRoute: 'dictionary' });
  const stored = f.run('dictDraft.exact[0].id');
  f.nodes.get('dict-add-exact').listeners.click();
  const added = f.run('dictDraft.exact[1].id');
  assertSaveChanged(f, 'dictionary', true);
  f.nodes.get(`dict-delete-${added}`).listeners.click();
  assertSaveChanged(f, 'dictionary', false);
  f.nodes.get(`dict-delete-${stored}`).listeners.click();
  assertSaveChanged(f, 'dictionary', true);
  f.nodes.get('dict-add-exact').listeners.click();
  const replacement = f.run('dictDraft.exact[0].id');
  f.edit(`dict-key-${replacement}`, 'Stored');
  f.edit(`dict-values-${replacement}`, ' stored \n');
  assertSaveChanged(f, 'dictionary', false);
  f.nav('glossary');
  f.nav('dictionary');
  assert.equal(f.nodes.get(`dict-values-${replacement}`).value, ' stored \n');
  assertSaveChanged(f, 'dictionary', false);
  f.nodes.get('dict-add-regex').listeners.click();
  assertSaveChanged(f, 'dictionary', true);
  const regex = f.run('dictDraft.regex[0].id');
  f.nodes.get(`dict-delete-${regex}`).listeners.click();
  assertSaveChanged(f, 'dictionary', false);
});

test('dictionary matching order and duplicate keys count as unsaved content', () => {
  const f = fixture({ initialRoute: 'dictionary' });
  f.run(`
    state.dictionary.regex = { First: ['one'], Second: ['two'] };
    dictDraft = null;
    render();
    dictDraft.regex.reverse();
    dictDraft.edited = true;
    updateSaveFeedback('dictionary');
  `);
  assertSaveChanged(f, 'dictionary', true);
  f.run(`dictDraft.regex.reverse(); updateSaveFeedback('dictionary');`);
  assertSaveChanged(f, 'dictionary', false);
  f.nodes.get('dict-add-exact').listeners.click();
  const row = f.run('dictDraft.exact[1].id');
  f.edit(`dict-key-${row}`, 'Stored');
  f.edit(`dict-values-${row}`, 'stored');
  assertSaveChanged(f, 'dictionary', true);
  f.nodes.get(`dict-delete-${row}`).listeners.click();
  assertSaveChanged(f, 'dictionary', false);
});

test('untouched saved exact whitespace starts clean and follows native rule updates', () => {
  const f = fixture();
  const snapshot = structuredClone(f.initial);
  snapshot.dictionary.exact = { Stored: [' stored '] };
  f.push('daemonState', snapshot);
  f.nav('dictionary');
  let id = `dict-values-${f.run('dictDraft.exact[0].id')}`;
  assert.equal(f.nodes.get(id).value, ' stored ');
  assertSaveChanged(f, 'dictionary', false);
  assert.equal(f.run('dictDraft.edited'), false);

  snapshot.dictionary.exact.Stored = [' updated '];
  f.push('daemonState', snapshot);
  id = `dict-values-${f.run('dictDraft.exact[0].id')}`;
  assert.equal(f.nodes.get(id).value, ' updated ');
  assertSaveChanged(f, 'dictionary', false);
});

for (const key of ['Stored', ' Stored ']) {
  test(`saved exact whitespace edits and reverts compare cleanly (${JSON.stringify(key)})`, () => {
    const f = fixture();
    const snapshot = structuredClone(f.initial);
    snapshot.dictionary.exact = { [key]: [' stored ', '\tcandidate\t'] };
    f.push('daemonState', snapshot);
    f.nav('dictionary');
    const row = f.run('dictDraft.exact[0].id');
    assertSaveChanged(f, 'dictionary', false);
    f.edit(`dict-values-${row}`, 'changed');
    assertSaveChanged(f, 'dictionary', true);
    f.edit(`dict-values-${row}`, ' stored, \tcandidate\t\n');
    assertSaveChanged(f, 'dictionary', false);
    f.edit(`dict-key-${row}`, 'Changed');
    assertSaveChanged(f, 'dictionary', true);
    f.edit(`dict-key-${row}`, 'Stored');
    assertSaveChanged(f, 'dictionary', false);
    assert.deepEqual(JSON.parse(f.run('JSON.stringify(buildDictionaryPayload().payload.exact)')),
      { Stored: ['stored', 'candidate'] });
  });
}

test('saved regex whitespace stays significant when comparing actual persisted and outgoing patterns', () => {
  const f = fixture({ initialRoute: 'dictionary' });
  f.run(`state.dictionary.regex = { Spaced: [' foo '] }; dictDraft = null; render();`);
  const id = `dict-values-${f.run('dictDraft.regex[0].id')}`;
  const payload = () => JSON.parse(f.run('JSON.stringify(buildDictionaryPayload().payload)'));
  assert.equal(f.nodes.get(id).value, ' foo ');
  assert.deepEqual(payload().regex, { Spaced: ['foo'] });
  assertSaveChanged(f, 'dictionary', true); // Saving already normalizes this non-round-tripping value.
  f.edit(id, 'foo');
  assertSaveChanged(f, 'dictionary', true);
  assert.equal(new RegExp(' foo ').test('foo'), false);
  assert.equal(new RegExp(payload().regex.Spaced[0]).test('foo'), true);
  f.edit(id, ' foo ');
  assert.deepEqual(payload().regex, { Spaced: ['foo'] });
  assertSaveChanged(f, 'dictionary', true); // Restoring raw padding still submits the normalized pattern.
  const snapshot = structuredClone(f.initial);
  snapshot.dictionary.regex = { Spaced: ['foo'] };
  f.push('daemonState', snapshot);
  assert.equal(f.nodes.get(id).value, ' foo ');
  assertSaveChanged(f, 'dictionary', false); // The payload now equals the actual saved pattern.
});

test('regex and exact edits revert cleanly when the outgoing payload matches persisted values', () => {
  const f = fixture({ initialRoute: 'dictionary' });
  f.run(`state.dictionary.regex = { Pattern: ['foo'] }; dictDraft = null; render();`);
  const regex = `dict-values-${f.run('dictDraft.regex[0].id')}`;
  const exact = `dict-values-${f.run('dictDraft.exact[0].id')}`;
  f.edit(regex, 'f oo');
  assertSaveChanged(f, 'dictionary', true);
  f.edit(regex, '\tfoo \n\n');
  assertSaveChanged(f, 'dictionary', false);
  f.edit(exact, 'other, candidate');
  assertSaveChanged(f, 'dictionary', true);
  f.edit(exact, ' stored, \n\n');
  assertSaveChanged(f, 'dictionary', false);
  assert.deepEqual(JSON.parse(f.run('JSON.stringify(buildDictionaryPayload().payload)')),
    { exact: { Stored: ['stored'] }, regex: { Pattern: ['foo'] } });
});

for (const stage of ['save', 'refresh']) {
  for (const field of ['key', 'values']) {
    test(`dictionary ${field} edit during pending ${stage} keeps its focused node and selection on completion`, async () => {
      const f = fixture({ initialRoute: 'dictionary' });
      editScreen(f, 'dictionary', 'submitted candidate');
      const saved = savedScreenSnapshot(f, 'dictionary', 'submitted candidate');
      f.run(`bridge = action => {
        if (action === 'saveDictionary') return new Promise(resolve => {
          finishSave = () => resolve({ saved: true });
        });
        return new Promise(resolve => { finishRefresh = () => resolve(${JSON.stringify(saved)}); });
      };`);
      const saving = f.nodes.get('save-dictionary').listeners.click();
      if (stage === 'refresh') {
        f.run('finishSave()');
        await new Promise(resolve => setImmediate(resolve));
      }
      const id = `dict-${field}-${f.run('dictDraft.exact[0].id')}`;
      const input = f.nodes.get(id);
      input.focus();
      f.edit(id, 'newer unsaved content');
      input.setSelectionRange(2, 8);
      const assertEditor = () => {
        assert.equal(f.nodes.get(id), input);
        assert.equal(f.run(`document.activeElement === document.getElementById('${id}')`), true);
        assert.equal(input.value, 'newer unsaved content');
        assert.equal(input.selectionStart, 2);
        assert.equal(input.selectionEnd, 8);
      };
      assertEditor();
      if (stage === 'save') {
        f.run('finishSave()');
        await new Promise(resolve => setImmediate(resolve));
        assertEditor();
      }
      f.run('finishRefresh()');
      await saving;
      assertEditor();
      assertSaveChanged(f, 'dictionary', true);
      assert.equal(f.nodes.get('dictionary-error').textContent, 'dictionary_saved_message');
    });
  }
}

test('validation response for an older dictionary save patches feedback without replacing a newer focused edit', async () => {
  const f = fixture({ initialRoute: 'dictionary' });
  editScreen(f, 'dictionary', 'submitted candidate');
  f.run(`bridge = () => new Promise(resolve => { finishSave = () => resolve({
    saved: false, errors: [{ section: 'exact', key: 'Stored', message: 'invalid old edit' }],
  }); });`);
  const saving = f.nodes.get('save-dictionary').listeners.click();
  const id = `dict-values-${f.run('dictDraft.exact[0].id')}`;
  const input = f.nodes.get(id);
  input.focus();
  f.edit(id, 'newer unsaved content');
  input.setSelectionRange(3, 9);
  f.run('finishSave()');
  await saving;
  assert.equal(f.nodes.get(id), input);
  assert.equal(f.run(`document.activeElement === document.getElementById('${id}')`), true);
  assert.equal(input.value, 'newer unsaved content');
  assert.equal(input.selectionStart, 3);
  assert.equal(input.selectionEnd, 9);
  assert.equal(f.run('dictDraft.exact[0].error'), null);
  assertSaveChanged(f, 'dictionary', true);
  assert.equal(f.nodes.get('save-dictionary').textContent, 'save_retry_button');
});

test('all three dirty headers stay independent when dictionary save completes on another screen', async () => {
  const f = fixture({ glossary: ['Stored'] });
  editScreen(f, 'settings', 'http://unsaved');
  f.nav('dictionary');
  editScreen(f, 'dictionary', 'dictionary-edit');
  f.nav('glossary');
  editScreen(f, 'glossary', 'glossary-edit');
  for (const scope of ['settings', 'dictionary', 'glossary']) {
    f.nav(scope);
    assertSaveChanged(f, scope, true);
  }
  f.nav('dictionary');
  const saved = savedScreenSnapshot(f, 'dictionary', 'dictionary-edit');
  f.run(`bridge = (action) => {
    if (action === 'saveDictionary') return new Promise(resolve => { finishSave = () => resolve({ saved: true }); });
    return Promise.resolve(${JSON.stringify(saved)});
  };`);
  const saving = f.nodes.get('save-dictionary').listeners.click();
  f.nav('settings');
  const snapshot = structuredClone(f.initial);
  snapshot.settings.llm_correction.provider = 'external-provider';
  f.push('daemonState', snapshot);
  assertSaveChanged(f, 'settings', true);
  f.run('finishSave()');
  await saving;
  assert.equal(f.run('state.settings.llm_correction.provider'), 'external-provider');
  assertSaveChanged(f, 'settings', true);
  f.nav('glossary');
  assert.equal(screenValue(f, 'glossary'), 'glossary-edit');
  assertSaveChanged(f, 'glossary', true);
  f.nav('dictionary');
  assertSaveChanged(f, 'dictionary', false);
});

test('shared save styling stays top-right and sticky with finite glow and a reduced-motion override', () => {
  const css = fs.readFileSync(path.join(__dirname, '../src/ptarmigan_flow/webui/app.css'), 'utf8');
  const header = css.match(/\.screen-save-header\s*\{([^}]+)\}/)[1];
  assert.match(header, /position: sticky/);
  assert.match(header, /top: var\(--space-3\)/);
  assert.match(header, /grid-template-columns: minmax\(0, 1fr\) auto/);
  assert.match(css.match(/\.screen-save-actions\s*\{([^}]+)\}/)[1], /justify-content: flex-end/);
  const changed = css.match(/\.save-button\.has-changes\s*\{([^}]+)\}/)[1];
  assert.match(changed, /animation: save-changed-glow 2\.8s ease-in-out 2/);
  assert.doesNotMatch(changed, /infinite|transform|bounce/);
  assert.match(css, /\.save-button\.has-changes::before\s*\{[^}]*content: ""/);
  assert.match(css, /\.save-button\.has-changes:focus-visible\s*\{[^}]*animation: none;[^}]*box-shadow: var\(--accent-ring\)/);
  const reduced = css.match(/@media \(prefers-reduced-motion: reduce\)\s*\{([\s\S]*?)\n\}/)[1];
  assert.match(reduced, /\*::before/);
  assert.match(reduced, /animation: none !important/);
  assert.match(reduced, /transition: none !important/);
});

test('unchanged raw input events acknowledge the save and allow later external updates', async () => {
  const f = fixture({ initialRoute: 'glossary' });
  f.edit('llm_glossary', ' Added \n\n');
  f.run(`bridge = (action, payload) => {
    if (action === 'saveSettings') {
      const settings = JSON.parse(JSON.stringify(payload));
      return new Promise(resolve => { finishSave = () => resolve({ saved: true, settings }); });
    }
    return Promise.resolve({ ...state, settings: {
      ...state.settings, llm_correction: { ...state.settings.llm_correction, glossary: ['Added'] },
    } });
  };`);
  const saving = f.run('saveGlossary()');
  f.edit('llm_glossary', ' Added \n\n'); // Repeated input/change events without an actual text edit.
  f.run('finishSave()');
  await saving;
  assert.equal(f.run('Object.keys(settingsDraft).length'), 0);
  assert.equal(f.run('glossaryDraftText'), null);
  const snapshot = structuredClone(f.initial);
  snapshot.settings.llm_correction.glossary = ['External'];
  f.push('daemonState', snapshot);
  assert.equal(f.nodes.get('llm_glossary').value, 'External');
});

test('three navigation buttons open independent screens with glossary absent from general settings', () => {
  const f = fixture({ glossary: ['Loaded'] });
  f.run(`state.strings = { route_settings: '設定', route_dictionary: '辞書', route_glossary: 'よく使う言葉', dictionary_editor_title: '変換辞書' }; render();`);
  for (const [route, label] of [['settings', '設定'], ['dictionary', '辞書'], ['glossary', 'よく使う言葉']]) {
    assert.equal(f.nodes.get(`nav-${route}`).textContent, label);
  }
  assert.equal([...f.nodes.values()].filter(n => n.dataset.route).length, 3);
  assert.ok(!f.nodes.has('llm_glossary'));
  assert.ok(!Object.hasOwn(f.payload().llm_correction, 'glossary'));
  f.nav('dictionary');
  assert.match(f.html(), /<h2>変換辞書<\/h2>/);
  assert.match(f.html(), /Stored/);
  assert.match(f.html(), /data-add-dictionary="exact"/);
  f.run('dictDraft.exact[0].key = "Unsaved dictionary"; dictDraft.dirty = true');
  f.nav('glossary');
  assert.match(f.html(), /<h2>よく使う言葉<\/h2>/);
  assert.match(f.html(), /class="active" data-route="glossary"/);
  assert.equal(f.nodes.get('llm_glossary').value, 'Loaded');
  assert.ok(!f.nodes.has('llm_provider'));
  f.nav('settings');
  assert.ok(!f.nodes.has('llm_glossary'));
  f.nav('dictionary');
  assert.match(f.html(), /Unsaved dictionary/);
});

test('many glossary terms use a bounded scrollable editor and survive navigation without truncation', () => {
  const glossary = Array.from({ length: 2000 }, (_, index) => `Term ${index}`);
  const f = fixture({ glossary, initialRoute: 'glossary' });
  assert.equal(f.nodes.get('llm_glossary').value, glossary.join('\n'));
  assert.match(f.html(), /class="glossary-editor" rows="16"/);
  const css = fs.readFileSync(path.join(__dirname, '../src/ptarmigan_flow/webui/app.css'), 'utf8');
  const style = css.match(/\.glossary-editor\s*\{([^}]+)\}/)[1];
  assert.match(style, /height: clamp\(/);
  assert.match(style, /max-height: 65vh/);
  assert.match(style, /overflow-y: auto/);
  f.edit('llm_glossary', `\n${glossary.join('\n')}\n\n`);
  f.nav('settings');
  f.nav('glossary');
  assert.equal(f.nodes.get('llm_glossary').value, `\n${glossary.join('\n')}\n\n`);
  assert.deepEqual(f.glossaryPayload(), { llm_correction: { glossary } });
});

test('settings and glossary drafts remain independent across switches and native pushes', () => {
  const f = fixture({ glossary: ['Stored'] });
  f.edit('llm_provider', ' unsaved provider ');
  f.edit('language', 'ja');
  f.nav('glossary');
  const text = '\n Preferred \n\n';
  f.edit('llm_glossary', text);
  assert.deepEqual(f.glossaryPayload(), { llm_correction: { glossary: ['Preferred'] } });
  const snapshot = structuredClone(f.initial);
  snapshot.settings.llm_correction.provider = 'external-provider';
  snapshot.settings.llm_correction.glossary = ['External'];
  for (const event of ['daemonState', 'permissionsChanged']) {
    f.push(event, snapshot);
    assert.equal(f.nodes.get('llm_glossary').value, text);
    f.nav('settings');
    assert.equal(f.nodes.get('llm_provider').value, ' unsaved provider ');
    assert.equal(f.payload().language, 'ja');
    assert.ok(!Object.hasOwn(f.payload().llm_correction, 'glossary'));
    f.push(event, snapshot);
    f.nav('dictionary');
    f.nav('glossary');
    assert.equal(f.nodes.get('llm_glossary').value, text);
  }
});

test('glossary-only save and general settings omission preserve the other screen and persisted list', async () => {
  const f = fixture({ glossary: ['Stored'] });
  f.edit('llm_provider', ' unsaved provider ');
  f.edit('language', 'ja');
  f.run(`
    let persisted = JSON.parse(JSON.stringify(state.settings));
    let saves = [];
    bridge = async (action, payload) => {
      if (action === 'saveSettings') {
        saves.push(JSON.parse(JSON.stringify(payload)));
        persisted = { ...persisted, ...payload, llm_correction: {
          ...persisted.llm_correction, ...payload.llm_correction,
        } };
        return { saved: true, settings: JSON.parse(JSON.stringify(persisted)) };
      }
      return { ...state, settings: JSON.parse(JSON.stringify(persisted)) };
    };
  `);
  f.nav('glossary');
  f.edit('llm_glossary', 'Saved glossary');
  await f.nodes.get('save-glossary').listeners.click();
  assert.deepEqual(JSON.parse(f.run('JSON.stringify(saves[0])')),
    { llm_correction: { glossary: ['Saved glossary'] } });
  assert.equal(f.run('persisted.llm_correction.provider'), 'ollama');
  assert.equal(f.run('persisted.language'), 'en');
  f.edit('llm_glossary', 'Next unsaved glossary\n\n');
  f.nav('settings');
  assert.equal(f.payload().llm_correction.provider, ' unsaved provider ');
  assert.equal(f.payload().language, 'ja');
  await f.nodes.get('save-settings').listeners.click();
  assert.ok(!Object.hasOwn(JSON.parse(f.run('JSON.stringify(saves[1].llm_correction)')), 'glossary'));
  assert.equal(f.run('persisted.llm_correction.glossary[0]'), 'Saved glossary');
  f.nav('glossary');
  assert.equal(f.nodes.get('llm_glossary').value, 'Next unsaved glossary\n\n');
});

for (const first of ['settings', 'glossary']) {
  test(`out-of-order ${first} ACK merges only its screen while both drafts stay editable`, async () => {
    const f = fixture({ glossary: ['Stored'] });
    f.run(`
      let saveRequests = [];
      let refreshRequests = [];
      bridge = (action, payload) => new Promise((resolve, reject) => {
        (action === 'saveSettings' ? saveRequests : refreshRequests).push({ payload, resolve, reject });
      });
    `);
    f.edit('llm_provider', 'Sent provider');
    const settingsSaving = f.run('saveSettings()');
    f.nav('glossary');
    f.edit('llm_glossary', 'Sent glossary');
    const glossarySaving = f.run('saveGlossary()');
    f.run(`
      const settingsAck = { ...state.settings, llm_correction: {
        ...state.settings.llm_correction, provider: 'Sent provider', glossary: ['Stored'],
      } };
      const glossaryAck = { ...state.settings, llm_correction: {
        ...state.settings.llm_correction, provider: 'ollama', glossary: ['Sent glossary'],
      } };
    `);
    // Edit both scopes after submission; even matching native pushes do not acknowledge edits.
    f.edit('llm_glossary', 'New glossary\n\n');
    f.nav('settings');
    f.edit('llm_provider', 'New provider');
    const snapshot = structuredClone(f.initial);
    snapshot.settings.llm_correction.provider = 'New provider';
    snapshot.settings.llm_correction.glossary = ['New glossary'];
    f.push('daemonState', snapshot);
    snapshot.settings.llm_correction.provider = 'ollama';
    snapshot.settings.llm_correction.glossary = ['Stored'];
    f.push('permissionsChanged', snapshot);
    // Writes are serialized; their subsequent state refreshes may still finish
    // in either order and carry snapshots from before the other write.
    f.run('saveRequests[0].resolve({ saved: true, settings: settingsAck })');
    await new Promise(resolve => setImmediate(resolve));
    f.run('saveRequests[1].resolve({ saved: true, settings: glossaryAck })');
    await new Promise(resolve => setImmediate(resolve));
    for (const scope of [first, first === 'settings' ? 'glossary' : 'settings']) {
      const index = scope === 'settings' ? 0 : 1;
      f.run(`refreshRequests[${index}].resolve({ ...state, settings: ${scope}Ack })`);
      await (scope === 'settings' ? settingsSaving : glossarySaving);
    }
    // Stale full snapshots in the second ACK cannot roll back the first save.
    assert.equal(f.run('state.settings.llm_correction.provider'), 'Sent provider');
    assert.equal(f.run('state.settings.llm_correction.glossary[0]'), 'Sent glossary');
    assert.equal(f.payload().llm_correction.provider, 'New provider');
    f.nav('glossary');
    assert.equal(f.nodes.get('llm_glossary').value, 'New glossary\n\n');
    assert.equal(f.run('settingsSaving || glossarySaving'), false);
  });
}

for (const scope of ['settings', 'glossary']) {
  test(`${scope} failure and retry preserve the other screen's draft and independent feedback`, async () => {
    const f = fixture({ glossary: ['Stored'] });
    f.edit('llm_provider', ' Unsaved provider ');
    f.nav('glossary');
    f.edit('llm_glossary', '\n Unsaved glossary \n\n');
    f.run(`
      let fail = true;
      let persisted = JSON.parse(JSON.stringify(state.settings));
      bridge = async (action, payload) => {
        if (action === 'saveSettings') {
          if (fail) throw Error('failed to write');
          persisted = { ...persisted, ...payload, llm_correction: {
            ...persisted.llm_correction, ...payload.llm_correction,
          } };
          return { saved: true, settings: persisted };
        }
        return { ...state, settings: persisted };
      };
    `);
    f.nav(scope);
    await f.nodes.get(`save-${scope}`).listeners.click();
    assert.equal(f.nodes.get(`save-${scope}`).textContent, 'save_retry_button');
    const other = scope === 'settings' ? 'glossary' : 'settings';
    f.nav(other);
    assert.equal(f.nodes.get(`save-${other}`).disabled, false);
    assert.equal(f.nodes.get(`${other}-error`).textContent, '');
    f.push('permissionsChanged', structuredClone(f.initial));
    f.nav(scope);
    assert.equal(f.nodes.get(`save-${scope}`).textContent, 'save_retry_button');
    f.run('fail = false');
    await f.nodes.get(`save-${scope}`).listeners.click();
    f.nav(other);
    if (other === 'glossary') {
      assert.equal(f.nodes.get('llm_glossary').value, '\n Unsaved glossary \n\n');
      assert.equal(f.run('state.settings.llm_correction.glossary[0]'), 'Stored');
    } else {
      assert.equal(f.nodes.get('llm_provider').value, ' Unsaved provider ');
      assert.equal(f.run('state.settings.llm_correction.provider'), 'ollama');
    }
  });
}

test('glossary sends one scoped write despite an unrelated model and external provider changes', async () => {
  const f = fixture({ glossary: ['Stored'] });
  f.edit('llm_provider', 'Unsaved provider');
  f.edit('language', 'ja');
  f.nav('glossary');
  f.edit('llm_glossary', ' Preferred \n\n');
  f.run(`
    state.settings.model = 'vllm:mistralai/Voxtral-Mini-4B-Realtime-2602';
    let requests = [];
    bridge = (action, payload) => {
      requests.push({ action, payload });
      if (action === 'saveSettings') {
        const settings = { ...state.settings, llm_correction: {
          ...state.settings.llm_correction, glossary: ['Preferred'],
        } };
        return new Promise(resolve => { finishSave = () => resolve({ saved: true, settings }); });
      }
      return Promise.resolve(state);
    };
  `);
  const saving = f.nodes.get('save-glossary').listeners.click();
  assert.deepEqual(JSON.parse(f.run('JSON.stringify(requests)')), [
    { action: 'saveSettings', payload: { llm_correction: { glossary: ['Preferred'] } } },
  ]);
  const snapshot = structuredClone(f.initial);
  snapshot.settings.model = 'vllm:mistralai/Voxtral-Mini-4B-Realtime-2602';
  snapshot.settings.llm_correction.provider = 'externally-updated-provider';
  f.push('daemonState', snapshot);
  f.run('finishSave()');
  await saving;
  assert.deepEqual(JSON.parse(f.run('JSON.stringify(requests.map(r => r.action))')),
    ['saveSettings', 'getState']);
  assert.deepEqual(JSON.parse(f.run('JSON.stringify(requests[0].payload)')),
    { llm_correction: { glossary: ['Preferred'] } });
  assert.equal(f.run('state.settings.llm_correction.provider'), 'externally-updated-provider');
  assert.equal(f.run('state.settings.model'), snapshot.settings.model);
  f.nav('settings');
  assert.equal(f.payload().language, 'ja');
  assert.equal(f.nodes.get('llm_provider').value, 'Unsaved provider');
  assert.equal(f.run('glossaryMessage.kind'), 'success');
});

for (const errors of [['model'], ['model', 'language', 'hotkey', 'output_mode'], ['llm_correction.glossary']]) {
  test(`glossary errors ${JSON.stringify(errors)} never retry with unrelated settings`, async () => {
    const f = fixture({ glossary: ['Stored'] });
    f.edit('llm_provider', 'Unsaved provider');
    f.nav('glossary');
    f.edit('llm_glossary', ' \n\n');
    f.run(`
      let requests = [];
      let fail = true;
      bridge = async (action, payload) => {
        requests.push({ action, payload });
        if (action === 'saveSettings') {
          if (fail) return { saved: false, errors: ${JSON.stringify(errors)} };
          return { saved: true, settings: { llm_correction: { glossary: [] } } };
        }
        return state;
      };
    `);
    await f.nodes.get('save-glossary').listeners.click();
    const write = { action: 'saveSettings', payload: { llm_correction: { glossary: [] } } };
    assert.deepEqual(JSON.parse(f.run('JSON.stringify(requests)')), [write]);
    assert.equal(f.nodes.get('save-glossary').textContent, 'save_retry_button');
    assert.equal(f.nodes.get('llm_glossary').value, ' \n\n');
    assert.equal(f.run('settingsWriteBusy'), false);
    f.nav('settings');
    assert.equal(f.nodes.get('llm_provider').value, 'Unsaved provider');
    assert.equal(f.nodes.get('settings-error').textContent, '');
    f.nav('glossary');
    f.run('fail = false');
    await f.nodes.get('save-glossary').listeners.click(); // Only the user retries.
    assert.deepEqual(JSON.parse(f.run('JSON.stringify(requests)')),
      [write, write, { action: 'getState' }]);
    assert.equal(f.nodes.get('llm_glossary').value, '');
    assert.equal(f.run('glossaryMessage.kind'), 'success');
  });
}

test('server-normalized NEL glossary is adopted cleanly and later native updates are visible', async () => {
  const f = fixture({ glossary: ['Stored'] });
  f.edit('llm_provider', 'Unsaved provider');
  f.nav('glossary');
  const raw = '\u0085Alpha\u0085'; // Python str.strip removes NEL; JavaScript trim does not.
  f.edit('llm_glossary', raw);
  assert.deepEqual(f.glossaryPayload(), { llm_correction: { glossary: [raw] } });
  f.run(`
    let requests = [];
    bridge = async (action, payload) => {
      requests.push({ action, payload });
      if (action === 'saveSettings') {
        return { saved: true, settings: { ...state.settings,
          llm_correction: { ...state.settings.llm_correction, glossary: ['Alpha'] },
        } };
      }
      return state; // Even an older refresh cannot replace the normalized save ACK.
    };
  `);
  await f.nodes.get('save-glossary').listeners.click();
  assert.deepEqual(JSON.parse(f.run('JSON.stringify(requests)')), [
    { action: 'saveSettings', payload: { llm_correction: { glossary: [raw] } } },
    { action: 'getState' },
  ]);
  assert.equal(f.nodes.get('llm_glossary').value, 'Alpha');
  assert.equal(f.run('glossaryDraftText'), null);
  assert.equal(f.run('glossaryMessage.kind'), 'success');
  f.nav('settings');
  assert.equal(f.nodes.get('llm_provider').value, 'Unsaved provider');
  f.nav('glossary');
  const snapshot = structuredClone(f.initial);
  snapshot.settings.llm_correction.glossary = ['External'];
  f.push('daemonState', snapshot);
  assert.equal(f.nodes.get('llm_glossary').value, 'External');
  snapshot.settings.llm_correction.glossary = ['Another external update'];
  f.push('permissionsChanged', snapshot);
  assert.equal(f.nodes.get('llm_glossary').value, 'Another external update');
});

for (const stage of ['save', 'refresh']) {
  for (const returnToSubmittedText of [false, true]) {
    test(`normalized NEL ACK preserves newer ${stage} edits${returnToSubmittedText ? ' even after returning to submitted text' : ''}`, async () => {
      const f = fixture();
      f.edit('llm_provider', 'Unsaved provider');
      f.nav('glossary');
      const submitted = '\u0085Alpha\u0085';
      f.edit('llm_glossary', submitted);
      f.run(`
        bridge = (action, payload) => {
          const settings = { ...state.settings, llm_correction: {
            ...state.settings.llm_correction, glossary: ['Alpha'],
          } };
          if (action === 'saveSettings') {
            return new Promise(resolve => { finishSave = () => resolve({ saved: true, settings }); });
          }
          return new Promise(resolve => { finishRefresh = () => resolve({ ...state, settings }); });
        };
      `);
      const saving = f.run('saveGlossary()');
      if (stage === 'refresh') {
        f.run('finishSave()');
        await new Promise(resolve => setImmediate(resolve));
      }
      f.edit('llm_glossary', '\n New unsaved term \n\n');
      if (returnToSubmittedText) f.edit('llm_glossary', submitted);
      const newer = f.nodes.get('llm_glossary').value;
      f.nav('settings');
      assert.equal(f.nodes.get('llm_provider').value, 'Unsaved provider');
      f.nav('glossary');
      if (stage === 'save') {
        f.run('finishSave()');
        await new Promise(resolve => setImmediate(resolve));
      }
      f.run('finishRefresh()');
      await saving;
      assert.equal(f.run('state.settings.llm_correction.glossary[0]'), 'Alpha');
      assert.equal(f.nodes.get('llm_glossary').value, newer);
      assert.equal(f.run('glossaryDraftText'), newer);
      const snapshot = structuredClone(f.initial);
      snapshot.settings.llm_correction.glossary = ['External'];
      f.push('daemonState', snapshot);
      assert.equal(f.nodes.get('llm_glossary').value, newer);
      f.nav('settings');
      assert.equal(f.nodes.get('llm_provider').value, 'Unsaved provider');
    });
  }
}
