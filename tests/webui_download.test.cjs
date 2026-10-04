// Exercise the real frontend lifecycle without a native bridge or DOM dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../src/ptarmigan_flow/webui/app.js'), 'utf8');
function setup() {
  const requests = [];
  const context = vm.createContext({
    window: { webkit: { messageHandlers: { bridge: { postMessage: message => requests.push(message) } } } },
    document: { getElementById: () => ({ querySelector: () => null }) },
  });
  vm.runInContext(source.replace('boot().catch(showError);', ''), context);
  const run = code => vm.runInContext(code, context);
  run(`
    state = {
      settings: { model: 'a', hotkey: 'left_cmd' },
      models: [{token:'a'}, {token:'b'}, {token:'c'}],
      strings: {
        settings_model_busy_message: 'Another download is in progress',
        settings_model_download_error_message: 'Download failed: {error}',
        settings_model_download_done_message: 'Download complete',
        settings_model_preparing_message: 'Preparing download',
        settings_model_download_button: 'Download',
        settings_model_retry_button: 'Retry',
        settings_model_refresh_failed_message: 'Could not refresh model availability',
        download_in_progress_message: 'Downloading model... {percent}',
      },
    };
    route = 'settings';
    let renders = 0;
    let patches = [];
    render = () => { renders++; };
    updateModelCardInPlace = token => { patches.push(token); };
  `);
  const click = token => run(`handleDownloadModelClick({ stopPropagation() {}, currentTarget: { dataset: { downloadModel: '${token}' } } })`);
  const reply = (index, result, error) => {
    context.window.app.dispatch({ id: requests[index].id, ok: !error, result, error });
  };
  const push = (type, model = 'a', extra = {}) => {
    context.window.app.dispatch({ event: 'downloadProgress', payload: { type, model, ...extra } });
  };
  return { run, click, reply, push, requests, context };
}
const flush = () => new Promise(resolve => setImmediate(resolve));

test('locks globally before awaiting start, progress is truthful and accessible', async () => {
  const h = setup();
  const started = h.click('a');
  await h.click('b');
  assert.equal(h.requests.length, 1);
  assert.equal(h.run('activeDownloadToken'), 'a');
  assert.match(h.run(`renderModelAction({token:'b'}, 'idle')`), /disabled/);
  assert.match(h.run(`renderModelAction({token:'c'}, 'error')`), /disabled/);
  const preparing = h.run(`renderModelCard({token:'a'}, 'a')`);
  assert.match(preparing, /role="status"/);
  assert.match(preparing, /role="progressbar"/);
  assert.doesNotMatch(preparing, /aria-valuenow/);
  h.reply(0, { started: true });
  await started;
  h.push('progress', 'a', { fraction: 0.42 });
  const progress = h.run(`renderModelCard({token:'a'}, 'a')`);
  assert.match(progress, /aria-valuenow="42"/);
  assert.match(progress, /Downloading model\.\.\. 42%/);
  h.push('progress', 'a', { fraction: null });
  assert.equal(h.run(`downloadStates.get('a').status`), 'preparing');
  assert.equal(h.run('renders'), 0);
});

test('done requests getState, patches cache only and preserves selection/settings', async () => {
  const h = setup();
  const started = h.click('a');
  h.reply(0, { started: true });
  await started;
  h.run(`state.settings.model = 'b'; state.settings.hotkey = 'draft';`);
  h.push('done');
  assert.equal(h.run('activeDownloadToken'), null);
  assert.equal(h.run(`downloadStates.get('a').status`), 'done');
  assert.equal(h.requests[1].action, 'getState');
  assert.match(h.run(`renderModelCard({token:'a'}, 'b')`), /Download complete/);
  h.reply(1, { settings: { model: 'a', hotkey: 'saved' }, models: [{ token: 'a', downloaded: true }, { token: 'b' }] });
  await flush();
  assert.equal(h.run(`state.models[0].downloaded`), true);
  assert.equal(h.run('state.settings.model'), 'b');
  assert.equal(h.run('state.settings.hotkey'), 'draft');
  assert.equal(h.run('renders'), 0);
  h.run(`renderModelCard(state.models[0], 'b')`);
  assert.equal(h.run(`downloadStates.has('a')`), false);
});

test('completion daemonState push and model selection never full-render the form', () => {
  const h = setup();
  h.context.window.app.dispatch({ event: 'daemonState', payload: { settings: { model: 'a' }, models: [{ token: 'a', downloaded: true }], daemon_running: true } });
  h.run(`
    const handlers = {};
    const card = { dataset: {selectModel:'b'}, addEventListener: (name, fn) => { handlers[name] = fn; }, querySelector: () => null };
    bindModelCard(card);
    handlers.click();
    handlers.keydown({target:card, currentTarget:card, key:'Enter', preventDefault() {}});
  `);
  assert.equal(h.run('state.settings.model'), 'b');
  assert.equal(h.run('state.daemon_running'), true);
  assert.equal(h.run('renders'), 0);
});

test('busy response is localized contention, not ordinary failure; retry is available', async () => {
  const h = setup();
  const started = h.click('a');
  h.reply(0, { started: false, errors: ['busy'] });
  await started;
  assert.equal(h.run('activeDownloadToken'), null);
  assert.equal(h.run(`downloadStates.get('a').status`), 'busy');
  const html = h.run(`renderModelCard({token:'a'}, 'b')`);
  assert.match(html, /Another download is in progress/);
  assert.doesNotMatch(html, /Download failed|disabled/);
  const retry = h.click('a');
  h.reply(1, { started: true });
  await retry;
  assert.equal(h.run('activeDownloadToken'), 'a');
});

test('failed start, rejected bridge, and download error release the lock', async () => {
  for (const outcome of ['invalid', 'rejected', 'error']) {
    const h = setup();
    const started = h.click('a');
    h.reply(0, outcome === 'invalid' ? { started: false, errors: ['model'] } : { started: true }, outcome === 'rejected' ? 'Connection lost' : null);
    await started;
    if (outcome === 'error') h.push('error', 'a', { message: 'Connection lost' });
    assert.equal(h.run('activeDownloadToken'), null);
    assert.equal(h.run(`downloadStates.get('a').status`), 'error');
    assert.match(h.run(`renderModelCard({token:'a'}, 'b')`), /Download failed/);
    assert.doesNotMatch(h.run(`renderModelAction({token:'b'}, 'idle')`), /disabled/);
  }
});

test('already downloaded refreshes cache without erasing unsaved selection', async () => {
  const h = setup();
  const started = h.click('a');
  h.run(`state.settings.model = 'b'`);
  h.reply(0, { started: false, already_downloaded: true });
  await flush();
  assert.equal(h.run('activeDownloadToken'), null);
  assert.equal(h.requests[1].action, 'getState');
  h.reply(1, { models: [{ token: 'a', downloaded: true }] });
  await started;
  assert.equal(h.run('state.settings.model'), 'b');
  assert.equal(h.run('renders'), 0);
});

test('refresh rejection keeps truthful completion, reports sync failure and unlocks', async () => {
  const h = setup();
  h.push('done');
  h.reply(0, null, 'getState unavailable');
  await flush();
  const html = h.run(`renderModelCard({token:'a'}, 'a')`);
  assert.match(html, /Download complete/);
  assert.match(html, /Could not refresh model availability/);
  assert.equal(h.run('activeDownloadToken'), null);
});

test('progress received before start response and completion away from settings remain coherent', async () => {
  const h = setup();
  const started = h.click('a');
  h.push('progress', 'a', { fraction: 0.42 });
  h.reply(0, { started: true });
  await started;
  assert.equal(h.run(`downloadStates.get('a').fraction`), 0.42);
  h.run(`route = 'dictionary'`);
  h.push('done');
  h.reply(1, { models: [{ token: 'a', downloaded: true }] });
  await flush();
  assert.equal(h.run('state.models[0].downloaded'), true);
  assert.equal(h.run('renders'), 0);
});
