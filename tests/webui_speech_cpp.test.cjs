const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../src/ptarmigan_flow/webui/app.js'), 'utf8');

function setup() {
  const requests = [];
  const status = { textContent: '' };
  const progress = { innerHTML: '' };
  let button = null;
  const action = {
    html: '',
    get innerHTML() { return this.html; },
    set innerHTML(value) {
      this.html = value;
      button = value.includes('data-install-speech-cpp') ? {
        listeners: {}, addEventListener(name, fn) { this.listeners[name] = fn; },
      } : null;
    },
    querySelector() { return button; },
    contains(value) { return Boolean(button && value === button); },
  };
  let focusCount = 0;
  const container = {
    querySelector(selector) {
      return { '[data-speech-cpp-status]': status, '[data-speech-cpp-progress]': progress,
        '[data-speech-cpp-action]': action }[selector];
    },
    closest() { return { focus() { focusCount++; } }; },
  };
  const field = { id: 'llm_base_url', value: '', listeners: {} };
  const app = {
    querySelector() { return null; },
    querySelectorAll(selector) {
      if (selector === '[data-speech-cpp]') return [container];
      if (selector === '[data-install-speech-cpp]') return button ? [button] : [];
      if (selector === 'select[id], input[id]') return [field];
      return [];
    },
  };
  const document = { activeElement: null, getElementById: id => id === 'app' ? app : null };
  const window = { webkit: { messageHandlers: { bridge: { postMessage: msg => requests.push(msg) } } } };
  const context = vm.createContext({ window, document });
  vm.runInContext(source.replace('boot().catch(showError);', ''), context);
  const run = code => vm.runInContext(code, context);
  run(`
    state = {
      setup_required: true, onboarding_step: 'done',
      settings: { model: 'haas:saved', language: 'ja', hotkey: 'right_cmd', output_mode: 'direct_typing',
        llm_correction: { mode: 'never', provider: 'saved', model: 'saved', base_url: 'saved' } },
      models: [{token:'haas:saved', backend:'haas', downloaded:true}],
      strings: {
        speech_cpp_title:'Runtime, separate from model file', speech_cpp_checking:'Checking runtime',
        speech_cpp_missing:'Runtime missing', speech_cpp_old:'Runtime old', speech_cpp_available:'Runtime available',
        speech_cpp_downloading:'Downloading runtime', speech_cpp_verifying:'Verifying runtime',
        speech_cpp_installing:'Installing runtime', speech_cpp_failed:'Runtime failed',
        speech_cpp_unsupported:'Apple Silicon required; Intel uses external runtime',
        speech_cpp_override_invalid:'Override invalid', speech_cpp_override_guidance:'Fix PTARMIGAN_FLOW_SPEECH_CPP_PATH and restart',
        speech_cpp_failure_guidance:'Check connection/disk and retry', speech_cpp_install:'Install runtime', speech_cpp_retry:'Retry runtime',
      },
      speech_cpp: {status:'missing', install_allowed:true, operation_id:1},
    };
    route = 'settings';
    let renders = 0;
    render = () => { renders++; };
    updateModelCardInPlace = () => {};
    acceptSpeechCppStatus(state.speech_cpp);
    updateSpeechCppInPlace();
  `);
  const reply = (index, result, error) => window.app.dispatch({ id: requests[index].id, ok: !error, result, error });
  const push = payload => window.app.dispatch({ event: 'speechCppStatus', payload });
  const click = () => run('handleInstallSpeechCppClick({stopPropagation() {}, currentTarget:{}})');
  return { run, requests, reply, push, click, status, progress, action, field, window, document,
    button: () => button, focusCount: () => focusCount };
}

test('installs without parameters, locks duplicate starts and never selects a model', async () => {
  const h = setup();
  h.run(`settingsDraft.model = 'moonshine:draft'; settingsDraft.llm_base_url = '';`);
  h.document.activeElement = h.button();
  const installing = h.click();
  await h.click();
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].action, 'installSpeechCpp');
  assert.equal(JSON.stringify(h.requests[0].payload), '{}');
  assert.equal(h.focusCount(), 1);
  assert.match(h.status.textContent, /Downloading runtime/);
  assert.match(h.progress.innerHTML, /role="progressbar"/);
  assert.doesNotMatch(h.progress.innerHTML, /aria-valuenow/);
  h.push({status:'installing', phase:'downloading', downloaded_bytes:25, total_bytes:100, operation_id:2});
  h.reply(0, {started:true, speech_cpp:{status:'installing', phase:'downloading', operation_id:2}});
  await installing;
  assert.match(h.progress.innerHTML, /aria-valuenow="25"/);
  assert.equal(h.run('settingsDraft.model'), 'moonshine:draft');
  assert.equal(h.run('state.settings.model'), 'haas:saved');
  assert.equal(h.run('renders'), 0);
  await h.click();
  assert.equal(h.requests.length, 1);
});

test('progress stages are accessible and model availability remains independent', async () => {
  const h = setup();
  h.push({status:'installing', phase:'downloading', downloaded_bytes:42, total_bytes:100, operation_id:2});
  assert.match(h.progress.innerHTML, /aria-valuenow="42"/);
  for (const phase of ['verifying', 'installing']) {
    h.push({status:'installing', phase, operation_id:2});
    assert.doesNotMatch(h.progress.innerHTML, /aria-valuenow/);
    assert.match(h.progress.innerHTML, /indeterminate/);
  }
  const markup = h.run('renderSpeechCpp()');
  assert.match(markup, /role="status" aria-live="polite" aria-atomic="true"/);
  assert.equal(h.run('state.models[0].downloaded'), true);
  h.push({status:'available', version:'0.10.1', source:'managed', install_allowed:false, operation_id:2});
  assert.match(h.status.textContent, /Runtime available \(0.10.1\)/);
  assert.equal(h.progress.innerHTML, '');
  assert.equal(h.action.innerHTML, '');
});

test('failures are escaped/actionable, release the lock and allow retry', async () => {
  const h = setup();
  let installing = h.click();
  h.reply(0, {started:true, speech_cpp:{status:'installing', phase:'downloading', operation_id:2}});
  await installing;
  h.push({status:'failed', message:'<disk full>', install_allowed:true, operation_id:2});
  assert.match(h.status.textContent, /Runtime failed.*Check connection\/disk and retry.*<disk full>/);
  assert.match(h.run('renderSpeechCpp()'), /&lt;disk full&gt;/);
  assert.match(h.action.innerHTML, /Retry runtime/);
  h.run('bindSpeechCppButtons()');
  assert.equal(typeof h.button().listeners.click, 'function');
  installing = h.click();
  h.reply(1, {started:true, speech_cpp:{status:'installing', phase:'downloading', operation_id:3}});
  await installing;
  assert.equal(h.run('speechCppStatus.operation_id'), 3);
  assert.match(h.status.textContent, /Downloading runtime/);
});

test('bridge rejection or failed start allows retry', async () => {
  for (const failure of ['rejected', 'failed']) {
    const h = setup();
    const installing = h.click();
    h.reply(0, {started:false}, failure === 'rejected' ? 'Bridge unavailable' : null);
    await installing;
    assert.equal(h.run('speechCppStatus.status'), 'failed');
    assert.match(h.action.innerHTML, /Retry runtime/);
    assert.equal(h.run('speechCppRequestPending'), false);
  }
});

test('old responses/events/snapshots cannot overwrite a completed or newer operation', async () => {
  const h = setup();
  const installing = h.click();
  h.push({status:'available', version:'0.10.1', install_allowed:false, operation_id:2});
  // Native completion may arrive before the bridge start response.
  assert.match(h.status.textContent, /Runtime available/);
  h.push({status:'installing', phase:'downloading', operation_id:2});
  h.push({status:'failed', message:'stale', install_allowed:true, operation_id:1});
  h.reply(0, {started:true, speech_cpp:{status:'installing', operation_id:2}});
  await installing;
  assert.equal(h.run('speechCppStatus.status'), 'available');
  h.push({status:'installing', phase:'verifying', operation_id:4});
  h.run(`acceptSpeechCppStatus({status:'failed', operation_id:3});`);
  assert.equal(h.run('speechCppStatus.operation_id'), 4);
  assert.match(h.status.textContent, /Verifying runtime/);
});

test('late bridge rejection cannot overwrite native completion', async () => {
  const h = setup();
  const installing = h.click();
  h.push({status:'available', version:'0.10.1', operation_id:2});
  h.reply(0, null, 'late rejection');
  await installing;
  assert.equal(h.run('speechCppStatus.status'), 'available');
});

test('drafts, model selection, dictionary and focused form nodes survive all lifecycle events', async () => {
  const h = setup();
  h.run(`
    Object.assign(settingsDraft, {model:'moonshine:draft', language:'zh', hotkey:'left_cmd', output_mode:'clipboard_paste',
      llm_mode:'ask', llm_provider:'draft', llm_model:'draft', llm_base_url:''});
    dictDraft = {dirty:true, exact:[{key:'draft'}], regex:[]};
  `);
  const draft = h.run('JSON.stringify(settingsDraft)');
  const dictionary = h.run('JSON.stringify(dictDraft)');
  const field = h.field;
  h.document.activeElement = field;
  for (const payload of [
    {status:'installing', phase:'downloading', operation_id:2},
    {status:'failed', message:'network', install_allowed:true, operation_id:2},
    {status:'installing', phase:'verifying', operation_id:3},
    {status:'available', version:'0.10.1', operation_id:3},
  ]) h.push(payload);
  h.run(`window.app.dispatch({event:'daemonState', payload:{...state, speech_cpp:{status:'installing',operation_id:3}, daemon_running:true}});`);
  assert.equal(h.run('JSON.stringify(settingsDraft)'), draft);
  assert.equal(h.run('JSON.stringify(dictDraft)'), dictionary);
  assert.equal(h.document.activeElement, field);
  assert.equal(h.field.value, '');
  assert.equal(h.run('settingsValues().model'), 'moonshine:draft');
  assert.equal(h.run('speechCppStatus.status'), 'available');
  assert.equal(h.run('renders'), 0);
});

test('unsupported and explicit invalid overrides suppress installation with recovery guidance', async () => {
  for (const status of ['unsupported', 'override_invalid']) {
    const h = setup();
    h.push({status, operation_id:2, install_allowed:false});
    await h.click();
    assert.equal(h.requests.length, 0);
    assert.equal(h.action.innerHTML, '');
    assert.match(h.status.textContent, status === 'unsupported' ? /Intel uses external/ : /PTARMIGAN_FLOW_SPEECH_CPP_PATH/);
  }
});

test('completion onboarding contains recovery only when the saved config is Haas', () => {
  const h = setup();
  assert.match(h.run('renderOnboarding()'), /data-speech-cpp/);
  h.run(`settingsDraft.model = 'moonshine:draft'`);
  assert.match(h.run('renderOnboarding()'), /data-speech-cpp/);
  h.run(`state.settings.model = 'moonshine:saved'`);
  assert.doesNotMatch(h.run('renderOnboarding()'), /data-speech-cpp/);
  assert.match(h.run(`renderModelCard({token:'haas:saved', downloaded:true}, 'moonshine:draft')`), /data-speech-cpp/);
});

test('Intel old/failed external runtime includes manual installation guidance', () => {
  const h = setup();
  h.push({status:'old', install_supported:false, install_allowed:false, operation_id:2});
  assert.match(h.status.textContent, /Runtime old.*Intel uses external/);
  assert.equal(h.action.innerHTML, '');
});

test('saved Haas missing runtime stays on completion recovery after permission setup', () => {
  const h = setup();
  h.run(`
    // Exercise render's route decision while suppressing only DOM construction.
    vmRender = ${source.match(/function render\(\) \{[\s\S]*?\n\}/)[0]};
    shell = () => {}; bindOnboarding = () => {}; bindSettings = () => {};
    state.setup_required = false; route = 'onboarding'; vmRender();
  `);
  assert.equal(h.run('route'), 'onboarding');
  h.push({status:'available', install_allowed:false, operation_id:2});
  h.run('vmRender()');
  assert.equal(h.run('route'), 'settings');
});
