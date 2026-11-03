// Real onboarding listeners with a mocked native bridge; no browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function setup() {
  const requests = [];
  const error = { textContent: '' };
  const buttons = ['en', 'ja', 'zh'].map(language => ({
    dataset: { language },
    addEventListener(type, listener) { this[type] = listener; },
  }));
  const app = {
    querySelector(selector) { return selector === '.error' ? error : null; },
    querySelectorAll(selector) { return selector === '[data-language]' ? buttons : []; },
  };
  const context = vm.createContext({
    document: { getElementById() { return app; } },
    window: { webkit: { messageHandlers: { bridge: {
      postMessage(message) { requests.push(message); },
    } } } },
  });
  const source = fs.readFileSync(path.join(__dirname, '../src/ptarmigan_flow/webui/app.js'), 'utf8');
  vm.runInContext(source.replace('boot().catch(showError);', ''), context);
  const run = code => vm.runInContext(code, context);
  run(`
    state = {
      language: 'ja', onboarding_step: 'language', strings: {},
      settings: { model: 'haas:sakasegawa/Haas-ja-v1.0-GGUF' },
    };
    let renders = 0;
    render = () => { renders++; };
    bindOnboarding();
  `);
  const reply = (index, result, message) => context.window.app.dispatch({
    id: requests[index].id, ok: !message, result, error: message,
  });
  return { requests, error, buttons, reply, run };
}

test('rejected language stays in onboarding and shows guidance; Japanese can proceed', async () => {
  const h = setup();
  assert.match(h.run('renderOnboarding()'), /class="error" role="alert"/);
  const rejected = h.buttons[0].click();
  assert.equal(h.requests[0].action, 'chooseLanguage');
  const message = 'Haas is Japanese-only; select Japanese.';
  h.reply(0, { saved: false, errors: ['language'], message });
  await rejected;
  assert.equal(h.requests.length, 1); // No getState or step advance after rejection.
  assert.equal(h.run('state.onboarding_step'), 'language');
  assert.equal(h.run('state.language'), 'ja');
  assert.equal(h.run('renders'), 0);
  assert.equal(h.error.textContent, message);

  const accepted = h.buttons[1].click();
  h.reply(1, { language: 'ja', onboarding_step: 'hotkey' });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.requests[2].action, 'getState');
  h.reply(2, { language: 'ja', onboarding_step: 'hotkey' });
  await accepted;
  assert.equal(h.run('state.onboarding_step'), 'hotkey');
  assert.equal(h.run('renders'), 1);
});

test('transport rejection does not advance or request a new state', async () => {
  const h = setup();
  const rejected = h.buttons[0].click();
  h.reply(0, null, 'Unable to save language');
  await rejected;
  assert.equal(h.requests.length, 1);
  assert.equal(h.run('state.onboarding_step'), 'language');
  assert.match(h.error.textContent, /Unable to save language/);
});
