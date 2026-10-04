// Runs the real frontend handlers with deferred bridge replies and a minimal
// DOM stub. Chrome preview verification covers real layout/live-region markup.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const strings = JSON.parse(fs.readFileSync(0, "utf8"));
const requests = [];
let elements = {};
const controls = [{ disabled: false }, { disabled: false }];
function element() {
  return {
    textContent: "", className: "", disabled: false, attrs: {},
    setAttribute(key, value) { this.attrs[key] = value; },
    addEventListener() {},
  };
}
const app = {
  html: "",
  set innerHTML(html) {
    this.html = html;
    elements = {};
    const scope = html.includes('id="settings-error"') ? "settings" : "dictionary";
    const status = elements[`${scope}-error`] = element();
    status.textContent = html.match(new RegExp(`id="${scope}-error"[^>]*>([^<]*)`))[1];
    const button = elements[`save-${scope}`] = element();
    const tag = html.match(new RegExp(`data-action="save-${scope}"([^>]*)>([^<]*)`));
    button.disabled = tag[1].includes("disabled");
    button.attrs["aria-busy"] = tag[1].match(/aria-busy="([^"]+)"/)[1];
    button.textContent = tag[2];
  },
  querySelector(selector) {
    const match = selector.match(/data-action=['"]([^'"]+)/);
    return match ? elements[match[1]] || null : null;
  },
  querySelectorAll(selector) {
    return selector.includes("[data-dictionary-row] input") ? controls : [];
  },
};
const context = vm.createContext({
  document: { getElementById(id) { return id === "app" ? app : elements[id] || null; } },
  window: { webkit: { messageHandlers: { bridge: { postMessage(message) { requests.push(message); } } } } },
});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
const run = (script) => vm.runInContext(script, context);
const tick = async () => { await Promise.resolve(); await Promise.resolve(); };
const snapshot = {
  strings, settings: {
    model: "moonshine:moonshine/tiny", language: "en", hotkey: "right_cmd",
    output_mode: "direct_typing", llm_correction: {
      mode: "never", provider: "ollama", model: "demo", base_url: "http://localhost:11434",
    },
  },
  dictionary: { exact: { Demo: ["demo"] }, regex: {} },
  models: [], setup_required: false,
};
async function reply(action, result, error) {
  const request = requests.shift();
  assert.equal(request?.action, action);
  context.window.app.dispatch({ id: request.id, ok: !error, result, error });
  await tick();
}
function checkPending(scope) {
  assert.equal(elements[`save-${scope}`].disabled, true);
  assert.equal(elements[`save-${scope}`].attrs["aria-busy"], "true");
  assert.equal(elements[`${scope}-error`].textContent, strings.save_in_progress);
  assert.ok(!elements[`${scope}-error`].textContent.includes("%"));
}
function checkError(scope, text) {
  assert.equal(elements[`save-${scope}`].disabled, false);
  assert.equal(elements[`save-${scope}`].attrs["aria-busy"], "false");
  assert.equal(elements[`save-${scope}`].textContent, strings.save_retry_button);
  assert.ok(elements[`${scope}-error`].textContent.includes(text));
  assert.ok(elements[`${scope}-error`].textContent.includes(strings.save_retry_hint));
}
(async () => {
  await reply("getState", snapshot);
  for (const scope of ["settings", "dictionary"]) {
    context.window.app.dispatch({ event: "routeChanged", payload: { route: scope } });
    assert.match(app.html, /role="status" aria-live="polite" aria-atomic="true"/);
    if (scope === "dictionary") run("dictDraft.dirty = true");
    const save = () => run(scope === "settings" ? "saveSettings()" : "saveDictionary()");
    // Immediate pending, duplicate click guard, and pending through refresh.
    let operation = save();
    checkPending(scope);
    await save();
    assert.equal(requests.length, 1);
    context.window.app.dispatch({ event: "daemonState", payload: snapshot });
    assert.match(app.html, /aria-busy="true" disabled/);
    await save();
    assert.equal(requests.length, 1);
    await reply(`save${scope === "settings" ? "Settings" : "Dictionary"}`, { saved: true });
    checkPending(scope);
    assert.equal(requests[0].action, "getState");
    await save();
    assert.equal(requests.length, 1);
    await reply("getState", snapshot);
    await operation;
    assert.equal(elements[`save-${scope}`].disabled, false);
    assert.equal(elements[`${scope}-error`].textContent, strings[`${scope}_saved_message`]);
    // Both transport failures and refresh failures release the guard for retry.
    for (const refreshFailure of [false, true]) {
      if (scope === "dictionary") run("dictDraft.dirty = true");
      operation = save();
      const action = `save${scope === "settings" ? "Settings" : "Dictionary"}`;
      if (refreshFailure) {
        await reply(action, { saved: true });
        await reply("getState", null, "demo refresh failure");
      } else {
        await reply(action, null, "demo save failure");
      }
      await operation;
      checkError(scope, refreshFailure ? "demo refresh failure" : "demo save failure");
      if (refreshFailure) {
        assert.ok(elements[`${scope}-error`].textContent.includes(
          strings.save_refresh_failed_message.split("{error}")[0],
        ));
      }
      if (scope === "dictionary") {
        assert.equal(run("dictDraft.exact[0].key"), "Demo");
        assert.ok(controls.every((control) => !control.disabled));
      }
    }
    operation = save();
    if (scope === "settings") {
      await reply("saveSettings", { saved: false, errors: ["output_mode", "llm_correction.mode", "future.internal_key"] });
      await operation;
      checkError(scope, strings.settings_output_mode_label);
      assert.ok(elements["settings-error"].textContent.includes(strings.settings_llm_mode_label));
      assert.ok(!elements["settings-error"].textContent.includes("output_mode"));
      assert.ok(!elements["settings-error"].textContent.includes("llm_correction"));
      assert.ok(!elements["settings-error"].textContent.includes("future.internal_key"));
      const allFields = run('settingsValidationText(["model", "language", "hotkey", "output_mode", "llm_correction.mode", "llm_correction.provider", "llm_correction.model", "llm_correction.base_url"])');
      for (const key of ["settings_model_label", "settings_language_label", "settings_hotkey_label", "settings_output_mode_label", "settings_llm_mode_label", "settings_llm_provider_label", "settings_llm_model_label", "settings_llm_base_url_label"]) {
        assert.ok(allFields.includes(strings[key]));
      }
    } else {
      assert.ok(controls.every((control) => control.disabled));
      await reply("saveDictionary", { saved: false, errors: [{ section: "exact", key: "Demo", message: "demo invalid rule" }] });
      await operation;
      checkError(scope, "demo invalid rule");
      assert.equal(run("dictDraft.exact[0].error.message"), "demo invalid rule");
      run('dictDraft.exact.push(makeDictRow("exact", "Demo", ["duplicate"])); dictDraft.dirty = true');
      await save();
      assert.equal(requests.length, 0);
      assert.ok(run("dictionaryMessage.text").includes(strings.dictionary_duplicate_key_message.replace("{key}", "Demo")));
      run("dictDraft.exact.pop()");
    }
    // Successful retry must actually submit again and wait for state reflection.
    operation = save();
    await reply(`save${scope === "settings" ? "Settings" : "Dictionary"}`, { saved: true });
    await reply("getState", snapshot);
    await operation;
    assert.equal(elements[`${scope}-error`].textContent, strings[`${scope}_saved_message`]);
  }
  assert.equal(requests.length, 0);
  console.log("save feedback regression checks passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });
