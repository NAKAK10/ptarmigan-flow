const app = document.getElementById("app");
let state = null;
let route = "onboarding";
let nextId = 1;
const pending = new Map();
// token -> { status: "preparing" | "downloading" | "done" | "error", fraction, message }
// absence of an entry means "idle" (no download in progress).
const downloadStates = new Map();

// Dictionary editor draft state, kept OUTSIDE `state` on purpose: `state` is
// wholesale replaced by daemonState/permissionsChanged push events (see
// dispatch() above), which would otherwise wipe in-progress edits. As long as
// renderDictionary() renders from `dictDraft` (not from `state.dictionary`)
// once it exists, those push events are harmless to the editor.
// Shape: { exact: [row...], regex: [row...], dirty } where
// row = { id, key, valuesText, error }. error is null or { message }.
let dictDraft = null;
let dictRowSeq = 1;
// { kind: "success" | "error", text } | null — summary banner above the
// dictionary sections; cleared on the next edit.
let dictionaryMessage = null;

function bridge(action, payload = {}) {
  const id = String(nextId++);
  const message = { id, action, payload };
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    window.webkit.messageHandlers.bridge.postMessage(message);
  });
}

window.app = {
  dispatch(message) {
    if (message && message.id && pending.has(message.id)) {
      const callbacks = pending.get(message.id);
      pending.delete(message.id);
      if (message.ok) {
        callbacks.resolve(message.result);
      } else {
        callbacks.reject(new Error(message.error || ""));
      }
      return;
    }
    if (!message || !message.event) {
      return;
    }
    if (message.event === "permissionsChanged" || message.event === "daemonState") {
      state = message.payload;
      render();
      return;
    }
    if (message.event === "downloadProgress") {
      const payload = message.payload || {};
      const token = payload.model || state?.model || "";
      if (!token) {
        return;
      }
      if (payload.type === "progress") {
        const fraction = payload.fraction;
        downloadStates.set(token, {
          status: fraction === null || fraction === undefined ? "preparing" : "downloading",
          fraction,
          message: payload.message,
        });
      } else if (payload.type === "done") {
        downloadStates.set(token, { status: "done" });
      } else if (payload.type === "error") {
        downloadStates.set(token, { status: "error", message: payload.message });
      }
      // Never full-render here: it would wipe in-progress edits in the settings
      // form while a download runs. Patch only the affected model card.
      updateModelCardInPlace(token);
      return;
    }
    if (message.event === "routeChanged") {
      route = message.payload?.route || route;
      render();
    }
  },
};
window.app.dispatch = window.app.dispatch.bind(window.app);

function strings() {
  return state?.strings || {};
}

function t(key) {
  const table = strings();
  return table[key] || key;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function shell(content) {
  const running = state?.daemon_running
    ? t("dictation_running_menu")
    : t("dictation_stopped_menu");
  app.innerHTML = `
    <div class="topbar">
      <div class="brand">
        <img class="brand-logo" src="./ptarmigan-logo.png" alt="" />
        <div class="brand-text">
          <div class="brand-title">PtarmiganFlow</div>
          <div class="brand-status ${state?.daemon_running ? "running" : ""}">
            <span class="status-dot" aria-hidden="true"></span>${escapeHtml(running)}
          </div>
        </div>
      </div>
      <nav class="nav">
        ${navButton("onboarding", "route_onboarding")}
        ${navButton("settings", "route_settings")}
        ${navButton("dictionary", "route_dictionary")}
      </nav>
    </div>
    ${content}
  `;
  app.querySelectorAll("[data-route]").forEach((button) => {
    button.addEventListener("click", () => {
      route = button.dataset.route;
      render();
    });
  });
}

function navButton(target, key) {
  const active = route === target ? "active" : "";
  return `<button class="${active}" data-route="${target}">${escapeHtml(t(key))}</button>`;
}

function render() {
  if (!state) {
    return;
  }
  if (route === "settings") {
    shell(renderSettings());
    bindSettings();
    return;
  }
  if (route === "dictionary") {
    shell(renderDictionary());
    bindDictionary();
    return;
  }
  shell(renderOnboarding());
  bindOnboarding();
}

function hotkeyOptions() {
  return [
    ["right_cmd", "right_cmd"],
    ["left_cmd", "left_cmd"],
    ["right_shift", "right_shift"],
    ["left_shift", "left_shift"],
    ["right_alt", "right_alt"],
    ["left_alt", "left_alt"],
    ["right_ctrl", "right_ctrl"],
    ["left_ctrl", "left_ctrl"],
  ];
}

function renderOnboarding() {
  const steps = ["language", "hotkey", "microphone", "accessibility", "input_monitoring", "done"];
  const current = state.onboarding_step || "language";
  const index = Math.max(0, steps.indexOf(current));
  const dots = steps
    .map((step, i) => {
      const cls = i === index ? "active" : i < index ? "done" : "";
      return `<span class="dot ${cls}" aria-label="${escapeHtml(step)}"></span>`;
    })
    .join("");
  if (current === "language") {
    const language = state.language || "en";
    return `
      <section class="onboarding-wrap">
        <div class="card">
          <div class="steps">${dots}</div>
          <h1>${escapeHtml(t("choose_language_title"))}</h1>
          <p>${escapeHtml(t("choose_language_body"))}</p>
          <div class="actions">
            <button class="button ${language === "en" ? "primary" : ""}" data-language="en">${escapeHtml(t("language_english"))}</button>
            <button class="button ${language === "ja" ? "primary" : ""}" data-language="ja">${escapeHtml(t("language_japanese"))}</button>
            <button class="button ${language === "zh" ? "primary" : ""}" data-language="zh">${escapeHtml(t("language_chinese"))}</button>
          </div>
        </div>
      </section>
    `;
  }
  if (current === "hotkey") {
    const hotkey = state.settings?.hotkey || "right_cmd";
    const options = hotkeyOptions()
      .map(([value, label]) => {
        const attr = value === hotkey ? "selected" : "";
        return `<option value="${escapeHtml(value)}" ${attr}>${escapeHtml(label)}</option>`;
      })
      .join("");
    return `
      <section class="onboarding-wrap">
        <div class="card">
          <div class="steps">${dots}</div>
          <h1>${escapeHtml(t("hotkey_confirm_title"))}</h1>
          <p>${escapeHtml(t("hotkey_confirm_body"))}</p>
          <div class="form-row">
            <label for="onboarding-hotkey">${escapeHtml(t("hotkey_select_label"))}</label>
            <select id="onboarding-hotkey">
              ${options}
            </select>
          </div>
          <div class="actions">
            <button class="button primary" data-action="confirm-hotkey">${escapeHtml(t("hotkey_confirm_button"))}</button>
          </div>
          <div class="error"></div>
        </div>
      </section>
    `;
  }
  if (current === "done") {
    return `
      <section class="onboarding-wrap">
        <div class="card">
          <div class="steps">${dots}</div>
          <h1>${escapeHtml(t("done_title"))}</h1>
          <p>${escapeHtml(t("done_body"))}</p>
          <div class="actions">
            <button class="button primary" data-route="settings">${escapeHtml(t("settings_button"))}</button>
            <button class="button" data-action="toggle-login">${escapeHtml(t("login_at_startup_button"))}</button>
          </div>
          <div class="error">${escapeHtml(state.daemon_error_message || "")}</div>
        </div>
      </section>
    `;
  }
  return renderPermissionStep(current, dots);
}

function renderPermissionStep(kind, dots) {
  const granted = Boolean(state.permissions?.[kind]);
  const titleKey = `${kind}_title`;
  const bodyKey = `${kind}_body`;
  const restart = kind === "accessibility" || kind === "input_monitoring";
  return `
    <section class="onboarding-wrap">
      <div class="card">
        <div class="steps">${dots}</div>
        <h1>${escapeHtml(t(titleKey))}</h1>
        <p>${escapeHtml(t(bodyKey))}</p>
        <div class="status-pill ${granted ? "granted" : ""}">
          ${escapeHtml(granted ? t("status_granted") : t("status_waiting"))}
        </div>
        <div class="actions">
          <button class="button primary" data-permission="${kind}">${escapeHtml(t("allow_button"))}</button>
          <button class="button" data-settings="${kind}">${escapeHtml(t("open_system_settings_button"))}</button>
          ${restart ? `<button class="button" data-action="restart">${escapeHtml(t("restart_app_button"))}</button>` : ""}
        </div>
        ${restart ? `<p>${escapeHtml(t("restart_required_note"))}</p>` : ""}
      </div>
    </section>
  `;
}

function bindOnboarding() {
  app.querySelectorAll("[data-language]").forEach((button) => {
    button.addEventListener("click", async () => {
      await bridge("chooseLanguage", { code: button.dataset.language });
      state = await bridge("getState");
      render();
    });
  });
  app.querySelector("[data-action='confirm-hotkey']")?.addEventListener("click", async () => {
    const selectedHotkey =
      document.getElementById("onboarding-hotkey")?.value || state.settings?.hotkey;
    const result = await bridge("confirmHotkey", { hotkey: selectedHotkey }).catch((error) => {
      showError(error);
      return null;
    });
    if (!result) {
      return;
    }
    if (result.saved === false) {
      const target = app.querySelector(".error");
      if (target) {
        target.textContent = result.errors.join(", ");
      }
      return;
    }
    state = await bridge("getState");
    render();
  });
  app.querySelectorAll("[data-permission]").forEach((button) => {
    button.addEventListener("click", () =>
      bridge("requestPermission", { kind: button.dataset.permission }).catch(showError),
    );
  });
  app.querySelectorAll("[data-settings]").forEach((button) => {
    button.addEventListener("click", () =>
      bridge("openSystemSettings", { kind: button.dataset.settings }).catch(showError),
    );
  });
  bindSharedActions();
}

function renderSettings() {
  const settings = state.settings || {};
  const llm = settings.llm_correction || {};
  const models = state.models || [];
  return `
    <section class="grid">
      <div class="panel full">
        <h2>${escapeHtml(t("settings_models_section_title"))}</h2>
        <div class="model-list">
          ${models.map((model) => renderModelCard(model, settings.model)).join("")}
        </div>
      </div>
      <div class="panel">
        <h2>${escapeHtml(t("settings_window_title"))}</h2>
        ${selectRow("language", "settings_language_label", settings.language, [
          ["en", t("language_english")],
          ["ja", t("language_japanese")],
          ["zh", t("language_chinese")],
        ])}
        ${selectRow("hotkey", "settings_hotkey_label", settings.hotkey, [
          ...hotkeyOptions(),
        ])}
        ${selectRow("output_mode", "settings_output_mode_label", settings.output_mode, [
          ["direct_typing", t("output_direct_typing")],
          ["clipboard_paste", t("output_clipboard_paste")],
        ])}
        <div class="form-row">
          <button class="button" data-action="open-config">${escapeHtml(t("settings_open_config_button"))}</button>
        </div>
      </div>
      <div class="panel">
        <h2>${escapeHtml(t("settings_llm_section_title"))}</h2>
        ${selectRow("llm_mode", "settings_llm_mode_label", llm.mode, [
          ["always", t("settings_llm_mode_always")],
          ["never", t("settings_llm_mode_never")],
          ["ask", t("settings_llm_mode_ask")],
        ])}
        ${inputRow("llm_provider", "settings_llm_provider_label", llm.provider)}
        ${inputRow("llm_model", "settings_llm_model_label", llm.model)}
        ${inputRow("llm_base_url", "settings_llm_base_url_label", llm.base_url)}
      </div>
      <div class="panel full action-bar">
        <div id="settings-error" class="error"></div>
        <button class="button primary" data-action="save-settings">${escapeHtml(t("settings_save_button"))}</button>
      </div>
    </section>
  `;
}

function renderModelCard(model, selected) {
  const isSelected = selected === model.token;
  const selectedClass = isSelected ? "selected" : "";
  const download = downloadStates.get(model.token);
  const status = model.downloaded ? "downloaded" : download?.status || "idle";
  return `
    <div class="model-card ${selectedClass}" data-select-model="${escapeHtml(model.token)}" tabindex="0" role="button">
      <div class="model-info">
        <h3>${escapeHtml(model.label)}</h3>
        <div class="model-meta">${escapeHtml(model.description)}</div>
        <div class="model-token">${escapeHtml(model.token)}</div>
      </div>
      <div class="model-action">
        ${renderModelAction(model, status, download)}
      </div>
      ${renderModelProgress(status, download)}
    </div>
  `;
}

function renderModelAction(model, status, download) {
  if (status === "downloaded") {
    return `<span class="badge">${escapeHtml(t("settings_model_downloaded_badge"))}</span>`;
  }
  if (status === "done") {
    return `<span class="badge">${escapeHtml(t("settings_model_download_done_message"))}</span>`;
  }
  if (status === "preparing") {
    return `<span class="model-downloading">${escapeHtml(t("settings_model_preparing_message"))}</span>`;
  }
  if (status === "downloading") {
    const percent = fractionToPercent(download?.fraction);
    const text = t("download_in_progress_message").replace("{percent}", percent + "%");
    return `<span class="model-downloading">${escapeHtml(text)}</span>`;
  }
  if (status === "error") {
    return `
      <div class="model-download-error">
        <span class="model-error-text">${escapeHtml(downloadErrorText(download?.message))}</span>
        <button class="button" data-download-model="${escapeHtml(model.token)}">${escapeHtml(t("settings_model_retry_button"))}</button>
      </div>
    `;
  }
  return `<button class="button" data-download-model="${escapeHtml(model.token)}">${escapeHtml(t("settings_model_download_button"))}</button>`;
}

function renderModelProgress(status, download) {
  if (status === "preparing") {
    return `
      <div class="model-progress-row">
        <div class="progress indeterminate"><span></span></div>
      </div>
    `;
  }
  if (status === "downloading") {
    const percent = fractionToPercent(download?.fraction);
    return `
      <div class="model-progress-row">
        <div class="progress"><span style="--value: ${percent}%"></span></div>
        <span class="model-percent">${percent}%</span>
      </div>
    `;
  }
  return "";
}

function fractionToPercent(fraction) {
  const clamped = Math.max(0, Math.min(1, Number(fraction || 0)));
  return Math.round(clamped * 100);
}

// Formats the localized download-failure text; never returns an empty string.
// With no detail message, the ": {error}" tail (ASCII or fullwidth colon) is
// stripped from the template so we don't render a dangling separator.
function downloadErrorText(message) {
  const template = t("settings_model_download_error_message");
  if (message) {
    return template.replace("{error}", message);
  }
  const generic = template.replace(/[:：]?\s*\{error\}/, "").trim();
  return generic || template;
}

function findModelCardElement(token) {
  return Array.from(app.querySelectorAll("[data-select-model]")).find(
    (el) => el.dataset.selectModel === token,
  );
}

function updateModelCardInPlace(token) {
  if (!state || route !== "settings") {
    return;
  }
  const cardEl = findModelCardElement(token);
  if (!cardEl) {
    // User navigated away from settings; the Map already holds the latest
    // state and will be reflected next time the settings view renders.
    return;
  }
  const model = (state.models || []).find((entry) => entry.token === token);
  if (!model) {
    return;
  }
  cardEl.outerHTML = renderModelCard(model, state.settings?.model);
  bindModelCard(findModelCardElement(token));
}

// `button` here is the whole `.model-card` element (selection target), named
// to match the click-to-select convention used across this file.
function bindModelCard(button) {
  if (!button) {
    return;
  }
  button.addEventListener("click", () => {
    state.settings.model = button.dataset.selectModel;
    render();
  });
  button.addEventListener("keydown", (event) => {
    // Keydown bubbles up from the nested Download/Retry <button>; only handle
    // keypresses aimed at the card itself so the inner button keeps its
    // native Enter/Space activation.
    if (event.target !== event.currentTarget) {
      return;
    }
    if (event.key === "Enter" || event.key === " " || event.key === "Spacebar") {
      event.preventDefault();
      state.settings.model = button.dataset.selectModel;
      render();
    }
  });
  button
    .querySelector("[data-download-model]")
    ?.addEventListener("click", handleDownloadModelClick);
}

async function handleDownloadModelClick(event) {
  // The card itself is also click-bound to model selection; downloading
  // must not trigger that.
  event.stopPropagation();
  const token = event.currentTarget.dataset.downloadModel;
  if (!token) {
    return;
  }
  downloadStates.set(token, { status: "preparing" });
  render();
  let result = null;
  try {
    result = await bridge("downloadModel", { model: token });
  } catch (error) {
    downloadStates.set(token, { status: "error", message: error.message });
    updateModelCardInPlace(token);
    return;
  }
  if (result?.started === false) {
    if (result.already_downloaded) {
      downloadStates.delete(token);
      state = await bridge("getState");
      render();
      return;
    }
    // Store the raw detail; downloadErrorText() applies the localized
    // "download failed" template at render time.
    downloadStates.set(token, { status: "error", message: (result.errors || []).join(", ") });
    updateModelCardInPlace(token);
    return;
  }
  // result.started === true: leave status "preparing"; downloadProgress
  // events drive the rest of the state machine from here.
}

function selectRow(id, labelKey, selected, options) {
  return `
    <div class="form-row">
      <label for="${id}">${escapeHtml(t(labelKey))}</label>
      <select id="${id}">
        ${options
          .map(([value, label]) => {
            const attr = value === selected ? "selected" : "";
            return `<option value="${escapeHtml(value)}" ${attr}>${escapeHtml(label)}</option>`;
          })
          .join("")}
      </select>
    </div>
  `;
}

function inputRow(id, labelKey, value) {
  return `
    <div class="form-row">
      <label for="${id}">${escapeHtml(t(labelKey))}</label>
      <input id="${id}" value="${escapeHtml(value || "")}" />
    </div>
  `;
}

function settingsPayload(modelOverride = null) {
  const settings = state.settings || {};
  return {
    model: modelOverride || settings.model,
    language: document.getElementById("language")?.value || settings.language,
    hotkey: document.getElementById("hotkey")?.value || settings.hotkey,
    output_mode: document.getElementById("output_mode")?.value || settings.output_mode,
    llm_correction: {
      mode: document.getElementById("llm_mode")?.value || settings.llm_correction?.mode,
      provider: document.getElementById("llm_provider")?.value || settings.llm_correction?.provider,
      model: document.getElementById("llm_model")?.value || settings.llm_correction?.model,
      base_url: document.getElementById("llm_base_url")?.value || settings.llm_correction?.base_url,
    },
  };
}

function bindSettings() {
  app.querySelector("[data-action='save-settings']")?.addEventListener("click", async () => {
    await saveSettings();
  });
  app.querySelector("[data-action='open-config']")?.addEventListener("click", () => {
    bridge("openConfigFile").catch(showError);
  });
  app.querySelectorAll("[data-select-model]").forEach((cardEl) => {
    bindModelCard(cardEl);
  });
}

async function saveSettings(modelOverride = null) {
  const result = await bridge("saveSettings", settingsPayload(modelOverride)).catch((error) => {
    showError(error, "settings-error");
    return null;
  });
  if (!result) {
    return;
  }
  if (result.saved === false) {
    const target = document.getElementById("settings-error");
    if (target) {
      target.textContent = result.errors.join(", ");
    }
    return;
  }
  state = await bridge("getState");
  render();
}

function renderDictionary() {
  if (!dictDraft) {
    dictDraft = buildDictDraft(state.dictionary);
  }
  return `
    <section class="grid">
      <div class="panel full">
        <h2>${escapeHtml(t("dictionary_editor_title"))}</h2>
        ${dictionarySection("exact", dictDraft.exact, "dictionary_exact_rules_title", "dictionary_exact_rules_description", "dictionary_add_exact_button")}
        ${dictionarySection("regex", dictDraft.regex, "dictionary_regex_rules_title", "dictionary_regex_rules_description", "dictionary_add_regex_button")}
        <div class="actions">
          <button class="button" data-add-dictionary="exact">${escapeHtml(t("dictionary_add_exact_button"))}</button>
          <button class="button" data-add-dictionary="regex">${escapeHtml(t("dictionary_add_regex_button"))}</button>
          <button class="button primary" data-action="save-dictionary">${escapeHtml(t("dictionary_save_button"))}</button>
          <span class="unsaved-indicator${dictDraft.dirty ? "" : " hidden"}" data-dictionary-unsaved>${escapeHtml(t("dictionary_unsaved_changes_label"))}</span>
        </div>
        ${dictionaryMessageBox()}
      </div>
    </section>
  `;
}

function buildDictDraft(dictionary) {
  const source = dictionary || { exact: {}, regex: {} };
  return {
    exact: Object.entries(source.exact || {}).map(([key, values]) => makeDictRow(key, values)),
    regex: Object.entries(source.regex || {}).map(([key, values]) => makeDictRow(key, values)),
    dirty: false,
  };
}

function makeDictRow(key, values) {
  return { id: dictRowSeq++, key, valuesText: (values || []).join(", "), error: null };
}

function dictionaryMessageBox() {
  if (!dictionaryMessage) {
    return `<div id="dictionary-error"></div>`;
  }
  const cls = dictionaryMessage.kind === "success" ? "alert success" : "alert error";
  return `<div id="dictionary-error" class="${cls}">${escapeHtml(dictionaryMessage.text)}</div>`;
}

function dictionarySection(section, rows, titleKey, descriptionKey, addLabelKey) {
  const body = rows.length
    ? rows.map((row) => dictionaryRow(section, row)).join("")
    : dictionaryEmptyState(section, descriptionKey, addLabelKey);
  return `
    <div class="dictionary-list" data-section="${section}">
      <h3>${escapeHtml(t(titleKey))}</h3>
      <p class="dictionary-section-description">${escapeHtml(t(descriptionKey))}</p>
      ${body}
    </div>
  `;
}

function dictionaryEmptyState(section, descriptionKey, addLabelKey) {
  return `
    <div class="empty-state">
      <p>${escapeHtml(t("dictionary_no_rules"))}</p>
      <p class="dictionary-section-description">${escapeHtml(t(descriptionKey))}</p>
      <button class="button ghost" data-add-dictionary="${section}">${escapeHtml(t(addLabelKey))}</button>
    </div>
  `;
}

function dictionaryRow(section, row) {
  const invalidClass = row.error ? " invalid" : "";
  return `
    <div class="dictionary-row${invalidClass}" data-dictionary-row="${row.id}" data-section="${section}">
      <div class="dictionary-row-header">
        <input
          data-dictionary-key
          value="${escapeHtml(row.key)}"
          aria-label="${escapeHtml(t("dictionary_canonical_label"))}"
        />
        <button class="button ghost danger" data-delete-row>${escapeHtml(t("dictionary_delete_button"))}</button>
      </div>
      <label>${escapeHtml(t("dictionary_candidates_patterns_label"))}</label>
      <textarea data-dictionary-values>${escapeHtml(row.valuesText)}</textarea>
      ${row.error ? `<div class="field-error">${escapeHtml(row.error.message)}</div>` : ""}
    </div>
  `;
}

function bindDictionary() {
  app.querySelectorAll("[data-dictionary-row]").forEach((rowEl) => {
    const id = Number(rowEl.dataset.dictionaryRow);
    const section = rowEl.dataset.section;
    const keyInput = rowEl.querySelector("[data-dictionary-key]");
    const valuesInput = rowEl.querySelector("[data-dictionary-values]");
    keyInput?.addEventListener("input", () => {
      updateDictRow(section, id, { key: keyInput.value });
    });
    valuesInput?.addEventListener("input", () => {
      updateDictRow(section, id, { valuesText: valuesInput.value });
    });
    rowEl.querySelector("[data-delete-row]")?.addEventListener("click", () => {
      deleteDictRow(section, id);
    });
  });
  app.querySelectorAll("[data-add-dictionary]").forEach((button) => {
    button.addEventListener("click", () => {
      addDictRow(button.dataset.addDictionary);
    });
  });
  app.querySelector("[data-action='save-dictionary']")?.addEventListener("click", saveDictionary);
}

function findDictRow(section, id) {
  const list = dictDraft?.[section];
  return list ? list.find((row) => row.id === id) : undefined;
}

// Edits patch the draft in place and only touch the DOM directly (dirty
// indicator, clearing a row's stale error) — never render() here, or every
// keystroke would rebuild the section and drop focus/cursor position.
function updateDictRow(section, id, patch) {
  const row = findDictRow(section, id);
  if (!row) {
    return;
  }
  Object.assign(row, patch);
  const hadError = Boolean(row.error);
  row.error = null;
  dictDraft.dirty = true;
  clearDictionaryMessage();
  if (hadError) {
    const rowEl = app.querySelector(`[data-dictionary-row="${id}"]`);
    rowEl?.classList.remove("invalid");
    rowEl?.querySelector(".field-error")?.remove();
  }
  app.querySelector("[data-dictionary-unsaved]")?.classList.toggle("hidden", !dictDraft.dirty);
}

function addDictRow(section) {
  if (!dictDraft) {
    return;
  }
  const list = dictDraft[section];
  const baseKey =
    section === "exact" ? t("dictionary_new_exact_rule") : t("dictionary_new_regex_rule");
  const defaultValue =
    section === "exact" ? t("dictionary_default_candidate") : t("dictionary_default_pattern");
  list.push({
    id: dictRowSeq++,
    key: uniqueDictionaryKey(list, baseKey),
    valuesText: defaultValue,
    error: null,
  });
  dictDraft.dirty = true;
  clearDictionaryMessage();
  render();
}

function deleteDictRow(section, id) {
  if (!dictDraft) {
    return;
  }
  const list = dictDraft[section];
  const index = list.findIndex((row) => row.id === id);
  if (index === -1) {
    return;
  }
  list.splice(index, 1);
  dictDraft.dirty = true;
  clearDictionaryMessage();
  render();
}

function uniqueDictionaryKey(rows, base) {
  const existing = new Set(rows.map((row) => row.key));
  if (!existing.has(base)) {
    return base;
  }
  let index = 2;
  while (existing.has(`${base} ${index}`)) {
    index += 1;
  }
  return `${base} ${index}`;
}

function clearDictionaryMessage() {
  if (!dictionaryMessage) {
    return;
  }
  dictionaryMessage = null;
  const target = document.getElementById("dictionary-error");
  if (target) {
    target.className = "";
    target.textContent = "";
  }
}

// Builds the saveDictionary payload from the draft and runs the client-side
// duplicate-key check (per section) before any bridge call. Rows with both an
// empty key and no values are dropped silently; a row with a key but no
// values is still sent so the backend's non-empty-values validation reports
// it against that row.
function buildDictionaryPayload() {
  const computed = new Map();
  const grouped = { exact: new Map(), regex: new Map() };
  for (const section of ["exact", "regex"]) {
    for (const row of dictDraft[section]) {
      row.error = null;
      const key = row.key.trim();
      const values = row.valuesText
        .split(/[,\n]/)
        .map((item) => item.trim())
        .filter(Boolean);
      computed.set(row.id, { key, values });
      if (!key && values.length === 0) {
        continue;
      }
      if (!grouped[section].has(key)) {
        grouped[section].set(key, []);
      }
      grouped[section].get(key).push(row);
    }
  }
  let hasDuplicates = false;
  for (const section of ["exact", "regex"]) {
    for (const [key, rows] of grouped[section].entries()) {
      if (rows.length > 1) {
        hasDuplicates = true;
        const message = t("dictionary_duplicate_key_message").replace("{key}", key);
        rows.forEach((row) => {
          row.error = { message };
        });
      }
    }
  }
  if (hasDuplicates) {
    return { payload: null, hasDuplicates: true };
  }
  const payload = { exact: {}, regex: {} };
  for (const section of ["exact", "regex"]) {
    for (const row of dictDraft[section]) {
      const { key, values } = computed.get(row.id);
      if (!key && values.length === 0) {
        continue;
      }
      payload[section][key] = values;
    }
  }
  return { payload, hasDuplicates: false };
}

// error = { section, key, index, pattern, message } from the backend
// (index is 0-based within the rule's value list, or null for row-level
// errors such as an empty key).
function composeDictionaryErrorMessage(error) {
  if (error.index !== null && error.index !== undefined) {
    return `#${error.index + 1} (${error.pattern}): ${error.message}`;
  }
  return error.message;
}

// Keys are unique within a section after the client-side duplicate check, so
// matching errors back to rows by (section, trimmed key) is unambiguous.
function applyDictionarySaveErrors(errors) {
  errors.forEach((error) => {
    const row = (dictDraft[error.section] || []).find((r) => r.key.trim() === error.key);
    if (row) {
      row.error = { message: composeDictionaryErrorMessage(error) };
    }
  });
  const first = errors[0];
  dictionaryMessage = first ? { kind: "error", text: composeDictionaryErrorMessage(first) } : null;
}

async function saveDictionary() {
  if (!dictDraft) {
    return;
  }
  const { payload, hasDuplicates } = buildDictionaryPayload();
  if (hasDuplicates) {
    render();
    return;
  }
  const result = await bridge("saveDictionary", payload).catch((error) => {
    dictionaryMessage = { kind: "error", text: `${t("webui_error_title")}: ${error.message || error}` };
    render();
    return null;
  });
  if (!result) {
    return;
  }
  if (result.saved === false) {
    applyDictionarySaveErrors(result.errors || []);
    render();
    return;
  }
  dictDraft = null;
  state = await bridge("getState");
  dictionaryMessage = { kind: "success", text: t("dictionary_saved_message") };
  render();
}

function bindSharedActions() {
  app.querySelector("[data-action='toggle-login']")?.addEventListener("click", async () => {
    await bridge("toggleLogin").catch(showError);
    state = await bridge("getState");
    render();
  });
  app.querySelector("[data-action='restart']")?.addEventListener("click", () => {
    bridge("restartApp").catch(showError);
  });
}

function showError(error, targetId = null) {
  const title = state && state.strings ? state.strings.webui_error_title : "";
  const text = `${title}: ${error.message || error}`;
  const target = targetId ? document.getElementById(targetId) : app.querySelector(".error");
  if (target) {
    target.textContent = text;
  }
}

async function boot() {
  state = await bridge("getState");
  render();
}

boot().catch(showError);
