const app = document.getElementById("app");
let state = null;
let route = "onboarding";
let nextId = 1;
const pending = new Map();
const downloadProgress = new Map();

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
      if (token) {
        downloadProgress.set(token, payload);
      }
      render();
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
  const active = isSelected ? "primary" : "";
  const selectedClass = isSelected ? "selected" : "";
  const progress = downloadProgress.get(model.token);
  const fraction = Math.max(0, Math.min(1, Number(progress?.fraction || 0)));
  const percent = Math.round(fraction * 100);
  const downloadingText =
    progress && !model.downloaded
      ? t("download_in_progress_message").replace("{percent}", percent + "%")
      : "";
  return `
    <div class="model-card ${selectedClass}" data-select-model="${escapeHtml(model.token)}">
      <div class="model-info">
        <h3>${escapeHtml(model.label)}</h3>
        <div class="model-meta">${escapeHtml(model.description)}</div>
        <div class="model-token">${escapeHtml(model.token)}</div>
      </div>
      <div class="model-action">
        ${
          model.downloaded
            ? `<span class="badge">${escapeHtml(t("settings_model_downloaded_badge"))}</span>`
            : progress
              ? `<span class="model-downloading">${escapeHtml(downloadingText)}</span>`
              : `<button class="button ${active}" data-download-model="${escapeHtml(model.token)}">${escapeHtml(t("settings_model_download_button"))}</button>`
        }
      </div>
      ${
        progress
          ? `<div class="progress"><span style="--value: ${percent}%"></span></div>`
          : ""
      }
    </div>
  `;
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
  app.querySelectorAll("[data-select-model]").forEach((button) => {
    button.addEventListener("click", () => {
      state.settings.model = button.dataset.selectModel;
      render();
    });
  });
  app.querySelectorAll("[data-download-model]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      await saveSettings(button.dataset.downloadModel);
      await bridge("startDictation");
      state = await bridge("getState");
      render();
    });
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
  const dictionary = state.dictionary || { exact: {}, regex: {} };
  return `
    <section class="grid">
      <div class="panel full">
        <h2>${escapeHtml(t("dictionary_editor_title"))}</h2>
        ${dictionarySection("exact", "dictionary_exact_rules_title", dictionary.exact)}
        ${dictionarySection("regex", "dictionary_regex_rules_title", dictionary.regex)}
        <div class="actions">
          <button class="button" data-add-dictionary="exact">${escapeHtml(t("dictionary_add_exact_button"))}</button>
          <button class="button" data-add-dictionary="regex">${escapeHtml(t("dictionary_add_regex_button"))}</button>
          <button class="button primary" data-action="save-dictionary">${escapeHtml(t("dictionary_save_button"))}</button>
        </div>
        <div id="dictionary-error" class="error"></div>
      </div>
    </section>
  `;
}

function dictionarySection(section, titleKey, entries) {
  const rows = Object.entries(entries || {});
  return `
    <div class="dictionary-list" data-section="${section}">
      <h3>${escapeHtml(t(titleKey))}</h3>
      ${
        rows.length
          ? rows.map(([key, values]) => dictionaryRow(section, key, values)).join("")
          : `<p>${escapeHtml(t("dictionary_no_rules"))}</p>`
      }
    </div>
  `;
}

function dictionaryRow(section, key, values) {
  return `
    <div class="dictionary-row" data-dictionary-row="${section}">
      <label>${escapeHtml(t("dictionary_canonical_label"))}</label>
      <input data-dictionary-key value="${escapeHtml(key)}" />
      <label>${escapeHtml(t("dictionary_candidates_patterns_label"))}</label>
      <textarea data-dictionary-values>${escapeHtml((values || []).join(", "))}</textarea>
      <button class="button" data-delete-row>${escapeHtml(t("dictionary_delete_button"))}</button>
    </div>
  `;
}

function bindDictionary() {
  app.querySelectorAll("[data-add-dictionary]").forEach((button) => {
    button.addEventListener("click", () => {
      const section = button.dataset.addDictionary;
      const table = state.dictionary?.[section] || {};
      const key =
        section === "exact" ? t("dictionary_new_exact_rule") : t("dictionary_new_regex_rule");
      table[uniqueDictionaryKey(table, key)] = [
        section === "exact" ? t("dictionary_default_candidate") : t("dictionary_default_pattern"),
      ];
      state.dictionary[section] = table;
      render();
    });
  });
  app.querySelectorAll("[data-delete-row]").forEach((button) => {
    button.addEventListener("click", () => {
      button.closest("[data-dictionary-row]")?.remove();
    });
  });
  app.querySelector("[data-action='save-dictionary']")?.addEventListener("click", saveDictionary);
}

function uniqueDictionaryKey(table, base) {
  if (!table[base]) {
    return base;
  }
  let index = 2;
  while (table[`${base} ${index}`]) {
    index += 1;
  }
  return `${base} ${index}`;
}

async function saveDictionary() {
  const payload = { exact: {}, regex: {} };
  app.querySelectorAll("[data-dictionary-row]").forEach((row) => {
    const section = row.dataset.dictionaryRow;
    const key = row.querySelector("[data-dictionary-key]").value.trim();
    const values = row
      .querySelector("[data-dictionary-values]")
      .value.split(/[,\n]/)
      .map((item) => item.trim())
      .filter(Boolean);
    payload[section][key] = values;
  });
  const result = await bridge("saveDictionary", payload).catch((error) => {
    showError(error, "dictionary-error");
    return null;
  });
  if (!result) {
    return;
  }
  if (result.saved === false) {
    const target = document.getElementById("dictionary-error");
    if (target) {
      target.textContent = result.errors.map((error) => error.message).join(", ");
    }
    return;
  }
  state = await bridge("getState");
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
