const app = document.getElementById("app");
let state = null;
let route = "onboarding";
let nextId = 1;
const pending = new Map();
// token -> { status: "preparing" | "downloading" | "done" | "error" | "busy", fraction, message }
// absence of an entry means "idle" (no download in progress).
const downloadStates = new Map();
let activeDownloadToken = null;
// Only edited fields override snapshots, so native state updates still reach
// untouched settings. Keep these outside `state`, like the dictionary draft.
const settingsDraft = {};
// Preserve whitespace/blank lines independently of the normalized glossary.
let glossaryDraftText = null;
let glossaryDraftRevision = 0;
// Runtime installation has its own lifecycle; GGUF download flags stay independent.
let speechCppStatus = null;
let speechCppRequestPending = false;
let speechCppRequestOperationId = 0;

// Dictionary editor draft state, kept OUTSIDE `state` on purpose: `state` is
// wholesale replaced by daemonState/permissionsChanged push events (see
// dispatch() above), which would otherwise wipe in-progress edits. As long as
// renderDictionary() renders from `dictDraft` (not from `state.dictionary`)
// once it exists, those push events are harmless to the editor.
// Shape: { exact: [row...], regex: [row...], dirty } where
// row = { id, key, valuesText, error }. error is null or { message }.
let dictDraft = null;
let dictRowSeq = 1;
// Save feedback survives re-renders; pending includes the getState refresh.
let settingsSaving = false;
let settingsMessage = null;
let glossarySaving = false;
let glossaryMessage = null;
// Serialize scoped settings writes.
// Refreshes/feedback remain independent, so both screens can keep editing.
const settingsWrites = [];
let settingsWriteBusy = false;
let dictionarySaving = false;
// { kind: "success" | "error", text } | null; cleared on the next edit.
let dictionaryMessage = null;

// Active press-to-set hotkey capture session, or null when idle. Holds a
// `cleanup` callback that removes the window-level listeners, tells the
// native side capture has ended (so the dictation hotkey monitor stops
// suppressing itself), and resets the widget's UI. Kept as a single
// module-level slot (only one capture widget is ever visible at a time)
// so a route change / re-render can always find and tear it down.
let hotkeyCapture = null;

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
    if (message.event === "speechCppStatus") {
      acceptSpeechCppStatus(message.payload);
      updateSpeechCppInPlace();
      return;
    }
    if (message.event === "permissionsChanged" || message.event === "daemonState") {
      acceptSpeechCppStatus(message.payload?.speech_cpp, true);
      // Native download completion also pushes daemonState. While settings is
      // open, update the cards/status without replacing any live form inputs.
      if (message.event === "daemonState" && state && ["settings", "glossary"].includes(route)) {
        state = message.payload;
        if (route === "settings") {
          updateSettingsInputsInPlace();
          updateModelCardsInPlace();
        } else {
          const input = document.getElementById("llm_glossary");
          const text = glossaryText(state.settings);
          if (input && input.value !== text) input.value = text;
        }
        updateSaveFeedback(route);
        const status = app.querySelector(".brand-status");
        if (status) {
          status.classList.toggle("running", Boolean(state.daemon_running));
          status.innerHTML = `<span class="status-dot" aria-hidden="true"></span>${escapeHtml(t(state.daemon_running ? "dictation_running_menu" : "dictation_stopped_menu"))}`;
        }
      } else {
        state = message.payload;
        render();
      }
      return;
    }
    if (message.event === "downloadProgress") {
      const payload = message.payload || {};
      const token = payload.model || state?.model || "";
      if (!token) {
        return;
      }
      if (payload.type === "progress") {
        activeDownloadToken = token;
        const fraction = payload.fraction;
        downloadStates.set(token, {
          status: fraction === null || fraction === undefined ? "preparing" : "downloading",
          fraction,
          message: payload.message,
        });
      } else if (payload.type === "done") {
        if (activeDownloadToken === token) activeDownloadToken = null;
        downloadStates.set(token, { status: "done" });
        void refreshDownloadedModels(token);
      } else if (payload.type === "error") {
        if (activeDownloadToken === token) activeDownloadToken = null;
        downloadStates.set(token, { status: "error", message: payload.message });
      } else {
        return;
      }
      // Global Download/Retry availability changes, but never replace the form.
      updateModelCardsInPlace();
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
        ${state.setup_required ? navButton("onboarding", "route_onboarding") : ""}
        ${navButton("settings", "route_settings")}
        ${navButton("dictionary", "route_dictionary")}
        ${navButton("glossary", "route_glossary")}
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
  acceptSpeechCppStatus(state.speech_cpp, true);
  // Any in-progress hotkey capture is about to lose its DOM (rebuilt below);
  // tear it down so the native side stops suppressing the dictation hotkey.
  endActiveHotkeyCapture();
  const haasNeedsRuntime = (state.settings?.model || state.model || "").startsWith("haas:") &&
    currentSpeechCppStatus().status !== "available";
  if (route === "onboarding" && !state.setup_required && !haasNeedsRuntime) {
    route = "settings";
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
  if (route === "glossary") {
    shell(renderGlossary());
    bindGlossary();
    return;
  }
  shell(renderOnboarding());
  bindOnboarding();
}

// KeyboardEvent.code -> supported hotkey token. Only these eight modifier
// keys are valid (see SUPPORTED_HOTKEYS in app_settings_model.py /
// onboarding_flow.py / hotkey_monitor.py); anything else is "unsupported".
const HOTKEY_CODE_TO_TOKEN = {
  MetaRight: "right_cmd",
  MetaLeft: "left_cmd",
  ShiftRight: "right_shift",
  ShiftLeft: "left_shift",
  AltRight: "right_alt",
  AltLeft: "left_alt",
  ControlRight: "right_ctrl",
  ControlLeft: "left_ctrl",
};

function hotkeyLabel(token) {
  return t(`hotkey_label_${token}`);
}

// Renders a press-to-set hotkey widget: a localized label for the current
// key plus a "Change" button. `id` becomes the container's DOM id so
// settingsPayload()/onboarding's confirm handler can read the captured
// value back out via `.dataset.hotkeyValue`.
function renderHotkeyCapture(id, value) {
  return `
    <div class="hotkey-capture" id="${id}" data-hotkey-capture data-hotkey-value="${escapeHtml(value)}">
      <span class="hotkey-capture-badge" data-hotkey-badge>${escapeHtml(hotkeyLabel(value))}</span>
      <button type="button" class="button ghost" data-hotkey-change>${escapeHtml(t("hotkey_capture_change_button"))}</button>
      <span class="hotkey-capture-message" data-hotkey-message></span>
    </div>
  `;
}

// Tears down whatever hotkey capture session is active, if any. Safe to
// call when nothing is capturing. Called before every re-render (the
// capture widget's DOM is about to be replaced/removed) and on window
// blur, so the native hotkey monitor never stays suppressed longer than
// the capture UI is actually visible.
function endActiveHotkeyCapture() {
  if (hotkeyCapture) {
    hotkeyCapture.cleanup();
  }
}

function bindHotkeyCapture(containerEl) {
  const changeButton = containerEl?.querySelector("[data-hotkey-change]");
  const badge = containerEl?.querySelector("[data-hotkey-badge]");
  const messageEl = containerEl?.querySelector("[data-hotkey-message]");
  if (!containerEl || !changeButton || !badge || !messageEl) {
    return;
  }
  changeButton.addEventListener("click", () => {
    beginHotkeyCaptureUi(containerEl, badge, messageEl, changeButton);
  });
}

function beginHotkeyCaptureUi(containerEl, badge, messageEl, changeButton) {
  endActiveHotkeyCapture();
  changeButton.disabled = true;
  messageEl.textContent = t("hotkey_capture_prompt");
  bridge("beginHotkeyCapture").catch(() => {});

  const onKeydown = (event) => {
    event.preventDefault();
    if (event.key === "Escape") {
      finish();
      return;
    }
    const token = HOTKEY_CODE_TO_TOKEN[event.code];
    if (!token) {
      messageEl.textContent = t("hotkey_capture_unsupported");
      return;
    }
    containerEl.dataset.hotkeyValue = token;
    if (containerEl.id === "hotkey") {
      settingsDraft.hotkey = token;
      updateSaveFeedback("settings");
    }
    badge.textContent = hotkeyLabel(token);
    finish();
  };
  const onBlur = () => finish();

  function finish() {
    window.removeEventListener("keydown", onKeydown, true);
    window.removeEventListener("blur", onBlur);
    changeButton.disabled = false;
    messageEl.textContent = "";
    bridge("endHotkeyCapture").catch(() => {});
    if (hotkeyCapture && hotkeyCapture.cleanup === finish) {
      hotkeyCapture = null;
    }
  }

  window.addEventListener("keydown", onKeydown, true);
  window.addEventListener("blur", onBlur);
  hotkeyCapture = { cleanup: finish };
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
          <p class="error" role="alert"></p>
        </div>
      </section>
    `;
  }
  if (current === "hotkey") {
    const hotkey = state.settings?.hotkey || "right_cmd";
    return `
      <section class="onboarding-wrap">
        <div class="card">
          <div class="steps">${dots}</div>
          <h1>${escapeHtml(t("hotkey_confirm_title"))}</h1>
          <p>${escapeHtml(t("hotkey_confirm_body"))}</p>
          <div class="form-row">
            <label>${escapeHtml(t("hotkey_select_label"))}</label>
            ${renderHotkeyCapture("onboarding-hotkey", hotkey)}
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
          ${(state.settings?.model || state.model || "").startsWith("haas:") ? renderSpeechCpp() : ""}
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
  bindSpeechCppButtons();
  app.querySelectorAll("[data-language]").forEach((button) => {
    button.addEventListener("click", async () => {
      const result = await bridge("chooseLanguage", { code: button.dataset.language }).catch((error) => {
        showError(error);
        return null;
      });
      if (!result) {
        return;
      }
      if (result.saved === false) {
        const target = app.querySelector(".error");
        if (target) {
          target.textContent = result.message || settingsValidationText(result.errors);
        }
        return;
      }
      state = await bridge("getState");
      render();
    });
  });
  app.querySelectorAll("[data-hotkey-capture]").forEach((el) => bindHotkeyCapture(el));
  app.querySelector("[data-action='confirm-hotkey']")?.addEventListener("click", async () => {
    const selectedHotkey =
      document.getElementById("onboarding-hotkey")?.dataset.hotkeyValue || state.settings?.hotkey;
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

function settingsValueEqual(left, right) {
  return Array.isArray(left) && Array.isArray(right)
    ? left.length === right.length && left.every((value, index) => value === right[index])
    : left === right;
}

function parseGlossary(text) {
  return [...new Set(text.split(/\r\n?|\n/).map(term => term.trim()).filter(Boolean))];
}

function glossaryText(settings) {
  return glossaryDraftText ?? (settings.llm_correction?.glossary || []).join("\n");
}

function settingsValues() {
  const settings = { ...state.settings, llm_correction: { ...state.settings?.llm_correction } };
  for (const [key, value] of Object.entries(settingsDraft)) {
    const target = key.startsWith("llm_") ? settings.llm_correction : settings;
    const field = key.startsWith("llm_") ? key.slice(4) : key;
    // Only an explicit save acknowledges edits. Matching native snapshots
    // may be followed by stale pushes or another screen's save response.
    target[field] = value;
  }
  return settings;
}

// Apply the new snapshot plus edited-field overrides without remounting the form.
function updateSettingsInputsInPlace() {
  const settings = settingsValues();
  app.querySelectorAll("select[id], input[id]").forEach((input) => {
    const target = input.id.startsWith("llm_") ? settings.llm_correction : settings;
    const field = input.id.startsWith("llm_") ? input.id.slice(4) : input.id;
    input.value = target[field] ?? "";
  });
  const hotkey = document.getElementById("hotkey");
  if (hotkey) {
    hotkey.dataset.hotkeyValue = settings.hotkey;
    const badge = hotkey.querySelector("[data-hotkey-badge]");
    if (badge) badge.textContent = hotkeyLabel(settings.hotkey);
  }
}

function renderSettings() {
  const settings = settingsValues();
  const llm = settings.llm_correction || {};
  const models = state.models || [];
  return `
    <section class="grid">
      ${saveScreenHeader("settings", "settings_window_title")}
      <div class="panel full">
        <div class="panel-header">
          <h2>${escapeHtml(t("settings_models_section_title"))}</h2>
        </div>
        <div class="model-list">
          ${models.map((model) => renderModelCard(model, settings.model)).join("")}
        </div>
      </div>
      <div class="panel">
        ${selectRow(
          "language",
          "settings_language_label",
          settings.language,
          [
            ["en", t("language_english")],
            ["ja", t("language_japanese")],
            ["zh", t("language_chinese")],
            ...(settings.model?.startsWith("haas:") || settings.language === "auto" ? [["auto", t("language_auto")]] : []),
          ],
          "settings_language_help",
        )}
        <div class="form-row">
          <label>${escapeHtml(t("settings_hotkey_label"))}</label>
          ${renderHotkeyCapture("hotkey", settings.hotkey)}
          <p class="form-help">${escapeHtml(t("settings_hotkey_help"))}</p>
        </div>
        ${selectRow(
          "output_mode",
          "settings_output_mode_label",
          settings.output_mode,
          [
            ["direct_typing", t("output_direct_typing")],
            ["clipboard_paste", t("output_clipboard_paste")],
          ],
          "settings_output_mode_help",
        )}
        <div class="form-row">
          <button class="button" data-action="open-config">${escapeHtml(t("settings_open_config_button"))}</button>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <h2>${escapeHtml(t("settings_llm_section_title"))}</h2>
        </div>
        ${selectRow("llm_mode", "settings_llm_mode_label", llm.mode, [
          ["always", t("settings_llm_mode_always")],
          ["never", t("settings_llm_mode_never")],
          ["ask", t("settings_llm_mode_ask")],
        ])}
        ${inputRow("llm_provider", "settings_llm_provider_label", llm.provider)}
        ${inputRow("llm_model", "settings_llm_model_label", llm.model)}
        ${inputRow("llm_base_url", "settings_llm_base_url_label", llm.base_url)}
      </div>
    </section>
  `;
}

function renderModelCard(model, selected) {
  const isSelected = selected === model.token;
  const selectedClass = isSelected ? "selected" : "";
  const download = downloadStates.get(model.token);
  // Once the cache confirms the model is downloaded, drop a stale "done"
  // entry so a later cache change (e.g. manual deletion) doesn't leave a
  // phantom "Download complete" badge behind.
  if (model.downloaded && download?.status === "done") {
    downloadStates.delete(model.token);
  }
  const status = model.downloaded ? "downloaded" : download?.status || "idle";
  return `
    <div class="model-card ${selectedClass}" data-select-model="${escapeHtml(model.token)}" tabindex="0" role="button">
      <div class="model-info">
        <h3>${escapeHtml(model.label)}</h3>
        <div class="model-meta">${escapeHtml(model.description)}</div>
        <div class="model-token">${escapeHtml(model.token)}</div>
      </div>
      <div class="model-action">
        <div role="status" aria-live="polite" aria-atomic="true">${renderModelStatus(status, download)}</div>
        ${renderModelAction(model, status)}
      </div>
      ${renderModelProgress(status, download)}
      ${model.backend === "haas" || model.token?.startsWith("haas:") ? renderSpeechCpp() : ""}
    </div>
  `;
}

function acceptSpeechCppStatus(incoming, snapshot = false) {
  if (!incoming || !Number.isInteger(incoming.operation_id)) return;
  const previous = speechCppStatus;
  if (previous && incoming.operation_id < previous.operation_id) return;
  // A start response/getState snapshot can lag already delivered native progress.
  if (snapshot && previous && incoming.operation_id === previous.operation_id &&
      ["checking", "installing"].includes(incoming.status)) return;
  // A delayed snapshot/progress event cannot roll a finished operation back.
  if (previous && incoming.operation_id === previous.operation_id &&
      !["checking", "installing"].includes(previous.status)) return;
  speechCppStatus = { ...incoming };
}

function currentSpeechCppStatus() {
  const current = speechCppStatus || state?.speech_cpp || { status: "checking" };
  return speechCppRequestPending && (current.operation_id || 0) <= speechCppRequestOperationId &&
    !["installing", "checking"].includes(current.status)
    ? { ...current, status: "installing", phase: "downloading", downloaded_bytes: 0, total_bytes: null }
    : current;
}

function speechCppStatusText(runtime) {
  const key = runtime.status === "installing" ? runtime.phase || "installing" : runtime.status;
  let text = t(`speech_cpp_${key}`);
  if (runtime.status === "available") {
    text += runtime.version ? ` (${runtime.version})` : "";
  }
  if (runtime.source === "override" || runtime.status === "override_invalid") {
    text += ` ${t("speech_cpp_override_guidance")}`;
  }
  if (runtime.status === "failed") text += ` ${t("speech_cpp_failure_guidance")}`;
  if (runtime.install_supported === false && ["old", "failed"].includes(runtime.status)) {
    text += ` ${t("speech_cpp_unsupported")}`;
  }
  if (runtime.message && ["failed", "old", "override_invalid"].includes(runtime.status)) {
    text += ` ${runtime.message}`;
  }
  return text;
}

function renderSpeechCppProgress(runtime) {
  if (runtime.status !== "installing" && runtime.status !== "checking") return "";
  const total = Number(runtime.total_bytes);
  const known = runtime.status === "installing" && runtime.phase === "downloading" &&
    Number.isFinite(total) && total > 0 && Number.isFinite(Number(runtime.downloaded_bytes));
  const percent = known ? fractionToPercent(Number(runtime.downloaded_bytes) / total) : null;
  const label = escapeHtml(speechCppStatusText(runtime));
  return `<div class="progress${known ? "" : " indeterminate"}" role="progressbar" aria-label="${label}"${known ? ` aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}"` : ""}><span${known ? ` style="--value: ${percent}%"` : ""}></span></div>`;
}

function renderSpeechCppAction(runtime) {
  if (speechCppRequestPending || !runtime.install_allowed ||
      !["missing", "old", "failed"].includes(runtime.status)) return "";
  const key = runtime.status === "failed" ? "speech_cpp_retry" : "speech_cpp_install";
  return `<button type="button" class="button" data-install-speech-cpp>${escapeHtml(t(key))}</button>`;
}

function renderSpeechCpp() {
  const runtime = currentSpeechCppStatus();
  return `<div class="model-info" style="grid-column: 1 / -1" tabindex="-1" data-speech-cpp>
    <div class="model-meta">${escapeHtml(t("speech_cpp_title"))}</div>
    <div role="status" aria-live="polite" aria-atomic="true" data-speech-cpp-status>${escapeHtml(speechCppStatusText(runtime))}</div>
    <div data-speech-cpp-progress>${renderSpeechCppProgress(runtime)}</div>
    <div data-speech-cpp-action>${renderSpeechCppAction(runtime)}</div>
  </div>`;
}

function bindSpeechCppButtons() {
  app.querySelectorAll("[data-install-speech-cpp]").forEach(button => {
    button.addEventListener("click", handleInstallSpeechCppClick);
  });
}

function updateSpeechCppInPlace() {
  const runtime = currentSpeechCppStatus();
  app.querySelectorAll("[data-speech-cpp]").forEach(container => {
    // Keep the live region and the rest of the form mounted during progress.
    container.querySelector("[data-speech-cpp-status]").textContent = speechCppStatusText(runtime);
    const progress = container.querySelector("[data-speech-cpp-progress]");
    const nextProgress = renderSpeechCppProgress(runtime);
    if (progress.innerHTML !== nextProgress) progress.innerHTML = nextProgress;
    const action = container.querySelector("[data-speech-cpp-action]");
    const nextAction = renderSpeechCppAction(runtime);
    if (action.innerHTML !== nextAction) {
      const hadFocus = action.contains(document.activeElement);
      action.innerHTML = nextAction;
      action.querySelector("[data-install-speech-cpp]")?.addEventListener("click", handleInstallSpeechCppClick);
      if (hadFocus) (container.closest("[data-select-model]") || container).focus();
    }
  });
}

async function handleInstallSpeechCppClick(event) {
  event.stopPropagation();
  const runtime = currentSpeechCppStatus();
  if (speechCppRequestPending || !runtime.install_allowed ||
      !["missing", "old", "failed"].includes(runtime.status)) return;
  const operationId = runtime.operation_id || 0;
  speechCppRequestPending = true;
  speechCppRequestOperationId = operationId;
  updateSpeechCppInPlace();
  try {
    const result = await bridge("installSpeechCpp");
    if (result?.speech_cpp) acceptSpeechCppStatus(result.speech_cpp, true);
    if (!result?.started && !result?.speech_cpp &&
        (speechCppStatus?.operation_id || 0) <= operationId) {
      speechCppStatus = { ...runtime, status: "failed", message: t("speech_cpp_start_failed") };
    }
  } catch (error) {
    // A late bridge rejection cannot overwrite native progress/completion.
    if ((speechCppStatus?.operation_id || 0) <= operationId) {
      speechCppStatus = { ...runtime, status: "failed", message: error.message };
    }
  } finally {
    speechCppRequestPending = false;
    updateSpeechCppInPlace();
  }
}

function renderModelStatus(status, download) {
  if (status === "downloaded") {
    return `<span class="badge">${escapeHtml(t("settings_model_downloaded_badge"))}</span>`;
  }
  if (status === "done") {
    return `<span class="badge">${escapeHtml(t("settings_model_download_done_message"))}</span>${download?.message ? `<span class="model-error-text">${escapeHtml(download.message)}</span>` : ""}`;
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
    return `<span class="model-error-text">${escapeHtml(downloadErrorText(download?.message))}</span>`;
  }
  if (status === "busy") {
    return `<span class="model-busy-text">${escapeHtml(t("settings_model_busy_message"))}</span>`;
  }
  return "";
}

function renderModelAction(model, status) {
  if (!["idle", "error", "busy"].includes(status)) return "";
  const disabled = activeDownloadToken ? "disabled" : "";
  const label = status === "idle" ? "settings_model_download_button" : "settings_model_retry_button";
  return `<button class="button" data-download-model="${escapeHtml(model.token)}" ${disabled}>${escapeHtml(t(label))}</button>`;
}

function renderModelProgress(status, download) {
  if (status === "preparing") {
    return `
      <div class="model-progress-row">
        <div class="progress indeterminate" role="progressbar" aria-label="${escapeHtml(t("settings_model_preparing_message"))}"><span></span></div>
      </div>
    `;
  }
  if (status === "downloading") {
    const percent = fractionToPercent(download?.fraction);
    return `
      <div class="model-progress-row">
        <div class="progress" role="progressbar" aria-label="${escapeHtml(t("settings_model_download_button"))}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}"><span style="--value: ${percent}%"></span></div>
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

function updateModelCardsInPlace() {
  for (const model of state?.models || []) updateModelCardInPlace(model.token);
  updateSpeechCppInPlace();
}

async function refreshDownloadedModels(token) {
  try {
    const refreshed = await bridge("getState");
    // Only synchronize model cache flags, not persisted settings/selection.
    state.models = refreshed.models;
    updateModelCardsInPlace();
  } catch {
    // Keep the truthful done state even if the cache snapshot cannot be read.
    const download = downloadStates.get(token);
    if (download?.status === "done") {
      download.message = t("settings_model_refresh_failed_message");
      updateModelCardInPlace(token);
    }
  }
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
  // Keep the selection target and live status region mounted. Only replace
  // changed card contents; unrelated cards and the settings form stay intact.
  const template = document.createElement("template");
  const selected = settingsValues().model;
  template.innerHTML = renderModelCard(model, selected);
  const nextCard = template.content.firstElementChild;
  cardEl.classList.toggle("selected", selected === token);
  const statusEl = cardEl.querySelector('[role="status"]');
  const nextStatus = nextCard.querySelector('[role="status"]');
  if (statusEl.innerHTML !== nextStatus.innerHTML) statusEl.innerHTML = nextStatus.innerHTML;
  const button = cardEl.querySelector("[data-download-model]");
  const nextButton = nextCard.querySelector("[data-download-model]");
  if (button && nextButton) {
    button.disabled = nextButton.disabled;
    button.textContent = nextButton.textContent;
    if (button.disabled && document.activeElement === button) cardEl.focus();
  } else if (button) {
    const hadFocus = document.activeElement === button;
    button.remove();
    if (hadFocus) cardEl.focus();
  } else if (nextButton) {
    cardEl.querySelector(".model-action").append(nextButton);
    nextButton.addEventListener("click", handleDownloadModelClick);
  }
  const progress = cardEl.querySelector(".model-progress-row");
  const nextProgress = nextCard.querySelector(".model-progress-row");
  if (progress && nextProgress) {
    if (!progress.isEqualNode(nextProgress)) progress.replaceWith(nextProgress);
  } else if (progress) {
    progress.remove();
  } else if (nextProgress) {
    cardEl.append(nextProgress);
  }
}

// `button` here is the whole `.model-card` element (selection target), named
// to match the click-to-select convention used across this file.
function bindModelCard(button) {
  if (!button) {
    return;
  }
  button.addEventListener("click", () => {
    settingsDraft.model = button.dataset.selectModel;
    updateSaveFeedback("settings");
    updateModelCardsInPlace();
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
      settingsDraft.model = button.dataset.selectModel;
      updateSaveFeedback("settings");
      updateModelCardsInPlace();
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
  if (!token || activeDownloadToken) {
    return;
  }
  activeDownloadToken = token;
  downloadStates.set(token, { status: "preparing" });
  updateModelCardsInPlace();
  let result = null;
  try {
    result = await bridge("downloadModel", { model: token });
  } catch (error) {
    if (activeDownloadToken === token) activeDownloadToken = null;
    downloadStates.set(token, { status: "error", message: error.message });
    updateModelCardsInPlace();
    return;
  }
  if (result?.started === false) {
    if (activeDownloadToken === token) activeDownloadToken = null;
    if (result.already_downloaded) {
      downloadStates.set(token, { status: "done" });
      updateModelCardsInPlace();
      await refreshDownloadedModels(token);
      return;
    }
    // Busy is contention, not a failed download. Unknown reason tokens still
    // use the existing localized generic failure copy.
    const status = result.errors?.includes("busy") ? "busy" : "error";
    downloadStates.set(token, { status, message: null });
    updateModelCardsInPlace();
    return;
  }
  // result.started === true: leave status "preparing"; downloadProgress
  // events drive the rest of the state machine from here.
}

function selectRow(id, labelKey, selected, options, helpKey = null) {
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
      ${helpKey ? `<p class="form-help">${escapeHtml(t(helpKey))}</p>` : ""}
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
  const settings = settingsValues();
  return {
    model: modelOverride || settings.model,
    language: document.getElementById("language")?.value ?? settings.language,
    hotkey: document.getElementById("hotkey")?.dataset.hotkeyValue ?? settings.hotkey,
    output_mode: document.getElementById("output_mode")?.value ?? settings.output_mode,
    llm_correction: {
      mode: document.getElementById("llm_mode")?.value ?? settings.llm_correction?.mode,
      provider: document.getElementById("llm_provider")?.value ?? settings.llm_correction?.provider,
      model: document.getElementById("llm_model")?.value ?? settings.llm_correction?.model,
      base_url: document.getElementById("llm_base_url")?.value ?? settings.llm_correction?.base_url,
    },
  };
}

function bindSettings() {
  bindSpeechCppButtons();
  app.querySelectorAll("select[id], input[id]").forEach((input) => {
    const remember = () => {
      settingsDraft[input.id] = input.value;
      updateSaveFeedback("settings");
    };
    input.addEventListener("input", remember);
    input.addEventListener("change", remember);
  });
  app.querySelector("[data-action='save-settings']")?.addEventListener("click", async () => {
    await saveSettings();
  });
  app.querySelector("[data-action='open-config']")?.addEventListener("click", () => {
    bridge("openConfigFile").catch((error) => showError(error, "settings-error"));
  });
  app.querySelectorAll("[data-select-model]").forEach((cardEl) => {
    bindModelCard(cardEl);
  });
  app.querySelectorAll("[data-hotkey-capture]").forEach((el) => bindHotkeyCapture(el));
}

function saveButtonLabel(saving, message, labelKey) {
  return t(saving ? "save_in_progress" : message?.kind === "error" ? "save_retry_button" : labelKey);
}

function saveMessageBox(id, message, saving) {
  const kind = saving ? "pending" : message?.kind;
  const text = saving ? t("save_in_progress") : message?.text || "";
  return `<div id="${id}" class="save-status${kind ? ` alert ${kind}` : ""}" role="status" aria-live="polite" aria-atomic="true">${escapeHtml(text)}</div>`;
}

function saveState(scope) {
  if (scope === "settings") return { saving: settingsSaving, message: settingsMessage };
  if (scope === "glossary") return { saving: glossarySaving, message: glossaryMessage };
  return { saving: dictionarySaving, message: dictionaryMessage };
}

function settingsHaveChanges() {
  const current = settingsValues();
  const saved = state.settings;
  return ["model", "language", "hotkey", "output_mode"].some(key => current[key] !== saved[key]) ||
    ["mode", "provider", "model", "base_url"].some(key =>
      String(current.llm_correction?.[key] ?? "").trim() !==
      String(saved.llm_correction?.[key] ?? "").trim());
}

function dictionaryContent(draft = null) {
  return JSON.stringify(["exact", "regex"].map(section => {
    const content = draft ? draft[section].map(row =>
      [row.key.trim(), parseDictionaryValues(section, row.valuesText)])
      .filter(([key, values]) => key || values.length) :
      Object.entries(state.dictionary?.[section] || {});
    // Persisted regex whitespace affects matching; exact rules ignore padding.
    // Regex rule order can change matching precedence. Exact rules are a lookup.
    return section === "regex" ? content :
      content.map(([key, values]) =>
        [key.trim(), values.map(value => value.trim()).filter(Boolean)])
        .sort((left, right) => left[0] < right[0] ? -1 : left[0] > right[0] ? 1 : 0);
  }));
}

function saveHasChanges(scope) {
  if (scope === "settings") return settingsHaveChanges();
  if (scope === "glossary") {
    return !settingsValueEqual(parseGlossary(glossaryText(state.settings)),
      parseGlossary((state.settings?.llm_correction?.glossary || []).join("\n")));
  }
  if (!dictDraft) return false;
  dictDraft.dirty = dictionaryContent(dictDraft) !== dictionaryContent();
  return dictDraft.dirty;
}

function saveButtonClass(changed, saving) {
  return `button primary save-button${changed && !saving ? " has-changes" : ""}`;
}

function saveScreenHeader(scope, titleKey) {
  const { saving, message } = saveState(scope);
  const changed = saveHasChanges(scope);
  return `<header class="panel full screen-save-header" data-save-header="${scope}">
    <h2>${escapeHtml(t(titleKey))}</h2>
    <div class="screen-save-actions">
      <span id="${scope}-unsaved" class="unsaved-indicator${changed && !saving ? "" : " hidden"}">${escapeHtml(t("dictionary_unsaved_changes_label"))}</span>
      <button class="${saveButtonClass(changed, saving)}" data-action="save-${scope}" aria-describedby="${changed && !saving ? `${scope}-unsaved` : ""}" aria-busy="${saving}" ${saving ? "disabled" : ""}>${escapeHtml(saveButtonLabel(saving, message, `${scope}_save_button`))}</button>
    </div>
    ${saveMessageBox(`${scope}-error`, message, saving)}
  </header>`;
}

// Patch feedback immediately without rebuilding the settings form. Rendering
// after a push or navigation uses the same pending flags and keeps Save locked.
function updateSaveFeedback(scope) {
  const { saving, message } = saveState(scope);
  const changed = saveHasChanges(scope);
  const target = document.getElementById(`${scope}-error`);
  if (target) {
    const kind = saving ? "pending" : message?.kind;
    target.className = `save-status${kind ? ` alert ${kind}` : ""}`;
    target.textContent = saving ? t("save_in_progress") : message?.text || "";
  }
  const button = app.querySelector(`[data-action='save-${scope}']`);
  if (button) {
    button.className = saveButtonClass(changed, saving);
    button.disabled = saving;
    button.setAttribute("aria-busy", String(saving));
    button.setAttribute("aria-describedby", changed && !saving ? `${scope}-unsaved` : "");
    button.textContent = saveButtonLabel(saving, message, `${scope}_save_button`);
  }
  const indicator = document.getElementById(`${scope}-unsaved`);
  if (indicator) indicator.className = `unsaved-indicator${changed && !saving ? "" : " hidden"}`;
}

function saveFailureText(scope, error, refreshing = false) {
  const key = refreshing ? "save_refresh_failed_message" : `${scope}_save_failed_message`;
  const detail = error?.message || t("save_error_unknown");
  return `${t(key).replace("{error}", detail)} ${t("save_retry_hint")}`;
}

function settingsValidationText(errors) {
  const labels = {
    model: "settings_model_label",
    language: "settings_language_label",
    hotkey: "settings_hotkey_label",
    output_mode: "settings_output_mode_label",
    "llm_correction.mode": "settings_llm_mode_label",
    "llm_correction.provider": "settings_llm_provider_label",
    "llm_correction.model": "settings_llm_model_label",
    "llm_correction.base_url": "settings_llm_base_url_label",
    "llm_correction.glossary": "settings_llm_glossary_label",
  };
  const fields = [...new Set((errors || []).map((field) => {
    const label = t(labels[field] || "settings_window_title");
    return field.startsWith("llm_correction.") && labels[field]
      ? `${t("settings_llm_section_title")} / ${label}` : label;
  }))];
  return `${t("settings_validation_error").replace("{fields}", fields.join(", ") || t("settings_window_title"))} ${t("save_retry_hint")}`;
}

async function saveSettings(modelOverride = null) {
  if (settingsSaving) {
    return;
  }
  const payload = settingsPayload(modelOverride);
  const submittedDraft = { ...settingsDraft };
  settingsSaving = true;
  settingsMessage = null;
  updateSaveFeedback("settings");
  let refreshing = false;
  try {
    const result = await writeSettings(payload);
    if (result.saved === false) {
      settingsMessage = { kind: "error", text: [settingsValidationText(result.errors), result.message].filter(Boolean).join(" ") };
      return;
    }
    refreshing = true;
    const refreshed = await bridge("getState");
    // Keep drafts through the refresh too: pushes and failed refreshes must
    // not expose old persisted values or discard text needed for a retry.
    for (const [key, value] of Object.entries(submittedDraft)) {
      const saved = result.settings || refreshed.settings;
      const target = key.startsWith("llm_") ? saved?.llm_correction : saved;
      const field = key.startsWith("llm_") ? key.slice(4) : key;
      // The saved value may be normalized. Acknowledge only the sent edit,
      // not a newer draft entered while the save request was in flight.
      if (target?.[field] !== undefined && settingsValueEqual(settingsDraft[key], value)) {
        delete settingsDraft[key];
      }
    }
    mergeSavedSettings(result.settings || refreshed.settings, "settings");
    state.strings = refreshed.strings || state.strings;
    settingsMessage = { kind: "success", text: t("settings_saved_message") };
    render();
  } catch (error) {
    settingsMessage = { kind: "error", text: saveFailureText("settings", error, refreshing) };
  } finally {
    settingsSaving = false;
    updateSaveFeedback("settings");
  }
}

function writeSettings(payload) {
  return new Promise((resolve, reject) => {
    settingsWrites.push({ payload, resolve, reject });
    void drainSettingsWrites();
  });
}

async function drainSettingsWrites() {
  if (settingsWriteBusy || !settingsWrites.length) return;
  settingsWriteBusy = true;
  const request = settingsWrites.shift();
  try {
    request.resolve(await bridge("saveSettings", request.payload));
  } catch (error) {
    request.reject(error);
  } finally {
    settingsWriteBusy = false;
    void drainSettingsWrites();
  }
}

// Save replies contain complete settings and can arrive out of order. Merge
// only the owning screen's fields so neither save rolls the other one back.
function mergeSavedSettings(saved, scope) {
  const settings = { ...state.settings, llm_correction: { ...state.settings?.llm_correction } };
  if (scope === "glossary") {
    settings.llm_correction.glossary = [...saved.llm_correction.glossary];
  } else {
    for (const key of ["model", "language", "hotkey", "output_mode"]) {
      settings[key] = saved[key];
    }
    for (const key of ["mode", "provider", "model", "base_url"]) {
      settings.llm_correction[key] = saved.llm_correction[key];
    }
  }
  state.settings = settings;
}

function glossaryPayload() {
  return { llm_correction: {
    glossary: parseGlossary(document.getElementById("llm_glossary")?.value ?? glossaryText(state.settings)),
  } };
}

function renderGlossary() {
  return `
    <section class="grid">
      ${saveScreenHeader("glossary", "route_glossary")}
      <div class="panel full">
        <div class="form-row">
          <label for="llm_glossary">${escapeHtml(t("settings_llm_glossary_label"))}</label>
          <p id="llm_glossary_help" class="form-help">${escapeHtml(t("settings_llm_glossary_help"))}</p>
          <textarea id="llm_glossary" class="glossary-editor" rows="16" aria-describedby="llm_glossary_help">
${escapeHtml(glossaryText(state.settings))}</textarea>
        </div>
      </div>
    </section>
  `;
}

function bindGlossary() {
  const input = document.getElementById("llm_glossary");
  const remember = () => {
    if (glossaryDraftText !== input.value) {
      glossaryDraftText = input.value;
      glossaryDraftRevision++;
    }
    updateSaveFeedback("glossary");
  };
  input.addEventListener("input", remember);
  input.addEventListener("change", remember);
  app.querySelector("[data-action='save-glossary']")?.addEventListener("click", saveGlossary);
}

async function saveGlossary() {
  if (glossarySaving) return;
  const payload = glossaryPayload();
  const submittedRevision = glossaryDraftRevision;
  glossarySaving = true;
  glossaryMessage = null;
  updateSaveFeedback("glossary");
  let refreshing = false;
  try {
    const result = await writeSettings(payload);
    if (result.saved === false) {
      glossaryMessage = { kind: "error", text: [settingsValidationText(result.errors), result.message].filter(Boolean).join(" ") };
      return;
    }
    refreshing = true;
    const refreshed = await bridge("getState");
    const saved = result.settings || refreshed.settings;
    mergeSavedSettings(saved, "glossary");
    // Server normalization can differ from JavaScript trim. Adopt the saved
    // value unless an actual edit occurred after this submission, including
    // an edit away from and back to the original text.
    if (glossaryDraftRevision === submittedRevision) {
      glossaryDraftText = null;
    }
    glossaryMessage = { kind: "success", text: t("glossary_saved_message") };
    render();
  } catch (error) {
    glossaryMessage = { kind: "error", text: saveFailureText("glossary", error, refreshing) };
  } finally {
    glossarySaving = false;
    updateSaveFeedback("glossary");
  }
}

function renderDictionary() {
  // Untouched drafts follow native snapshots. Keep edited raw rows even when
  // their normalized content matches saved data, so tab switches lose no work.
  if (!dictDraft || (!dictDraft.edited && !dictDraft.dirty && !dictionarySaving)) {
    dictDraft = buildDictDraft(state.dictionary);
  }
  return `
    <section class="grid">
      ${saveScreenHeader("dictionary", "dictionary_editor_title")}
      <div class="panel full">
        ${dictionarySection("exact", dictDraft.exact, "dictionary_exact_rules_title", "dictionary_exact_rules_description", "dictionary_add_exact_button")}
        ${dictionarySection("regex", dictDraft.regex, "dictionary_regex_rules_title", "dictionary_regex_rules_description", "dictionary_add_regex_button")}
        <div class="actions">
          <button class="button" data-add-dictionary="exact">${escapeHtml(t("dictionary_add_exact_button"))}</button>
          <button class="button" data-add-dictionary="regex">${escapeHtml(t("dictionary_add_regex_button"))}</button>
        </div>
      </div>
    </section>
  `;
}

function buildDictDraft(dictionary) {
  const source = dictionary || { exact: {}, regex: {} };
  return {
    exact: Object.entries(source.exact || {}).map(([key, values]) =>
      makeDictRow("exact", key, values),
    ),
    regex: Object.entries(source.regex || {}).map(([key, values]) =>
      makeDictRow("regex", key, values),
    ),
    dirty: false,
    edited: false,
    revision: 0,
  };
}

// Exact variants are short comma-friendly tokens, so they display comma-joined
// and split on commas or newlines. Regex patterns may contain literal commas
// (e.g. \d{1,2}), so they display and parse one pattern per line only.
function makeDictRow(section, key, values) {
  const joiner = section === "regex" ? "\n" : ", ";
  return { id: dictRowSeq++, key, valuesText: (values || []).join(joiner), error: null };
}

function dictionarySection(section, rows, titleKey, descriptionKey, addLabelKey) {
  const body = rows.length
    ? rows.map((row) => dictionaryRow(section, row)).join("")
    : dictionaryEmptyState(section, addLabelKey);
  return `
    <div class="dictionary-list" data-section="${section}">
      <div class="dictionary-section-header">
        <h3>${escapeHtml(t(titleKey))}</h3>
        <p class="dictionary-section-description">${escapeHtml(t(descriptionKey))}</p>
      </div>
      ${body}
    </div>
  `;
}

// The section description already lives next to the h3 above; the empty
// state only needs the "no rules yet" message plus an inline way to add one.
function dictionaryEmptyState(section, addLabelKey) {
  return `
    <div class="empty-state">
      <p>${escapeHtml(t("dictionary_no_rules"))}</p>
      <button class="button ghost" data-add-dictionary="${section}">${escapeHtml(t(addLabelKey))}</button>
    </div>
  `;
}

function dictionaryRow(section, row) {
  const invalidClass = row.error ? " invalid" : "";
  const keyId = `dict-key-${row.id}`;
  const valuesId = `dict-values-${row.id}`;
  const valuesLabelKey =
    section === "regex" ? "dictionary_patterns_label" : "dictionary_candidates_patterns_label";
  return `
    <div class="dictionary-row${invalidClass}" data-dictionary-row="${row.id}" data-section="${section}">
      <div class="dictionary-row-header">
        <div class="dictionary-field">
          <label for="${keyId}">${escapeHtml(t("dictionary_canonical_label"))}</label>
          <input id="${keyId}" data-dictionary-key value="${escapeHtml(row.key)}" />
        </div>
        <button class="button ghost danger" data-delete-row>${escapeHtml(t("dictionary_delete_button"))}</button>
      </div>
      <div class="dictionary-field">
        <label for="${valuesId}">${escapeHtml(t(valuesLabelKey))}</label>
        <textarea id="${valuesId}" data-dictionary-values>${escapeHtml(row.valuesText)}</textarea>
      </div>
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
  updateSaveFeedback("dictionary");
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
  const keyChanged = Object.prototype.hasOwnProperty.call(patch, "key");
  const changed = Object.entries(patch).some(([key, value]) => row[key] !== value);
  Object.assign(row, patch);
  const hadError = Boolean(row.error);
  row.error = null;
  dictDraft.edited = true;
  if (changed) dictDraft.revision++;
  clearDictionaryMessage();
  if (hadError) {
    clearDictRowErrorInDom(id);
  }
  if (keyChanged) {
    // Editing this row's key may resolve a duplicate-key collision on other
    // rows (e.g. renaming one of two rows both keyed "foo"). Recompute the
    // per-section duplicate set and drop any sibling's now-stale error.
    reconcileDuplicateKeyErrors(section);
  }
  updateSaveFeedback("dictionary");
}

function clearDictRowErrorInDom(id) {
  const rowEl = app.querySelector(`[data-dictionary-row="${id}"]`);
  rowEl?.classList.remove("invalid");
  rowEl?.querySelector(".field-error")?.remove();
}

// Rows with both an empty key and no values are treated as blank/ignorable
// throughout the editor (see buildDictionaryPayload); mirror that here so
// duplicate-key grouping stays consistent between live edits and save-time
// validation. Splitting is section-aware: see the makeDictRow comment.
function parseDictionaryValues(section, valuesText) {
  const splitter = section === "regex" ? /\n/ : /[,\n]/;
  return valuesText
    .split(splitter)
    .map((item) => item.trim())
    .filter(Boolean);
}

function reconcileDuplicateKeyErrors(section) {
  const rows = dictDraft?.[section] || [];
  const counts = new Map();
  rows.forEach((row) => {
    const key = row.key.trim();
    const values = parseDictionaryValues(section, row.valuesText);
    if (!key && values.length === 0) {
      return;
    }
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  rows.forEach((row) => {
    if (!row.error) {
      return;
    }
    const key = row.key.trim();
    const duplicateMessage = t("dictionary_duplicate_key_message").replace("{key}", key);
    const stillDuplicate = (counts.get(key) || 0) > 1;
    if (row.error.message === duplicateMessage && !stillDuplicate) {
      row.error = null;
      clearDictRowErrorInDom(row.id);
    }
  });
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
  dictDraft.edited = true;
  dictDraft.revision++;
  saveHasChanges("dictionary");
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
  dictDraft.edited = true;
  dictDraft.revision++;
  saveHasChanges("dictionary");
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
  updateSaveFeedback("dictionary");
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
      const values = parseDictionaryValues(section, row.valuesText);
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
  dictionaryMessage = {
    kind: "error",
    text: `${t("dictionary_save_failed_message").replace("{error}", first ? composeDictionaryErrorMessage(first) : t("save_error_unknown"))} ${t("save_retry_hint")}`,
  };
}

async function saveDictionary() {
  if (!dictDraft || dictionarySaving) {
    return;
  }
  const { payload, hasDuplicates } = buildDictionaryPayload();
  const submittedRevision = dictDraft.revision;
  if (hasDuplicates) {
    const first = [...dictDraft.exact, ...dictDraft.regex].find((row) => row.error);
    dictionaryMessage = { kind: "error", text: `${first.error.message} ${t("save_retry_hint")}` };
    render();
    return;
  }
  dictionarySaving = true;
  dictionaryMessage = null;
  updateSaveFeedback("dictionary");
  let refreshing = false;
  try {
    const result = await bridge("saveDictionary", payload);
    if (result.saved === false) {
      if (dictDraft.revision === submittedRevision) {
        applyDictionarySaveErrors(result.errors || []);
        if (route === "dictionary") render();
      } else {
        dictionaryMessage = { kind: "error", text: saveFailureText("dictionary", { message: result.errors?.[0]?.message }) };
      }
      return;
    }
    refreshing = true;
    const refreshedState = await bridge("getState");
    state.dictionary = refreshedState.dictionary;
    const hasNewerEdits = dictDraft.revision !== submittedRevision;
    if (!hasNewerEdits) dictDraft = null;
    dictionaryMessage = { kind: "success", text: t("dictionary_saved_message") };
    // Feedback is patched in finally. Keep newer editor nodes mounted so
    // active focus, selection and caret survive the save/refresh response.
    if (!hasNewerEdits && route === "dictionary") render();
  } catch (error) {
    // Keep the draft on save/refresh failure, including a save which succeeded
    // on disk but could not yet be reflected in the view.
    dictionaryMessage = { kind: "error", text: saveFailureText("dictionary", error, refreshing) };
  } finally {
    dictionarySaving = false;
    updateSaveFeedback("dictionary");
  }
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
    if (targetId === "settings-error") {
      target.className = "save-status alert error";
    }
    target.textContent = text;
  }
}

async function boot() {
  state = await bridge("getState");
  render();
}

boot().catch(showError);
