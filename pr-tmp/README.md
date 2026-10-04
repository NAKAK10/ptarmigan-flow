# Issue #3 save feedback screenshots

Captured before PR creation from `http://127.0.0.1:8763/`, serving this worktree's real `webui` assets and real `WebBridgeDispatcher`. Harness/config/dictionary writes are isolated in `/tmp/ptarmigan-issue-3/`; the shared port 8765 harness was not changed. All displayed rules/errors are synthetic.

Chrome DevTools MCP `wait_for` + `take_snapshot` preceded each full-page `take_screenshot`. Desktop viewports: 1440×1100 (settings), 1440×900 (dictionary); mobile: 390×844. The harness adapter initially rejected filesystem screenshot paths (missing workspace roots), so capture used a separate isolated Chrome DevTools MCP stdio client declaring only this worktree as its root, with its own dedicated pageId. No images were synthesized or edited.

- `settings-pending-en.png`: immediate indeterminate status and disabled/busy Save; two click attempts produced one bridge request. Injected save delay 10 seconds, state-refresh delay 3 seconds.
- `settings-success-en.png`: success only after saving and refreshed state; restart-to-apply note retained.
- `settings-validation-ja-mobile.png`: real dispatcher rejects an injected unsupported LLM mode; localized field label, explanation and Retry save (no internal field key).
- `dictionary-pending-zh.png`: pending dictionary save; Save/add/delete/edit controls locked; two click attempts produced one bridge request.
- `dictionary-refresh-error-zh.png`: save succeeded on disk but injected getState failure prevents success; retained rule and actionable retry.
- `dictionary-success-zh-mobile.png`: successful retry after state reflection. Document width verified 390px; no horizontal overflow. Browser error console empty.

**Native limitation:** this is a mocked browser transport/effect preview, not WKWebView or native responsiveness verification. Permissions, daemon, cache/availability, login, restart, hotkey effects and downloads are mocked/no-op. Real settings/dictionary validation and persistence run only in temporary files. No native builds/launches, TCC changes, permission probes or model downloads were performed. Native synchronous bridge scheduling remains unverified. Settings draft/model-selection preservation and download lifecycle are deliberately out of scope.

Screenshots are review aids only, not referenced by app code. No pr-tmp cleanup workflow exists here; retained in this PR branch.
