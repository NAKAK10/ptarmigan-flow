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

# Issue #4 — model download lifecycle

Safe synthetic data; real worktree webui assets and WebBridgeDispatcher served by an isolated copy of `/tmp/ptarmigan-dogfood` on port **8764** (`/tmp/ptarmigan-issue-4`). Shared port 8765 was not changed.

- `issue-4-desktop-preparing.png`: indeterminate preparation; other downloads disabled.
- `issue-4-desktop-progress.png` / `issue-4-mobile-progress.png`: supplied 42% progress; another model remains selectable; Download/Retry globally disabled.
- `issue-4-desktop-outcomes.png` / `issue-4-mobile-outcomes.png`: downloaded badge after done/getState, distinct backend busy text, ordinary failure with Retry; unsaved language/output/provider inputs retained.

Desktop viewport: 1440×900. Mobile-width viewport: 390×844. All images are full-page, uncropped browser screenshots, not image composites. Chrome DevTools MCP pageId 2 verified/captured these states via snapshots, wait_for and take_screenshot. MCP could return screenshot attachments but rejected filePath writes as outside its configured workspace roots; the committed files were captured by a separate temporary headless Chrome/Puppeteer session reproducing the same flow against the same isolated server (fallback capture). That session additionally asserted live input/card node identity, selection preservation, cache refresh, busy/error unlock and absence of JS errors.

Native downloads/cache availability, permissions/TCC, daemon, audio/input injection, login/relaunch effects are **mocked**. No native build, permission change, real model download or speech recognition was performed. Node lifecycle tests exercise real frontend functions with a stub bridge/DOM and are not a VoiceOver test.

No `pr-tmp` cleanup workflow exists in this repository; these small review-only images are intentionally committed and are not referenced by application code.

# Issue #2 visual verification

- `issue-2-draft-desktop.png`: full settings page at 1440×900 (full-page image 1440×1375).
- `issue-2-draft-mobile.png`: full settings page at 390×844 (full-page image 390×2202).

After editing language, output, LLM fields and the captured hotkey, selecting
Moonshine base and receiving native-state snapshots retains the unsaved values.
The simulated 42% download update also keeps the selected card and draft.
The saved configuration at screenshot time still had the original defaults.
All values are safe demo data; no model or LLM requests were made.

Verified with Chrome DevTools MCP on isolated page 2 at `http://127.0.0.1:8762/`,
using this worktree's real assets and dispatcher with mocked native transport,
permissions, model cache and side effects. MCP full-page screenshots were viewed
but its file-export tool rejected paths outside configured workspace roots.
Therefore the committed PNGs were captured in a separate isolated, headless
Chrome via Playwright, repeating the same real-page edits and asserting all
retained values before full-page capture. No image synthesis or editing.

Native WKWebView, permissions, app build, real downloads and inference were not
tested. The harness writes only to `/tmp/ptarmigan-issue-2-dogfood`, not user config.

## PR #5 review follow-up: normalized save acknowledgement

- `issue-2-normalized-save-desktop.png`: 1440×900 viewport, full-page 1440×1343.
- `issue-2-normalized-save-mobile.png`: 390×844 viewport, full-page 390×2186.

Entered ` openai `, ` demo-model ` and ` http://localhost:12345 `, selected
another model, then saved through the real dispatcher. The UI displayed trimmed
values and the sent drafts were cleared. An external write through the dispatcher
followed by a simulated `daemonState` push updated all three displayed fields to
`external-provider`, `external-model` and `http://localhost:23456` (shown here).
All values are safe demo data; no model or LLM requests were made.

Chrome DevTools MCP verified the real page, payloads and empty draft, and viewed
full-page screenshots at desktop and 390px width. File export was again rejected
by its workspace-root restriction. These PNGs were therefore captured using a
separate isolated headless Chrome/Playwright, repeating model selection, the real
normalized save and external update, with assertions before capture. No image
synthesis or editing. Native transport/effects, permissions and model cache
remain mocked; native WKWebView/TCC, app build, real downloads/inference were not
tested. Only the same isolated `/tmp` configuration was written.

No `pr-tmp` cleanup workflow exists here; these review-only assets are committed
with the PR and are not referenced by application code.

## PR #6 integration with merged PR #5

- `issue-4-integration-desktop-progress.png` / `issue-4-integration-mobile-progress.png`:
  synthetic 42% download, globally disabled Download buttons, draft Moonshine base
  selection, Japanese/provider/empty URL drafts retained after a daemon snapshot.
  Untouched hotkey/output/LLM mode/model reflect external changes in that snapshot.
- `issue-4-integration-desktop-done.png` / `issue-4-integration-mobile-done.png`:
  done/cache refresh and daemon snapshot keep the same selection/values while
  showing Downloaded and unlocking the other downloads.

Desktop viewport 1440×900; mobile-width 390×844; full-page, uncropped PNGs.
Chrome DevTools MCP verified these combined states at isolated port 8764 with
snapshot/wait_for/full-page screenshot and asserted input/card/status node identity.
Its file export was denied by workspace-root restrictions, so the committed PNGs
use a separate headless Chrome/Puppeteer capture repeating the same real-page flow
and asserting draft payload, clean-field updates, lock/unlock and DOM identity.
No image synthesis/editing; no JS errors in either browser session.

Safe synthetic values only; real worktree assets/dispatcher. External configuration
writes are confined to `/tmp/ptarmigan-issue-4`; bridge transport/native effects,
permissions and model cache/download are mocked. No native build, TCC, real model
or LLM request, inference or VoiceOver validation. Shared port 8765 untouched.
Both PRs' earlier evidence/descriptions are retained; no cleanup workflow exists.

## PR #7 integration with merged PR #5 and #6

Latest `origin/dev` (`152bae0`) is integrated without replacing any earlier
screenshots or descriptions. The original issue #3 scope limitation above applies
to its first submission; this follow-up verifies the combined contracts.

- `issue-3-integration-desktop-pending.png` / `issue-3-integration-mobile-pending.png`:
  save held at the mocked transport, immediate busy/disabled Save, synthetic 42%
  download with globally locked Download buttons, draft Moonshine base selection.
  Whitespace-padded LLM values were submitted; `new-unsaved-provider` was entered
  while saving and survives a daemon snapshot without replacing the mounted input.
- `issue-3-integration-desktop-success.png` / `issue-3-integration-mobile-success.png`:
  real dispatcher normalized the submitted values and acknowledged only those
  edits, then `getState` completed before success appeared. Download completed and
  availability/other Download buttons refreshed. A subsequent external dispatcher
  save is visible as `external-model` and `http://localhost:23456`, while the newer
  unsaved provider remains unchanged (the only remaining settings draft).

Desktop viewport 1440×900; mobile-width 390×844; full-page uncropped PNGs. Chrome
DevTools MCP verified the flow with assertions, snapshot and wait_for. The harness
adapter denied file exports, so a separate isolated Chrome DevTools MCP stdio
session declaring this worktree as its root repeated the same assertions and used
`take_screenshot` directly to save all four files. No image synthesis/editing.
Browser error consoles were empty and mobile document width was 390px.

Real assets/dispatcher, safe synthetic values, isolated port 8763 and temporary
configuration/dictionary writes only in `/tmp/ptarmigan-issue-3`. Native bridge
transport/effects, permissions, daemon and cache/download are mocked; pending save
is a manually released transport gate, not native performance evidence. Native
WKWebView scheduling, build/launch, TCC, real downloads/inference and VoiceOver
remain untested. Shared port 8765 untouched. No cleanup workflow; images retained
as review-only evidence, never referenced by application code.
