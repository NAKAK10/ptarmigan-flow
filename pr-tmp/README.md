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
