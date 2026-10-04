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

No `pr-tmp` cleanup workflow exists here; these review-only assets are committed
with the PR and are not referenced by application code.
