# Issue #4 — model download lifecycle

Safe synthetic data; real worktree webui assets and WebBridgeDispatcher served by an isolated copy of `/tmp/ptarmigan-dogfood` on port **8764** (`/tmp/ptarmigan-issue-4`). Shared port 8765 was not changed.

- `issue-4-desktop-preparing.png`: indeterminate preparation; other downloads disabled.
- `issue-4-desktop-progress.png` / `issue-4-mobile-progress.png`: supplied 42% progress; another model remains selectable; Download/Retry globally disabled.
- `issue-4-desktop-outcomes.png` / `issue-4-mobile-outcomes.png`: downloaded badge after done/getState, distinct backend busy text, ordinary failure with Retry; unsaved language/output/provider inputs retained.

Desktop viewport: 1440×900. Mobile-width viewport: 390×844. All images are full-page, uncropped browser screenshots, not image composites. Chrome DevTools MCP pageId 2 verified/captured these states via snapshots, wait_for and take_screenshot. MCP could return screenshot attachments but rejected filePath writes as outside its configured workspace roots; the committed files were captured by a separate temporary headless Chrome/Puppeteer session reproducing the same flow against the same isolated server (fallback capture). That session additionally asserted live input/card node identity, selection preservation, cache refresh, busy/error unlock and absence of JS errors.

Native downloads/cache availability, permissions/TCC, daemon, audio/input injection, login/relaunch effects are **mocked**. No native build, permission change, real model download or speech recognition was performed. Node lifecycle tests exercise real frontend functions with a stub bridge/DOM and are not a VoiceOver test.

No `pr-tmp` cleanup workflow exists in this repository; these small review-only images are intentionally committed and are not referenced by application code.
