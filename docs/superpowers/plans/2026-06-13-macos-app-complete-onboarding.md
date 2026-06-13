# PtarmiganFlow macOS アプリ完結型オンボーディング 実装計画

> **For agentic workers / codex:** この計画は Phase ごとに独立して動作・検証可能。各タスクは TDD（失敗するテスト→最小実装→緑→コミット）で進め、頻繁にコミットする。UI（PyObjC）のロジックは可能な限り純粋関数 / 状態機械に切り出してユニットテストし、PyObjC 呼び出し自体は既存 `tests/test_macos_app.py` のソース文字列検査パターンで担保する。

**Goal:** PtarmiganFlow.app を「署名済み単一プロセスで録音・文字起こし・常駐・設定を完結する」アプリにし、TCC 権限を "PtarmiganFlow" に帰属させ、Wispr Flow 風のステップ式オンボーディングと辞書 GUI を提供する。

**Architecture:** AppKit がメインスレッド/メイン RunLoop を専有し、`PtarmiganFlowDaemon.run_forever()` を background thread で実行。pynput の listener が自前スレッドで動くため event tap は background でも機能する。Homebrew CLI (`pflow run`) は従来どおり別経路で維持。

**Tech Stack:** Python 3.11, PyObjC (AppKit/Foundation/ServiceManagement), pynput, sounddevice, PyInstaller, pytest。

参照 spec: `docs/superpowers/specs/2026-06-13-macos-app-complete-onboarding-design.md`

実行コマンド前提: `uv run pytest` でテスト、ローカル `.app` 再ビルドは `packaging/macos/PtarmiganFlow.spec` を用いた PyInstaller（後述）。

---

## ファイル構成

| ファイル | 責務 | Phase |
|---|---|---|
| `src/ptarmigan_flow/app_daemon_controller.py`（新規） | アプリ内デーモンの起動/停止スレッド管理（PyObjC 非依存・テスト可能） | 1 |
| `src/ptarmigan_flow/onboarding_flow.py`（新規） | オンボーディング・ステップ状態機械（PyObjC 非依存・テスト可能） | 2 |
| `src/ptarmigan_flow/login_item.py`（新規） | SMAppService ラッパ（登録/解除/状態） | 3 |
| `src/ptarmigan_flow/corrections_editor_model.py`（新規） | 辞書 CRUD/検証ロジック（PyObjC 非依存・テスト可能） | 4 |
| `src/ptarmigan_flow/macos_app.py`（変更） | AppController + メニューバー + ウィザード UI を上記モデルに配線 | 1-4 |
| `packaging/macos/PtarmiganFlow.spec` / `app_bundle.py`（変更） | `LSUIElement=true` 追加 | 3 |
| `tests/test_app_daemon_controller.py` 他（新規） | 各モデルのユニットテスト | 1-4 |

設計方針: **ロジックは PyObjC 非依存モジュールに寄せ、`macos_app.py` は配線（thin glue）に留める。** これでテスト可能性と可読性を確保する。

---

## Phase 1 — 権限帰属の根治（アプリ内デーモン同一プロセス起動）

### Task 1.1: DaemonController（起動/停止スレッド管理）

**Files:**
- Create: `src/ptarmigan_flow/app_daemon_controller.py`
- Test: `tests/test_app_daemon_controller.py`

- [ ] **Step 1: 失敗するテストを書く**
  `DaemonController` の振る舞いを定義する。注入された factory（`() -> daemon-like`）で daemon を生成し、`start()` で background thread を起動して `run_forever()` を呼ぶ。`is_running` が True になる。`stop()` で daemon の `stop()` を呼びスレッド join。二重 start は無視。daemon 生成や run_forever が例外を投げても `start()` は呼び出し側を巻き込まず、`last_error` に記録し `is_running` が False に戻る。
  - フェイク daemon（`run_forever` がイベント待ち、`stop` でイベントセット）を使ったテスト: start→is_running True、stop→is_running False。
  - factory が例外→`last_error` 設定、`is_running` False。
- [ ] **Step 2: テストが失敗することを確認** `uv run pytest tests/test_app_daemon_controller.py -v`（ImportError）
- [ ] **Step 3: 最小実装** スレッド管理・例外捕捉・状態フラグを実装。
- [ ] **Step 4: テスト緑** `uv run pytest tests/test_app_daemon_controller.py -v`
- [ ] **Step 5: コミット** `feat(macos-app): add in-process daemon controller`

### Task 1.2: config から daemon を構築する factory

**Files:**
- Modify: `src/ptarmigan_flow/app_daemon_controller.py`（factory 関数追加）
- Test: `tests/test_app_daemon_controller.py`

- [ ] **Step 1: 失敗するテスト** `build_daemon_from_config(config_path)` が `ensure_config_exists` を呼び、`load_config` → `PtarmiganFlowDaemon` 相当を生成する経路をテスト（重い STT バックエンド生成は monkeypatch でフェイク化）。CLI `run` が既に行っている構築手順（`cli.py` の run 実装を参照）と同じ初期化を再利用する。
- [ ] **Step 2-5:** 失敗確認 → 実装（既存 CLI run の構築ロジックを共有関数に抽出して再利用、DRY）→ 緑 → コミット `refactor: share daemon build between cli run and app`

### Task 1.3: macos_app.py をデーモン制御に配線

**Files:**
- Modify: `src/ptarmigan_flow/macos_app.py`
- Test: `tests/test_macos_app.py`（ソース文字列検査）

- [ ] **Step 1: 失敗するテスト** `macos_app.py` ソースに `DaemonController`、`start`/`stop` 配線、Start/Stop アクションが存在することを検査。全権限付与時に自動 start する分岐を検査。
- [ ] **Step 2-4:** `OnboardingController` に `DaemonController` を保持し、「Start Dictation / Stop」操作と権限充足時の起動を実装。`applicationWillTerminate_` で `stop()`。
- [ ] **Step 5: コミット** `feat(macos-app): run dictation daemon in-process`

### Task 1.4: 実機検証（権限帰属）

- [ ] ローカルで `.app` 再ビルド（手順は下記「ローカル再ビルド手順」）。
- [ ] `.app` 起動 → Start → マイク/アクセシビリティ/入力監視のプロンプトが **"PtarmiganFlow"** で出ることを確認（"python3.11" が出ない）。
- [ ] ホットキー hold→release で録音→文字起こし→アクティブアプリへ貼り付けが動作。
- **受け入れ基準:** 上記すべて OK。`uv run pytest` 全緑。

---

## Phase 2 — ステップ式オンボーディング

### Task 2.1: OnboardingFlow 状態機械

**Files:**
- Create: `src/ptarmigan_flow/onboarding_flow.py`
- Test: `tests/test_onboarding_flow.py`

- [ ] **Step 1: 失敗するテスト** ステップ列 `[language, microphone, accessibility, input_monitoring, done]` を持つ状態機械を定義。
  - `current_step` 取得、`advance()` で次へ。
  - `refresh(report)` に `PermissionReport` を渡すと、現在が権限ステップで該当権限が True なら自動で次へ進む。
  - 既に付与済みの権限ステップは `start()`/`refresh()` 時に自動スキップ。
  - 言語ステップは `choose_language(code)` で確定して advance。
  - `is_complete` プロパティ。
- [ ] **Step 2-5:** 失敗確認 → 実装 → 緑 → コミット `feat: onboarding step state machine`

### Task 2.2: ウィザード UI + 自動ポーリング配線

**Files:**
- Modify: `src/ptarmigan_flow/macos_app.py`
- Test: `tests/test_macos_app.py`

- [ ] **Step 1: 失敗するテスト** ソース検査: `OnboardingFlow` 使用、`NSTimer`（`scheduledTimerWithTimeInterval_`）による定期 `refresh`、`applicationDidBecomeActive_` での即時再チェック、手動 Refresh ボタンの撤去、言語選択 UI（3 言語）と各権限ステップの「Allow」「Open System Settings」配線、ステップ単位表示。
- [ ] **Step 2-4:** ウィザードを 1 ステップずつ表示する UI に作り替え。タイマーで現在ステップ権限を `check_all_permissions()` 評価し、付与検知で次ステップへ。言語選択は config 保存（`config.py` の保存 API を利用）。
- [ ] **Step 5: コミット** `feat(macos-app): step-by-step onboarding wizard with auto-polling`

### Task 2.3: 実機検証

- [ ] 再ビルド → 各権限を System Settings で許可した瞬間（数秒以内）に画面が次ステップへ自動前進。
- [ ] 付与済み権限はスキップ。言語選択が config に保存される。
- **受け入れ基準:** 上記 OK、`uv run pytest` 全緑。

---

## Phase 3 — メニューバー常駐 + ログイン起動トグル

### Task 3.1: LSUIElement 追加（spec + app_bundle）

**Files:**
- Modify: `packaging/macos/PtarmiganFlow.spec`、`src/ptarmigan_flow/app_bundle.py`
- Test: `tests/test_macos_release_packaging.py`、`tests/test_app_bundle.py`

- [ ] **Step 1: 失敗するテスト** spec の `info_plist` と `app_bundle` の `info_payload` に `"LSUIElement": True` が含まれることを検査。
- [ ] **Step 2-5:** 失敗確認 → 両所に追加 → 緑 → コミット `feat(packaging): make app a menu-bar (LSUIElement) app`

### Task 3.2: LoginItem（SMAppService）ラッパ

**Files:**
- Create: `src/ptarmigan_flow/login_item.py`
- Test: `tests/test_login_item.py`

- [ ] **Step 1: 失敗するテスト** `register()` / `unregister()` / `is_enabled()` を定義。非 macOS では no-op で False。macOS は `SMAppService.mainAppService()` を呼ぶ（PyObjC 呼び出しは monkeypatch でフェイク注入できる構造にし、ロジック分岐をテスト）。
- [ ] **Step 2-5:** 失敗確認 → 実装 → 緑 → コミット `feat: SMAppService login item wrapper`

### Task 3.3: メニューバー UI 配線

**Files:**
- Modify: `src/ptarmigan_flow/macos_app.py`
- Test: `tests/test_macos_app.py`

- [ ] **Step 1: 失敗するテスト** ソース検査: `NSStatusBar`/`NSStatusItem` 使用、メニュー項目（録音状態 / Start・Stop / 設定（オンボーディング再表示）/ 辞書編集 / ログイン起動トグル / Quit）、launchd 用ボタン（`installLaunchAgent:` / `restartLaunchAgent:`）の撤去、`LoginItem` の配線。
- [ ] **Step 2-4:** メニューバーアイテムを実装。ステータスメニューからウィンドウ表示を制御。launchd UI を撤去（`launchd.py` 自体は残す）。
- [ ] **Step 5: コミット** `feat(macos-app): menu-bar residency and login-start toggle`

### Task 3.4: 実機検証

- [ ] 再ビルド → 起動後 Dock 非表示・メニューバー常駐。メニューから Start/Stop/設定/辞書編集。
- [ ] ログイン起動トグル ON → 再ログインで自動起動、OFF → 起動しない。
- [ ] `pflow run`（CLI 経路）が回帰なく動作。
- **受け入れ基準:** 上記 OK、`uv run pytest` 全緑。

---

## Phase 4 — 変換辞書 GUI エディタ

### Task 4.1: CorrectionsEditorModel（CRUD/検証）

**Files:**
- Create: `src/ptarmigan_flow/corrections_editor_model.py`
- Test: `tests/test_corrections_editor_model.py`

- [ ] **Step 1: 失敗するテスト** 既存 `TomlCorrectionRepository` を用いて load/save。
  - exact/regex セクションの行追加・編集・削除。
  - `validate()` が不正な正規表現を検出（`re.compile` 失敗を行単位で報告）。
  - `save(path)` が正しい TOML 構造（`[exact]` / `[regex]` に `"正規表記" = ["候補", ...]`）で書き出す。round-trip テスト（save→load で一致）。
- [ ] **Step 2-5:** 失敗確認 → 実装（書き出しは tomli-w 等、既存依存を確認し repository 経由で）→ 緑 → コミット `feat: corrections dictionary editor model`

### Task 4.2: 辞書エディタ UI 配線

**Files:**
- Modify: `src/ptarmigan_flow/macos_app.py`
- Test: `tests/test_macos_app.py`

- [ ] **Step 1: 失敗するテスト** ソース検査: 辞書エディタウィンドウ（`NSTableView` もしくは行編集 UI）、`CorrectionsEditorModel` 使用、保存アクション、regex 検証エラー表示の配線。
- [ ] **Step 2-4:** メニュー「辞書編集」から開くウィンドウで exact/regex を編集・保存。保存時に検証エラー表示。保存後、稼働中デーモンへ反映（リロード API があれば呼ぶ。無ければ「再起動で反映」表示 + 最小リロードフックの検討）。
- [ ] **Step 5: コミット** `feat(macos-app): corrections dictionary GUI editor`

### Task 4.3: 実機検証

- [ ] 再ビルド → GUI で exact/regex ルール追加 → ファイルに正しい TOML で保存。
- [ ] 不正な正規表現は保存前にエラー表示。
- [ ] 保存した辞書が文字起こし結果に反映。
- **受け入れ基準:** 上記 OK、`uv run pytest` 全緑。

---

## ローカル再ビルド手順（各 Phase の実機検証用）

1. リリース用 venv で依存を同期（既存 `.release-venv` を利用）。
2. PyInstaller でビルド: `packaging/macos/PtarmiganFlow.spec` を使用（既存リリース workflow / `README.dev.md` の手順に準拠。codex は実装時に正確なコマンドを README.dev.md / `.github/workflows` から確認すること）。
3. ad-hoc 署名: `codesign --force --deep --sign - dist/PtarmiganFlow.app`。
4. 起動: `open dist/PtarmiganFlow.app`。
5. TCC をリセットして再プロンプトを誘発する場合: `tccutil reset Microphone com.ptarmiganflow.app`（Accessibility / ListenEvent も同様）。

注: 実機の TCC プロンプト名・常駐・ログイン起動・辞書反映の最終確認はユーザーが手元で実施（検証方法の合意済み）。Claude/codex はビルド成功・ユニットテスト緑まで担保する。

---

## Self-Review（spec カバレッジ）

- python3.11 帰属根治 → Phase 1（同一プロセス起動）✓
- 設定後に画面が切り替わらない → Phase 2（自動ポーリング/`applicationDidBecomeActive_`）✓
- 言語選択 + 1 ステップずつ認証 → Phase 2（OnboardingFlow 状態機械）✓
- 常駐への疑問 → Phase 3（メニューバー常駐 + 任意ログイン起動、launchd アプリ経路撤去）✓
- 変換辞書を画面で操作 → Phase 4（CorrectionsEditorModel + UI）✓
- `pflow run` 維持 → Phase 1/3 で daemon.py 本体共有・CLI 経路非破壊、Phase 3 で回帰確認 ✓
- プレースホルダ無し / 型・名称整合（DaemonController, OnboardingFlow, LoginItem, CorrectionsEditorModel）✓
