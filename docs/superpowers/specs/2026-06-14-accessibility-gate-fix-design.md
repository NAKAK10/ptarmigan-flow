# Accessibility 権限ゲート修正 設計書

- 日付: 2026-06-14
- 対象: macOS アプリ（PyObjC + WKWebView オンボーディング）
- ブランチ: `feat/macos-app-complete-onboarding`

## 背景

ユーザー報告: 「(macOS) アプリでアクセシビリティへのアクセス画面から先に進めない」。
補足要望:
1. `pflow run` と同じく、キー長押しで音声入力できるようにしたい。
2. このアプリは「コマンド (`pflow run`) の代替」として動けばよい（過剰な作り込みは不要）。

注: 本リポジトリに iOS アプリは存在しない。ユーザーの言う「iOS app」= 本 macOS アプリ。

## 根本原因

- 権限チェックは `OnboardingController._check_permissions_in_background()` が
  `permissions.check_all_permissions_subprocess()` を呼び、
  `[sys.executable, "-m", "ptarmigan_flow.cli", "check-permissions"]` を**別プロセス**で実行している
  (`src/ptarmigan_flow/permissions.py:85`)。
- macOS のアクセシビリティ (`AXIsProcessTrusted`) と入力監視 (`CGPreflightListenEventAccess`) の
  trust は「アプリ署名 = 責任プロセス」単位。ビルドした `.app` では、System Settings で
  `PtarmiganFlow.app` に許可を出しても、子プロセスでの判定は trust 帰属がズレ
  `accessibility=False` を返し得る。
- 「GUI プロセスの TCC キャッシュ回避のため別プロセスで判定」という既存対策が、逆に
  **誤検出による詰まり**を生んでいる。
- さらに `OnboardingFlow.refresh()` (`onboarding_flow.py:74`) は許可が検出されるまで
  `accessibility` ステップで止まり続け、**手動で進む手段がない**ため完全に詰まる。

`pflow run` がターミナルで動くのは Terminal に許可が付いているため。`.app` 本体は別署名なので
許可の帰属がズレるのが効いている。

## 方針（ユーザー確定: 「検出を直す＋手動で進める」）

長押し録音デーモンは `pflow run` と同一 (`daemon_run_command` / `DaemonController`)。
オンボーディング完走→デーモン起動で**そのまま長押し録音が有効**になるため、ゲートを直せば
要望①②③はすべて満たされる。追加の入力機構実装は不要（YAGNI）。

### 柱1: 検出の修正（誤検出をなくす）

- アクセシビリティと入力監視の判定は **GUI 本体プロセス内**の
  `AXIsProcessTrusted()` / `CGPreflightListenEventAccess()` を正とする。
  - 実装方針: `_check_permissions_in_background` / `applyPermissionCheckResult_` の経路で、
    アクセシビリティと入力監視については in-process 判定を優先する。
    subprocess 判定と食い違った場合、**in-process が True なら True を採用**する
    （誤検出 False を潰す）。マイクは現状のままで可。
- macOS TCC は同一プロセス内で `False` をキャッシュし、許可付与の反映に**アプリ再起動**を
  要することがある。そこで**アクセシビリティ画面にも**（入力監視画面と同様の）
  **「アプリを再起動」ボタン**を出し、再起動後に正しい trust を読めるようにする。

### 柱2: 手動で進めるエスケープ

- アクセシビリティ／入力監視の権限ステップに **「次へ進む」ボタン**を追加。
- `OnboardingFlow` に検出待ちに依存せず前進させるメソッド（例: `advance_manually()` /
  現在の権限ステップを 1 つ進める）を追加し、新 bridge アクション経由で呼ぶ。
- これにより**絶対に詰まらない**。
- デーモン起動 `_start_daemon_if_ready()` (`macos_app.py:581`) は実権限を引き続き検証。
  手動で進めたが本当に未許可なら、**「○○の許可が必要です」と明示**し黙って失敗しない。

## 変更想定ファイル

- `src/ptarmigan_flow/permissions.py` — in-process 判定を正とするためのヘルパー/方針
- `src/ptarmigan_flow/macos_app.py` — 権限チェック経路の合成、手動 advance の配線、
  アクセシビリティ画面の再起動ボタン配線
- `src/ptarmigan_flow/onboarding_flow.py` — 手動 advance 用メソッド
- `src/ptarmigan_flow/web_bridge.py` — 新 bridge アクション（手動 advance、必要なら restart）
- `src/ptarmigan_flow/webui/app.js` — 「次へ進む」ボタン（アクセシビリティ/入力監視ステップ）
- `src/ptarmigan_flow/onboarding_strings.py` — ボタン文言（en/ja/zh）

## 点②③の扱い

- ②キー長押し入力: 既存デーモン (`pflow run` と同一) で充足。ゲート修正で完走すれば有効。
  既存ホットキー設定を使用、追加実装は不要。
- ③コマンド代替: `.app` は既に同じデーモンを起動する構造。本修正は最小限に留める。

## テスト方針

- `onboarding_flow.py`: 手動 advance メソッドの単体テスト（権限ステップで前進し、
  検出未達でも進むこと）。
- `permissions.py`: in-process 判定優先ロジックの単体テスト
  （subprocess=False かつ in-process=True のとき True を返す等）。
- 既存のパッケージング/スモークテストで権限名 (Microphone/Accessibility/Input Monitoring) の
  可視性が壊れないこと。
- 手動 QA: ビルド .app で許可付与→再起動で自動進行、未付与でも「次へ進む」で完走、
  完走後に長押し録音が動作することを確認。

## 非対象（スコープ外）

- iOS アプリ新規作成（存在しない）。
- 入力機構（ホットキー）の再設計。既存デーモンをそのまま使う。
- 権限そのものの不要化（出力にアクセシビリティ等が機能的に必要なため）。
