# PtarmiganFlow macOS アプリ完結型オンボーディング 設計書

- 日付: 2026-06-13
- 対象: `src/ptarmigan_flow/macos_app.py`、`permissions.py`、`launchd.py`、`app_bundle.py`、`daemon.py`、`packaging/macos/`
- 実装担当: codex（指示出し・受け入れ確認は Claude）

## 背景と問題

現状の PtarmiganFlow には実行経路が 2 つ混在している。

1. ダウンロード版（PyInstaller 署名済み `PtarmiganFlow.app`）: `macos_app.py` のオンボーディング画面が本体。
2. Homebrew 版（`pflow` CLI + launchd 常駐）: `~/Applications/PtarmiganFlow.app` は bootstrap スクリプト経由で **Homebrew の python3.11 + uv** を起動するラッパー。launchd 常駐デーモンもこれを使う。

ユーザーから報告された不具合:

- **「"python3.11" がマイクへのアクセス権を求めています」** … 実際に録音するプロセスが python3.11 のため、macOS の TCC が権限を "python3.11" に帰属させる。アプリとして不適切。（経路 2 が原因）
- **System Settings で許可しても画面が切り替わらない** … `macos_app.py` のオンボーディングに自動ポーリングが無く、手動「Refresh」が必要。
- **オンボーディングが一覧フラット表示**で、言語選択が無く、Wispr Flow / aquaVoice のような 1 ステップずつのウィザードになっていない。
- **変換辞書（transcription_corrections.toml）が TOML 手書き**で GUI から編集できない。
- **常駐（launchd CLI 再実行）方式**そのものへの疑問。

## 決定事項（ユーザー承認済み）

| 項目 | 決定 |
|---|---|
| 製品の正 | 署名済み `PtarmiganFlow.app` 完結型（TCC 帰属 = "PtarmiganFlow"） |
| 録音/文字起こし | アプリと**同一プロセス**で実行（AppKit がメインスレッド、デーモンは背景スレッド） |
| 常駐 | メニューバー常駐（LSUIElement）+ ログイン起動は**任意トグル（デフォルト OFF）**、オンボーディング末尾で提案 |
| ログイン起動 | SMAppService / ログイン項目（アプリ経路では launchd CLI デーモンを廃止） |
| Homebrew CLI | `pflow run` は**維持**（開発者/CLI ユーザー向け。CLI 経路の python 帰属は許容） |
| オンボーディング | 言語選択 → マイク → アクセシビリティ → 入力監視 を 1 ステップずつ、System Settings 復帰時に自動ポーリングで前進 |
| 変換辞書 | GUI エディタで追加/編集/削除/保存 |
| 検証 | ローカルで `.app` を再ビルド（ad-hoc 署名）して実機の TCC プロンプト名を確認 |

## アーキテクチャ概要

```
PtarmiganFlow.app (署名済 / notarized)  ── 単一プロセス ──
  ├─ NSApplication（メインスレッド / メインRunLoop を専有）
  │    ├─ メニューバー常駐 (LSUIElement = true)
  │    ├─ オンボーディング・ウィザード（言語→マイク→アクセシビリティ→入力監視）
  │    │    └─ NSTimer ポーリング（System Settings 復帰時に状態再チェックして自動前進）
  │    ├─ ログイン起動トグル（SMAppService）
  │    └─ 変換辞書エディタ
  └─ 背景スレッド: PtarmiganFlowDaemon.run_forever()
       （Quartz イベントタップ用に当該スレッドの CFRunLoop を回す）
       → 録音 = sounddevice, ホットキー = Quartz event tap, 出力注入
       → すべて同一署名プロセス内 → TCC は "PtarmiganFlow" に帰属
```

Homebrew CLI (`pflow run`) は従来どおり daemon.py を直接プロセスとして起動する別経路として残す。アプリ経路と CLI 経路で daemon.py 本体は共有する。

## Phase 1 — 権限帰属の根治（最優先 🔴）

**目的**: マイク/アクセシビリティ/入力監視のプロンプトが "PtarmiganFlow" として出るようにする。

### 変更点
- `macos_app.py` に「デーモンをアプリと同一プロセスの背景スレッドで起動/停止する」ライフサイクルを追加する。
  - `OnboardingController`（または新設の AppController）が `PtarmiganFlowDaemon` を保持。
  - 全権限が揃ったら「Start / Stop」操作（Phase 3 でメニューバー化）でデーモンスレッドを起動。
  - AppKit がメインスレッドを専有するため、`run_forever()` は `threading.Thread(daemon=True)` で実行。停止時に `stop()` を呼ぶ。
- 背景スレッドでの Quartz イベントタップ動作確認。`hotkey_monitor.py` が自前 run loop を持つか、専用スレッドの `CFRunLoopRun()` が必要かを実装時に確認し、必要なら最小限の調整を加える。
- `config.toml` が無い初回でも安全に起動できるよう `ensure_config_exists` を起動前に呼ぶ。

### 受け入れ基準
- ローカル再ビルドした `.app` を起動 → デーモン起動 → マイク/アクセシビリティ/入力監視のプロンプトに **"PtarmiganFlow"** が表示される（"python3.11" が出ない）。
- ホットキー押下→離しで録音→文字起こし→貼り付けがアプリ単体で動く。
- 既存ユニットテストが緑。daemon の起動/停止スレッド管理にユニットテストを追加。

## Phase 2 — ステップ式オンボーディング

**目的**: 言語選択から各権限を 1 つずつ確実にクリアさせ、System Settings 復帰で自動前進。

### ウィザード・ステップ
1. **言語選択** — en / ja / zh（既存 config の language キー）。選択を config に保存。
2. **マイク** — 説明 → 「許可」ボタンで `request_microphone_permission()` → 付与されたら自動で次へ。
3. **アクセシビリティ** — 「System Settings を開く」+ `request_accessibility_permission()`。付与で自動前進。
4. **入力監視** — 同上（`request_input_monitoring_permission()`）。
5. **完了** — Start 案内 + ログイン起動トグル提案（Phase 3）。

### 自動ポーリング
- `NSTimer`（例: 1.0 秒間隔）で現在ステップの権限を `check_*` で再評価。
- 付与を検知したらタイマーを止め、次ステップへ遷移し、新ステップ用にタイマー再開。
- ウィンドウが再アクティブ化（`applicationDidBecomeActive_`）したタイミングでも即時再チェック。
- 手動「Refresh」ボタンは廃止（または隠す）。

### 受け入れ基準
- 各権限を System Settings で許可した瞬間（数秒以内）に画面が次ステップへ進む。
- 既に許可済みの権限はステップを自動スキップ。
- 戻る/進む操作で状態が壊れない。

## Phase 3 — メニューバー常駐 + ログイン起動トグル

**目的**: アプリをメニューバー常駐にし、ログイン起動を任意 ON/OFF。launchd CLI デーモンをアプリ経路から撤去。

### 変更点
- `Info.plist`（PyInstaller spec と `app_bundle.py` 両方）に `LSUIElement = true` を追加し、Dock 非表示のメニューバーアプリにする。
- `NSStatusItem` でメニューバーアイコン + メニュー（録音状態表示 / Start・Stop / 設定（オンボーディング再表示） / 辞書編集 / ログイン起動トグル / 終了）。
- ログイン起動は `SMAppService.mainAppService`（macOS 13+）で register/unregister。デフォルト OFF。オンボーディング完了画面とメニューからトグル。
- アプリ経路では launchd エージェントのインストール/再起動 UI を撤去（`installLaunchAgent:` / `restartLaunchAgent:` ボタンを削除）。`launchd.py` 自体は CLI 用に残す。

### 受け入れ基準
- 起動後 Dock に出ず、メニューバーにアイコンが常駐。
- メニューから Start/Stop、辞書編集、設定再表示ができる。
- ログイン起動トグル ON → 再ログインでアプリが自動起動、OFF → 起動しない。
- `pflow run`（CLI 経路）は従来どおり動作（回帰なし）。

## Phase 4 — 変換辞書 GUI エディタ

**目的**: exact/regex ルールを画面で編集。

### 変更点
- 辞書の読み書きは既存 `text_processing/repository.py`（`TomlCorrectionRepository`）を再利用。
- エディタ UI（メニューから起動するウィンドウ）:
  - exact ルール: 「正規表記」→「誤変換候補（複数）」のテーブル。
  - regex ルール: 「正規表記」→「正規表現パターン（複数）」のテーブル。
  - 追加 / 編集 / 削除 / 保存。保存時に regex の妥当性を検証しエラー表示。
- 保存先は `resolve_dictionary_path()` が返すパス（デフォルト `~/.config/ptarmigan-flow/transcription_corrections.toml`）。
- 保存後、稼働中デーモンに辞書リロードを反映（再読込 API が無ければ最小限追加、または「再起動で反映」表示）。

### 受け入れ基準
- GUI で追加したルールがファイルに正しい TOML 形式で保存される。
- 不正な正規表現は保存前にエラー表示。
- 保存した辞書が文字起こし結果に反映される。

## テスト方針

- **ユニットテスト**: 既存の `tests/` パターンに従う。Phase ごとに、ロジック（デーモンスレッド管理、ステップ遷移の状態機械、辞書 CRUD、Info.plist 内容）をテスト。PyObjC の UI 呼び出しはソース文字列検査（既存 `test_macos_app.py` 方式）で担保。
- **実機検証**: 各 Phase 完了時にローカルで `.app` を再ビルド（ad-hoc 署名）し、TCC プロンプト名・画面遷移・常駐・辞書反映を手元で確認。

## 非対象（YAGNI）

- Homebrew CLI 経路の TCC 帰属改善（python3.11 のまま許容）。
- 自動アップデート機構の変更。
- Intel/Apple Silicon 双方の同時実機検証（Phase ごとは片側で可）。
- 多言語 UI（オンボーディング文言の国際化）。当面は既存の英語表記を踏襲。

## リスクと留意点

- **背景スレッドでの Quartz イベントタップ**: メインスレッド以外で event tap の CFRunLoop source を回す必要がある。動かない場合は、デーモンの run loop をメインに寄せ AppKit を別途調停する設計に切り替える可能性あり（Phase 1 実装時に最優先で検証）。
- **SMAppService は macOS 13+**: `LSMinimumSystemVersion` を確認し、下回る場合のフォールバック（ログイン項目 API）を検討。
- **ad-hoc 署名と notarized 署名で TCC 挙動が微妙に異なる**: ローカル検証は ad-hoc で行うが、本番は notarized。bundle identifier が一致していれば帰属名は一致する想定。
