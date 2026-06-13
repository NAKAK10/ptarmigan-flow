# Phase 5: アプリ内設定フォーム・モデル選択・完全ローカライズ 設計書

- 日付: 2026-06-13
- 前提: `2026-06-13-macos-app-complete-onboarding-design.md`（Phase 1-4 完了済み）
- 実装: codex / 指示・レビュー・実機確認: Claude

## 背景（実機テストで判明した問題）

1. **アプリが起動しない（実体: デーモンが起動時クラッシュ）**
   - ダウンロード版 `.app` は意図的に **moonshine 専用のコンパクトビルド**（spec が granite/mlx/voxtral/torch を除外）。
   - ユーザーの `~/.config/ptarmigan-flow/config.toml` は `model = "granite:..."`（CLI で使った設定）。
   - 起動時にデーモンが自動起動 → granite バックエンド構築 → `granite_mlx` 未バンドル → `ModuleNotFoundError` → デーモン失敗。GUI がエラーを握り潰し、ユーザーには起動失敗に見える。
2. **動的メッセージの英語残存**: 静的ラベルのみ Phase で localize したが、`_set_message` 等の動的メッセージ・メニュー項目・辞書エディタのラベルが英語のまま。
3. **「ディクテーション開始」が分かりにくい**（文言改善要望）。
4. **「設定を開く」が生 TOML を開く**ため手編集で壊れ得る。フォーム＋保存ボタンにすべき。
5. （別件・修正済み）`multiprocessing.freeze_support()` 欠如 → spawn ワーカーが GUI 再実行。

## 決定事項（ユーザー承認済み）

- バックエンド方針: **アプリ内にモデル選択UIを追加**（同梱・利用可能なモデルを画面で選び config に保存）。
- 設定編集: **生ファイルを開くのをやめ、フォーム＋保存ボタン**で特定フィールドのみ書き込む。
- 文言: 「音声入力」系の分かりやすい表現に変更し、**全ユーザー向け文字列を en/ja/zh ローカライズ**。

## 設計

### 1. バックエンド利用可否判定（新規 `src/ptarmigan_flow/stt/availability.py`）
- `is_backend_available(backend: str) -> bool`: 代表モジュールを `importlib.util.find_spec` で判定（moonshine→`ptarmigan_flow.stt.moonshine`, granite→`granite_mlx`/`granite_transformers`, mlx→`mlx_whisper`, voxtral→`voxtral_mlx`/`voxtral_transformers`, vllm→常時可）。例外は False。
- `available_model_entries() -> list[CatalogEntry]`: `verified_model_entries()` を利用可否で絞り込み。
- PyObjC 非依存・テスト可能。

### 2. 完全ローカライズ（`src/ptarmigan_flow/onboarding_strings.py` 拡張）
- **全ユーザー向け文字列**を文字列キー化（静的ラベル＋動的メッセージ＋メニュー＋辞書エディタ＋設定フォーム）。en/ja/zh、キー単位で en フォールバック。
- 書式付きメッセージは `{placeholder}` 付きテンプレートにし、呼び出し側で `.format(...)`。
- 文言改善: dictation 系は en="Start/Stop Voice Input" / "Voice input running/stopped"、ja="音声入力を開始/停止" / "音声入力 実行中/停止中"、zh="开始/停止语音输入" / "语音输入 运行中/已停止"。
- macos_app.py の全描画・全メッセージを `strings_for(self.ui_language)` 経由に統一（ハードコード英語ゼロ）。

### 3. アプリ内設定フォーム（macos_app.py に Settings ウィンドウ）
- メニュー「Settings / 設定」から開くフォーム。フィールド:
  - **Model**: `available_model_entries()` のラベルからプルダウン/ボタン選択（利用可能なもののみ）。選択した token を `config.model` に保存。
  - **Language**: en/ja/zh。
  - **Hotkey**: `config.hotkey.key`（既存の選択肢: right_cmd 等）。
  - **Output mode**: `config.output.mode`（direct_typing / clipboard_paste）。
- **Save ボタン**: 値を検証し `write_config()` で保存。生ファイルは書き換えず該当フィールドのみ更新（既存 config をロード→フィールド更新→保存）。保存後「再起動で反映」案内＋（任意）デーモン再起動。
- 「Open config file（上級者向け）」は小さく残す（既定の主動線はフォーム）。
- ロジック（フィールド定義・検証・保存）は PyObjC 非依存の `app_settings_model.py`（新規）に寄せてテスト可能化。

### 4. GUI 耐障害性・安全な自動起動（macos_app.py）
- 起動時の自動デーモン開始は、**設定モデルのバックエンドが利用可能なときのみ**実行。
- 利用不可なら起動せず、localized 通知「このアプリ版では設定中のモデル {model} は利用できません。設定からモデルを選択してください。」を表示し、**GUI は必ず表示・操作可能**を維持。
- デーモンが失敗しても last_error を localized メッセージで提示し、GUI は落とさない。

## テスト方針
- `availability.py`: find_spec を monkeypatch し各バックエンドの可否を検証。
- `onboarding_strings`: 3言語のキー集合一致・フォールバック・主要 ja 文言の厳密一致。
- `app_settings_model.py`: フィールド更新・検証・round-trip 保存（write_config→load_config 一致）。
- `macos_app`（ソース文字列検査）: Settings フォーム配線、ハードコード英語が残っていないこと（主要メッセージがキー経由であること）、自動起動のバックエンド可否ガード、文言改善キーの使用。
- **実機検証（Claude が実施）**: `.app` 再ビルド→起動→デーモンが moonshine 選択で起動→各画面の日本語表示→設定フォーム保存。

## 非対象（YAGNI）
- Hub 検索からの任意モデル追加 UI（当面は verified プリセットのみ）。
- granite/mlx 等の同梱（コンパクト維持）。
