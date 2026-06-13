# Phase 7: WKWebView による UI 刷新 設計書

- 日付: 2026-06-13
- 前提: Phase 1-6b 完了。全ロジックは PyObjC 非依存モジュールに分離済み。
- 方針(承認済み): 一括で全画面 WKWebView 化 / デザインは Claude 定義のダークモダン。

## 目的
素の AppKit 固定座標 UI を、WKWebView + HTML/CSS/JS のモダンUIに刷新し、Amical/Wispr Flow に見た目で対抗する。オンボーディング・設定・モデル選択(DL進捗)・辞書編集・LLM補正選択を新UIに集約。

## デザイントークン(ダークモダン)
- 背景 `#0E0F12` / サーフェス `#1A1D23` / 高位サーフェス `#22262E` / 境界 `#2A2E37`
- テキスト `#E6E8EC` / 補助 `#9AA0AA`
- アクセント `#6E8BFF`(hover `#869CFF`) / 成功 `#3DD68C` / 待機 `#9AA0AA` / エラー `#FF6B6B`
- 角丸 14px、ソフトシャドウ、150ms ease 遷移、フォント `-apple-system, "SF Pro Text"`、見出し semibold
- レイアウト: 中央カード(オンボーディング)、セクション分けフォーム(設定)、状態ピル、進捗バー

## アーキテクチャ
### ホスト(PyObjC)
- 新規 `src/ptarmigan_flow/web_ui.py`: 単一 NSWindow + WKWebView を生成するコントローラ。
  - `WKWebViewConfiguration` に `WKUserContentController` を設定し、`addScriptMessageHandler:name:"bridge"` を登録。
  - JS→Py: `window.webkit.messageHandlers.bridge.postMessage({id, action, payload})`。Py 側ハンドラが action を解釈し、結果を `id` 付きで JS へ返す。
  - Py→JS: `web_view.evaluateJavaScript_completionHandler_("window.app.dispatch(<json>)", None)`。
  - HTML/CSS/JS は同梱リソースから `loadFileURL:allowingReadAccessToURL:`(または loadHTMLString)で読み込む。
- `macos_app.py`: 既存の各 AppKit ウィンドウ描画(オンボーディング/設定/辞書/進捗)を WebUI に置換。NSStatusItem(ネイティブメニュー)・freeze_support・DaemonController(subprocess)・権限サブプロセスポーリング・モデルDL子プロセス・login_item は維持し、Web UI からのブリッジ要求に配線する。

### Web 資産
- `src/ptarmigan_flow/webui/index.html` / `app.css` / `app.js`(必要なら分割)。フレームワーク不要のバニラ実装(ビルド工程を増やさない)。クライアント側ルーティングで画面切替。
- PyInstaller spec と app_bundle に webui ディレクトリを datas として同梱。`importlib.resources` でパス解決。

### ブリッジ action 一覧(Py 側で処理、既存ロジックに委譲)
- `getState`: 現在の言語・オンボーディングstep・権限状態・モデル・設定・辞書・login有効・daemon稼働 を返す。
- `chooseLanguage{code}`: 言語保存(write_config)+ onboarding 永続化(mark_language_selected)+ ui言語切替。
- `requestPermission{kind}`: request_*_permission を呼ぶ。
- `openSystemSettings{kind}`: 該当 Privacy ペインを開く。
- 権限ポーリング: Py が背景スレッドでサブプロセス権限チェックし、変化を Py→JS push(`permissionsChanged`)。許可検知で onboarding 自動前進、全許可で daemon 起動(モデル未DLなら先にDL)。
- `startDictation`/`stopDictation`: DaemonController。is_model_downloaded で未DLなら download-model 子プロセス→進捗を `downloadProgress` push→完了で daemon 起動。
- `saveSettings{model,language,hotkey,output_mode, llm_correction{mode,provider,model,base_url}}`: AppSettingsModel + LLM補正設定を検証し write_config。
- `listModels`: available_model_entries + 各 is_model_downloaded 状態。
- 辞書: `loadDictionary`/`saveDictionary`(CorrectionsEditorModel、regex検証エラー返却)。
- `toggleLogin`: login_item register/unregister。
- `restartApp`: app_relaunch。
- 文言: onboarding_strings を Py→JS に渡し、JS は受け取った辞書で描画(英語ハードコードを JS に作らない。i18n は Python の単一ソースを使用)。

## LLM補正の選択式(新規・設定に集約)
- config の `[text.llm_correction]`(mode: always/never/ask, provider, model, base_url 等)を設定フォームで編集。AppSettingsModel を拡張 or 専用モデルを追加して検証・保存。

## 非対象(YAGNI)
- メニューバーの popover 化(NSStatusItem ネイティブメニュー維持)。
- Web フレームワーク/バンドラ導入(バニラ JS)。
- ダーク/ライト切替(当面ダーク固定)。

## テスト/検証
- ブリッジ action ハンドラ(Py)を PyObjC 非依存に切り出し、action→既存ロジック呼び出しの単体テスト。
- web 資産の同梱(datas)を packaging テストで確認。
- macos_app/web_ui のソース文字列検査(WKWebView/messageHandler/evaluateJavaScript/各 action 配線)。
- 実機: Claude が `.app` 再ビルド→起動→WebUI 描画・オンボーディング自動前進・設定保存・モデルDL進捗 を確認。
