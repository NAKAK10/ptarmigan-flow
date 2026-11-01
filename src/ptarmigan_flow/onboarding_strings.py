"""Localized strings for the native macOS app."""

from __future__ import annotations

ONBOARDING_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "accessibility_body": (
            "Allow PtarmiganFlow to control the active text field for voice input output."
        ),
        "accessibility_title": "Accessibility Access",
        "allow_button": "Allow",
        "all_permissions_granted_started_message": (
            "All permissions granted. Voice input started."
        ),
        "app_setup_title": "PtarmiganFlow Setup",
        "choose_language_body": "Select the transcription language to save into your config.",
        "choose_language_title": "Choose Language",
        "config_opened_message": "Opened config: {path}",
        "daemon_not_running_message": "Voice input daemon is not running yet.",
        "daemon_start_failed_message": "Could not start voice input: {error}",
        "dictation_running_menu": "Voice input running",
        "dictation_stopped_message": "Voice input stopped.",
        "dictation_stopped_menu": "Voice input stopped",
        "download_complete_message": "Download complete.",
        "download_failed_message": "Model download failed: {error}",
        "download_in_progress_message": "Downloading model... {percent}",
        "download_preparing_message": "Preparing model...",
        "dictionary_add_exact_button": "Add Exact",
        "dictionary_add_regex_button": "Add Regex",
        "dictionary_canonical_label": "Canonical",
        "dictionary_candidates_patterns_label": "Variants (comma or newline separated)",
        "dictionary_default_candidate": "candidate",
        "dictionary_default_pattern": "pattern",
        "dictionary_delete_button": "Delete",
        "dictionary_duplicate_key_message": "Duplicate rule name: {key}",
        "dictionary_editor_title": "Dictionary Editor",
        "dictionary_exact_rules_description": (
            "When the transcript exactly contains one of the listed variants, "
            "it is replaced with the rule name."
        ),
        "dictionary_exact_rules_title": "Exact Rules",
        "dictionary_invalid_rule_message": (
            "Invalid dictionary rule: [{section}] {key} {pattern}: {message}"
        ),
        "dictionary_load_failed_message": "Could not load dictionary: {error}",
        "dictionary_new_exact_rule": "New Exact Rule",
        "dictionary_new_regex_rule": "New Regex Rule",
        "dictionary_no_rules": "No rules yet.",
        "dictionary_patterns_label": "Patterns (one per line)",
        "dictionary_regex_rules_description": (
            "Text matching any of the regular-expression patterns is replaced with the "
            "rule name."
        ),
        "dictionary_regex_rules_title": "Regex Rules",
        "dictionary_save_button": "Save",
        "dictionary_save_failed_message": "Could not save dictionary: {error}",
        "dictionary_saved_message": (
            "Dictionary saved. Restart voice input to apply changes."
        ),
        "dictionary_unsaved_changes_label": "Unsaved changes",
        "done_body": (
            "Setup is complete. Hold your configured hotkey to dictate, or adjust settings."
        ),
        "done_title": "Ready for Voice Input",
        "edit_dictionary_menu": "Edit Dictionary",
        "grant_permissions_message": "Grant all permissions before starting voice input.",
        "hotkey_capture_change_button": "Change",
        "hotkey_capture_prompt": "Press the key you want to use…",
        "hotkey_capture_unsupported": (
            "Unsupported key. Press one of the modifier keys (⌘, ⇧, ⌥, ⌃)."
        ),
        "hotkey_confirm_body": (
            "Choose the key you will hold while speaking. You can change it later in Settings."
        ),
        "hotkey_confirm_button": "Use This Key",
        "hotkey_confirm_title": "Confirm Hotkey",
        "hotkey_label_left_alt": "Left ⌥ Option",
        "hotkey_label_left_cmd": "Left ⌘ Command",
        "hotkey_label_left_ctrl": "Left ⌃ Control",
        "hotkey_label_left_shift": "Left ⇧ Shift",
        "hotkey_label_right_alt": "Right ⌥ Option",
        "hotkey_label_right_cmd": "Right ⌘ Command",
        "hotkey_label_right_ctrl": "Right ⌃ Control",
        "hotkey_label_right_shift": "Right ⇧ Shift",
        "hotkey_select_label": "Key to use",
        "input_monitoring_body": "Allow PtarmiganFlow to detect the push-to-talk hotkey.",
        "input_monitoring_title": "Input Monitoring",
        "language_chinese": "Chinese",
        "language_english": "English",
        "language_japanese": "Japanese",
        "language_save_failed_message": "Could not save language: {error}",
        "language_saved_message": "Saved language to {path}.",
        "login_at_startup_button": "Login at Startup",
        "login_at_startup_menu": "Login at Startup",
        "login_disable_failed_message": "Could not disable login at startup.",
        "login_disabled_message": "Login at startup disabled.",
        "login_enable_failed_message": "Could not enable login at startup.",
        "login_enabled_message": "Login at startup enabled.",
        "microphone_body": "Allow PtarmiganFlow to capture audio while you hold the hotkey.",
        "microphone_title": "Microphone Access",
        "model_unavailable_message": (
            "The configured model {model} is not available in this app build. "
            "Choose a model in Settings."
        ),
        "open_config_advanced_button": "Open Config File (Advanced)",
        "open_setup_menu": "Open Setup…",
        "open_system_settings_button": "Open System Settings",
        "output_clipboard_paste": "Clipboard Paste",
        "output_direct_typing": "Direct Typing",
        "quit_menu": "Quit",
        "check_updates_menu": "Check for Updates…",
        "update_available_menu": "Update to v{version}…",
        "update_checking_menu": "Checking for updates…",
        "update_preparing_menu": "Preparing v{version}… {percent}",
        "update_restart_menu": "Restart to Update to v{version}",
        "update_unavailable_menu": "v{version}: Can't Update Automatically (Release Page)",
        "update_up_to_date_title": "You're up to date",
        "update_up_to_date_body": "PtarmiganFlow {version} is the latest version.",
        "update_available_title": "PtarmiganFlow {version} is available",
        "update_ready_title": "PtarmiganFlow {version} is ready to install",
        "update_ready_body": (
            "Restart now to update to v{version}. If you choose Later, the update is "
            "installed the next time you quit PtarmiganFlow."
        ),
        "update_dev_build_title": "Development build",
        "update_dev_build_body": (
            "This build ({current}) can't be updated automatically. The latest release is "
            "{version}."
        ),
        "update_blocked_dev_build": (
            "This build can't be updated automatically. Download the new version from the "
            "release page."
        ),
        "update_blocked_translocated": (
            "Move PtarmiganFlow.app to the Applications folder to enable automatic updates. "
            "You can also download the new version from the release page."
        ),
        "update_blocked_not_writable": (
            "PtarmiganFlow can't replace itself in its current location ({path}). Download the "
            "new version from the release page."
        ),
        "update_blocked_no_asset": (
            "The download for this Mac isn't ready yet. Try again later or check the release "
            "page."
        ),
        "update_check_failed_title": "Couldn't check for updates",
        "update_failed_title": "Update failed",
        "update_failed_body": "{error}\n\nThe current version was left unchanged.",
        "update_restart_button": "Restart to Update",
        "update_release_notes_button": "Release Notes",
        "update_later_button": "Later",
        "llm_ask_title": "Use LLM correction?",
        "llm_ask_body": (
            "LLM correction is set to \"Ask each time\". Choose whether to correct voice input "
            "with the LLM ({model}) while the app is running."
        ),
        "llm_ask_use_button": "Use",
        "llm_ask_skip_button": "Don't Use",
        "update_ok_button": "OK",
        "restart_app_button": "Restart App",
        "restart_failed_message": "Could not restart app.",
        "restart_required_note": (
            "If you just granted this in System Settings, you may need to restart the app to "
            "apply it."
        ),
        "route_dictionary": "Dictionary",
        "route_onboarding": "Setup",
        "route_settings": "Settings",
        "settings_button": "Settings",
        "settings_hotkey_help": "Voice input records only while this key is held down.",
        "settings_hotkey_label": "Hotkey",
        "settings_language_help": "Sets the language used to transcribe your speech.",
        "settings_language_label": "Language",
        "settings_llm_base_url_label": "Base URL",
        "settings_llm_mode_always": "Always",
        "settings_llm_mode_ask": "Ask each time",
        "settings_llm_mode_label": "Mode",
        "settings_llm_mode_never": "Never",
        "settings_llm_model_label": "Model",
        "settings_llm_provider_label": "Provider",
        "settings_llm_section_title": "LLM Correction",
        "settings_load_failed_message": "Could not load settings: {error}",
        "settings_menu": "Settings",
        "settings_model_download_button": "Download",
        "settings_model_download_done_message": "Download complete",
        "settings_model_busy_message": (
            "Another download is in progress. Try again when it finishes."
        ),
        "settings_model_refresh_failed_message": "Could not refresh model availability.",
        "settings_model_download_error_message": "Download failed: {error}",
        "settings_model_downloaded_badge": "Downloaded",
        "settings_model_label": "Model",
        "settings_model_preparing_message": "Preparing download…",
        "settings_model_retry_button": "Retry",
        "settings_models_section_title": "Models",
        "settings_no_models_message": "No bundled speech models are available.",
        "settings_open_config_button": "Open config file",
        "settings_output_mode_help": (
            "Choose how transcribed text is delivered to the active app."
        ),
        "settings_output_mode_label": "Output mode",
        "save_in_progress": "Saving and updating the view…",
        "save_retry_button": "Retry save",
        "save_retry_hint": "Check the values or error details, then retry saving.",
        "save_refresh_failed_message": "Saved, but the view could not be updated: {error}",
        "save_error_unknown": "No error details were provided.",
        "settings_save_button": "Save",
        "settings_save_failed_message": "Could not save settings: {error}",
        "settings_saved_message": "Settings saved. Restart voice input to apply changes.",
        "settings_validation_error": "Could not save settings. Check: {fields}",
        "settings_window_title": "Settings",
        "setup_item_accessibility": "Accessibility",
        "setup_item_hotkey": "Hotkey",
        "setup_item_input_monitoring": "Input Monitoring",
        "setup_item_language": "Language",
        "setup_item_microphone": "Microphone",
        "setup_item_separator": ", ",
        "setup_required_menu": "Setup required ({items})",
        "start_dictation_button": "Start Voice Input",
        "status_granted": "Granted",
        "status_waiting": "Waiting",
        "stop_dictation_button": "Stop Voice Input",
        "voice_input_started_message": "Voice input started.",
        "webui_error_title": "Error",
    },
    "ja": {
        "accessibility_body": (
            "音声入力の結果を入力するため、PtarmiganFlow が"
            "アクティブなテキスト欄を操作することを許可します。"
        ),
        "accessibility_title": "アクセシビリティへのアクセス",
        "allow_button": "許可する",
        "all_permissions_granted_started_message": (
            "すべての権限が許可されました。音声入力を開始しました。"
        ),
        "app_setup_title": "PtarmiganFlow セットアップ",
        "choose_language_body": "文字起こしに使う言語を選択して設定に保存します。",
        "choose_language_title": "言語を選択",
        "config_opened_message": "設定ファイルを開きました: {path}",
        "daemon_not_running_message": "音声入力デーモンはまだ起動していません。",
        "daemon_start_failed_message": "音声入力を開始できませんでした: {error}",
        "dictation_running_menu": "音声入力 実行中",
        "dictation_stopped_message": "音声入力を停止しました。",
        "dictation_stopped_menu": "音声入力 停止中",
        "download_complete_message": "ダウンロードが完了しました。",
        "download_failed_message": "モデルのダウンロードに失敗しました: {error}",
        "download_in_progress_message": "モデルをダウンロード中... {percent}",
        "download_preparing_message": "モデルを準備中...",
        "dictionary_add_exact_button": "完全一致を追加",
        "dictionary_add_regex_button": "正規表現を追加",
        "dictionary_canonical_label": "正規表記",
        "dictionary_candidates_patterns_label": "候補の表記（カンマまたは改行区切り）",
        "dictionary_default_candidate": "候補",
        "dictionary_default_pattern": "パターン",
        "dictionary_delete_button": "削除",
        "dictionary_duplicate_key_message": "ルール名が重複しています: {key}",
        "dictionary_editor_title": "変換辞書",
        "dictionary_exact_rules_description": (
            "認識結果に候補の表記が現れたとき、ルール名の表記に置き換えます。"
        ),
        "dictionary_exact_rules_title": "完全一致ルール",
        "dictionary_invalid_rule_message": (
            "無効な辞書ルール: [{section}] {key} {pattern}: {message}"
        ),
        "dictionary_load_failed_message": "変換辞書を読み込めませんでした: {error}",
        "dictionary_new_exact_rule": "新しい完全一致ルール",
        "dictionary_new_regex_rule": "新しい正規表現ルール",
        "dictionary_no_rules": "ルールがありません",
        "dictionary_patterns_label": "パターン（1行に1つ）",
        "dictionary_regex_rules_description": (
            "正規表現パターンに一致したテキストをルール名の表記に置き換えます。"
        ),
        "dictionary_regex_rules_title": "正規表現ルール",
        "dictionary_save_button": "保存",
        "dictionary_save_failed_message": "変換辞書を保存できませんでした: {error}",
        "dictionary_saved_message": "保存しました。反映には音声入力を再起動してください。",
        "dictionary_unsaved_changes_label": "未保存の変更",
        "done_body": (
            "セットアップが完了しました。設定したホットキーを押している間だけ"
            "音声入力できます。設定も変更できます。"
        ),
        "done_title": "音声入力の準備完了",
        "edit_dictionary_menu": "変換辞書を編集",
        "grant_permissions_message": "音声入力を開始する前にすべての権限を許可してください。",
        "hotkey_capture_change_button": "変更",
        "hotkey_capture_prompt": "設定したいキーを押してください…",
        "hotkey_capture_unsupported": (
            "サポートされていないキーです。修飾キー（⌘, ⇧, ⌥, ⌃）のいずれかを押してください。"
        ),
        "hotkey_confirm_body": (
            "音声入力中に押し続けるキーを選択します。あとで設定から変更できます。"
        ),
        "hotkey_confirm_button": "このキーを使う",
        "hotkey_confirm_title": "使用するホットキーを確認",
        "hotkey_label_left_alt": "左 ⌥ オプション",
        "hotkey_label_left_cmd": "左 ⌘ コマンド",
        "hotkey_label_left_ctrl": "左 ⌃ コントロール",
        "hotkey_label_left_shift": "左 ⇧ シフト",
        "hotkey_label_right_alt": "右 ⌥ オプション",
        "hotkey_label_right_cmd": "右 ⌘ コマンド",
        "hotkey_label_right_ctrl": "右 ⌃ コントロール",
        "hotkey_label_right_shift": "右 ⇧ シフト",
        "hotkey_select_label": "使用するキー",
        "input_monitoring_body": (
            "プッシュトゥトーク用のホットキーを検出するため、"
            "PtarmiganFlow に入力監視を許可します。"
        ),
        "input_monitoring_title": "入力監視",
        "language_chinese": "中国語",
        "language_english": "英語",
        "language_japanese": "日本語",
        "language_save_failed_message": "言語設定を保存できませんでした: {error}",
        "language_saved_message": "{path} に言語設定を保存しました。",
        "login_at_startup_button": "ログイン時に起動",
        "login_at_startup_menu": "ログイン時に起動",
        "login_disable_failed_message": "ログイン時に起動する設定を無効にできませんでした。",
        "login_disabled_message": "ログイン時に起動する設定を無効にしました。",
        "login_enable_failed_message": "ログイン時に起動する設定を有効にできませんでした。",
        "login_enabled_message": "ログイン時に起動する設定を有効にしました。",
        "microphone_body": (
            "ホットキーを押している間、PtarmiganFlow が音声を取得することを許可します。"
        ),
        "microphone_title": "マイクへのアクセス",
        "model_unavailable_message": (
            "このアプリ版では設定中のモデル {model} は利用できません。"
            "設定からモデルを選択してください。"
        ),
        "open_config_advanced_button": "設定ファイルを開く(上級者向け)",
        "open_setup_menu": "セットアップを開く…",
        "open_system_settings_button": "システム設定を開く",
        "output_clipboard_paste": "クリップボード貼り付け",
        "output_direct_typing": "直接入力",
        "quit_menu": "終了",
        "check_updates_menu": "アップデートを確認…",
        "update_available_menu": "v{version} に更新…",
        "update_checking_menu": "アップデートを確認中…",
        "update_preparing_menu": "v{version} を準備中… {percent}",
        "update_restart_menu": "再起動して v{version} に更新",
        "update_unavailable_menu": "v{version}: 自動更新できません（リリースページ）",
        "update_up_to_date_title": "最新版です",
        "update_up_to_date_body": "PtarmiganFlow {version} は最新のバージョンです。",
        "update_available_title": "PtarmiganFlow {version} が利用可能です",
        "update_ready_title": "PtarmiganFlow {version} の準備ができました",
        "update_ready_body": (
            "再起動すると v{version} に更新されます。"
            "「あとで」を選んだ場合は、次に PtarmiganFlow を終了したときに更新されます。"
        ),
        "update_dev_build_title": "開発ビルド",
        "update_dev_build_body": (
            "このビルド（{current}）は自動更新できません。最新のリリースは {version} です。"
        ),
        "update_blocked_dev_build": (
            "このビルドは自動更新できません。"
            "リリースページから新しいバージョンをダウンロードしてください。"
        ),
        "update_blocked_translocated": (
            "自動更新するには PtarmiganFlow.app "
            "を「アプリケーション」フォルダに移動してください。"
            "リリースページから新しいバージョンをダウンロードすることもできます。"
        ),
        "update_blocked_not_writable": (
            "現在の場所（{path}）では PtarmiganFlow を置き換えられません。"
            "リリースページから新しいバージョンをダウンロードしてください。"
        ),
        "update_blocked_no_asset": (
            "この Mac 用のダウンロードはまだ準備中です。しばらくしてから再度お試しいただくか、"
            "リリースページを確認してください。"
        ),
        "update_check_failed_title": "アップデートを確認できませんでした",
        "update_failed_title": "アップデートに失敗しました",
        "update_failed_body": "{error}\n\n現在のバージョンはそのまま残っています。",
        "update_restart_button": "再起動して更新",
        "update_release_notes_button": "リリースノート",
        "update_later_button": "あとで",
        "llm_ask_title": "LLM補正を使いますか？",
        "llm_ask_body": (
            "設定が「毎回確認」になっています。この起動中に音声入力の結果を "
            "LLM（{model}）で補正するか選んでください。"
        ),
        "llm_ask_use_button": "使う",
        "llm_ask_skip_button": "使わない",
        "update_ok_button": "OK",
        "restart_app_button": "アプリを再起動",
        "restart_failed_message": "アプリを再起動できませんでした。",
        "restart_required_note": (
            "システム設定で許可した直後は、反映のためアプリの再起動が必要な場合があります。"
        ),
        "route_dictionary": "変換辞書",
        "route_onboarding": "セットアップ",
        "route_settings": "設定",
        "settings_button": "設定",
        "settings_hotkey_help": "このキーを押している間だけ録音されます。",
        "settings_hotkey_label": "ホットキー",
        "settings_language_help": "音声認識に使う言語を設定します。",
        "settings_language_label": "言語",
        "settings_llm_base_url_label": "Base URL",
        "settings_llm_mode_always": "常に使う",
        "settings_llm_mode_ask": "毎回確認",
        "settings_llm_mode_label": "モード",
        "settings_llm_mode_never": "使わない",
        "settings_llm_model_label": "モデル",
        "settings_llm_provider_label": "プロバイダー",
        "settings_llm_section_title": "LLM補正",
        "settings_load_failed_message": "設定を読み込めませんでした: {error}",
        "settings_menu": "設定",
        "settings_model_download_button": "ダウンロード",
        "settings_model_download_done_message": "ダウンロード完了",
        "settings_model_busy_message": "別のダウンロードが進行中です。完了後に再試行してください。",
        "settings_model_refresh_failed_message": "モデルの状態を更新できませんでした。",
        "settings_model_download_error_message": "ダウンロードに失敗しました: {error}",
        "settings_model_downloaded_badge": "DL済み",
        "settings_model_label": "モデル",
        "settings_model_preparing_message": "ダウンロードを準備中…",
        "settings_model_retry_button": "再試行",
        "settings_models_section_title": "モデル",
        "settings_no_models_message": "このアプリ版で利用できる音声モデルがありません。",
        "settings_open_config_button": "設定ファイルを開く",
        "settings_output_mode_help": "文字起こし結果をアプリへ渡す方法を選びます。",
        "settings_output_mode_label": "出力方法",
        "save_in_progress": "保存して画面に反映中…",
        "save_retry_button": "保存を再試行",
        "save_retry_hint": "入力値やエラーの内容を確認して、保存を再試行してください。",
        "save_refresh_failed_message": "保存しましたが、画面に反映できませんでした: {error}",
        "save_error_unknown": "エラーの詳細はありません。",
        "settings_save_button": "保存",
        "settings_save_failed_message": "設定を保存できませんでした: {error}",
        "settings_saved_message": "設定を保存しました。反映には音声入力を再起動してください。",
        "settings_validation_error": "設定を保存できませんでした。確認してください: {fields}",
        "settings_window_title": "設定",
        "setup_item_accessibility": "アクセシビリティ",
        "setup_item_hotkey": "ホットキー",
        "setup_item_input_monitoring": "入力監視",
        "setup_item_language": "言語",
        "setup_item_microphone": "マイク",
        "setup_item_separator": "、",
        "setup_required_menu": "セットアップが必要です（{items}）",
        "start_dictation_button": "音声入力を開始",
        "status_granted": "許可済み",
        "status_waiting": "待機中",
        "stop_dictation_button": "音声入力を停止",
        "voice_input_started_message": "音声入力を開始しました。",
        "webui_error_title": "エラー",
    },
    "zh": {
        "accessibility_body": "允许 PtarmiganFlow 控制当前文本字段以输入语音输入结果。",
        "accessibility_title": "辅助功能访问权限",
        "allow_button": "允许",
        "all_permissions_granted_started_message": "所有权限已授予。已开始语音输入。",
        "app_setup_title": "PtarmiganFlow 设置",
        "choose_language_body": "选择用于转写的语言，并保存到配置中。",
        "choose_language_title": "选择语言",
        "config_opened_message": "已打开配置文件: {path}",
        "daemon_not_running_message": "语音输入守护进程尚未运行。",
        "daemon_start_failed_message": "无法开始语音输入: {error}",
        "dictation_running_menu": "语音输入 运行中",
        "dictation_stopped_message": "已停止语音输入。",
        "dictation_stopped_menu": "语音输入 已停止",
        "download_complete_message": "下载完成。",
        "download_failed_message": "模型下载失败: {error}",
        "download_in_progress_message": "正在下载模型... {percent}",
        "download_preparing_message": "正在准备模型...",
        "dictionary_add_exact_button": "添加完全匹配",
        "dictionary_add_regex_button": "添加正则表达式",
        "dictionary_canonical_label": "规范写法",
        "dictionary_candidates_patterns_label": "候选写法（逗号或换行分隔）",
        "dictionary_default_candidate": "候选",
        "dictionary_default_pattern": "模式",
        "dictionary_delete_button": "删除",
        "dictionary_duplicate_key_message": "规则名称重复：{key}",
        "dictionary_editor_title": "转换词典",
        "dictionary_exact_rules_description": (
            "当识别结果中出现所列写法时，替换为规则名称的写法。"
        ),
        "dictionary_exact_rules_title": "完全匹配规则",
        "dictionary_invalid_rule_message": (
            "无效的词典规则: [{section}] {key} {pattern}: {message}"
        ),
        "dictionary_load_failed_message": "无法加载转换词典: {error}",
        "dictionary_new_exact_rule": "新的完全匹配规则",
        "dictionary_new_regex_rule": "新的正则表达式规则",
        "dictionary_no_rules": "暂无规则",
        "dictionary_patterns_label": "模式（每行一个）",
        "dictionary_regex_rules_description": (
            "与正则表达式匹配的文本将替换为规则名称的写法。"
        ),
        "dictionary_regex_rules_title": "正则表达式规则",
        "dictionary_save_button": "保存",
        "dictionary_save_failed_message": "无法保存转换词典: {error}",
        "dictionary_saved_message": "已保存。请重启语音输入以应用更改。",
        "dictionary_unsaved_changes_label": "未保存的更改",
        "done_body": "设置已完成。按住所设置的热键即可语音输入，也可以调整设置。",
        "done_title": "语音输入已就绪",
        "edit_dictionary_menu": "编辑转换词典",
        "grant_permissions_message": "开始语音输入前，请授予所有权限。",
        "hotkey_capture_change_button": "更改",
        "hotkey_capture_prompt": "请按下想要设置的按键…",
        "hotkey_capture_unsupported": "不支持的按键。请按下修饰键之一（⌘、⇧、⌥、⌃）。",
        "hotkey_confirm_body": "选择语音输入时需要按住的键。之后可在设置中更改。",
        "hotkey_confirm_button": "使用此键",
        "hotkey_confirm_title": "确认要使用的热键",
        "hotkey_label_left_alt": "左 ⌥ Option",
        "hotkey_label_left_cmd": "左 ⌘ Command",
        "hotkey_label_left_ctrl": "左 ⌃ Control",
        "hotkey_label_left_shift": "左 ⇧ Shift",
        "hotkey_label_right_alt": "右 ⌥ Option",
        "hotkey_label_right_cmd": "右 ⌘ Command",
        "hotkey_label_right_ctrl": "右 ⌃ Control",
        "hotkey_label_right_shift": "右 ⇧ Shift",
        "hotkey_select_label": "使用的键",
        "input_monitoring_body": "允许 PtarmiganFlow 检测一键通话热键。",
        "input_monitoring_title": "输入监控",
        "language_chinese": "中文",
        "language_english": "英语",
        "language_japanese": "日语",
        "language_save_failed_message": "无法保存语言设置: {error}",
        "language_saved_message": "已将语言设置保存到 {path}。",
        "login_at_startup_button": "登录时启动",
        "login_at_startup_menu": "登录时启动",
        "login_disable_failed_message": "无法禁用登录时启动。",
        "login_disabled_message": "已禁用登录时启动。",
        "login_enable_failed_message": "无法启用登录时启动。",
        "login_enabled_message": "已启用登录时启动。",
        "microphone_body": "允许 PtarmiganFlow 在按住热键时采集音频。",
        "microphone_title": "麦克风访问权限",
        "model_unavailable_message": (
            "此应用版本无法使用当前设置的模型 {model}。请在设置中选择模型。"
        ),
        "open_config_advanced_button": "打开配置文件(高级)",
        "open_setup_menu": "打开初始设置…",
        "open_system_settings_button": "打开系统设置",
        "output_clipboard_paste": "剪贴板粘贴",
        "output_direct_typing": "直接输入",
        "quit_menu": "退出",
        "check_updates_menu": "检查更新…",
        "update_available_menu": "更新到 v{version}…",
        "update_checking_menu": "正在检查更新…",
        "update_preparing_menu": "正在准备 v{version}… {percent}",
        "update_restart_menu": "重新启动以更新到 v{version}",
        "update_unavailable_menu": "v{version}：无法自动更新（发布页面）",
        "update_up_to_date_title": "已是最新版本",
        "update_up_to_date_body": "PtarmiganFlow {version} 是最新版本。",
        "update_available_title": "PtarmiganFlow {version} 可用",
        "update_ready_title": "PtarmiganFlow {version} 已准备就绪",
        "update_ready_body": (
            "立即重新启动即可更新到 v{version}。如果选择“稍后”，"
            "将在下次退出 PtarmiganFlow 时安装更新。"
        ),
        "update_dev_build_title": "开发版本",
        "update_dev_build_body": "此版本（{current}）无法自动更新。最新发布版本为 {version}。",
        "update_blocked_dev_build": "此版本无法自动更新。请从发布页面下载新版本。",
        "update_blocked_translocated": (
            "请将 PtarmiganFlow.app 移动到“应用程序”文件夹以启用自动更新。"
            "也可以从发布页面下载新版本。"
        ),
        "update_blocked_not_writable": (
            "无法在当前位置（{path}）替换 PtarmiganFlow。请从发布页面下载新版本。"
        ),
        "update_blocked_no_asset": "适用于此 Mac 的下载尚未就绪。请稍后重试或查看发布页面。",
        "update_check_failed_title": "无法检查更新",
        "update_failed_title": "更新失败",
        "update_failed_body": "{error}\n\n当前版本保持不变。",
        "update_restart_button": "重新启动并更新",
        "update_release_notes_button": "发布说明",
        "update_later_button": "稍后",
        "llm_ask_title": "要使用 LLM 校正吗？",
        "llm_ask_body": (
            "LLM 校正设置为“每次询问”。请选择在本次运行期间是否使用 LLM（{model}）"
            "校正语音输入结果。"
        ),
        "llm_ask_use_button": "使用",
        "llm_ask_skip_button": "不使用",
        "update_ok_button": "好",
        "restart_app_button": "重新启动应用",
        "restart_failed_message": "无法重新启动应用。",
        "restart_required_note": (
            "如果刚刚在系统设置中授予了此权限，可能需要重新启动应用才能生效。"
        ),
        "route_dictionary": "转换词典",
        "route_onboarding": "设置",
        "route_settings": "设置",
        "settings_button": "设置",
        "settings_hotkey_help": "仅在按住此键期间进行录音。",
        "settings_hotkey_label": "热键",
        "settings_language_help": "设置用于语音转写的语言。",
        "settings_language_label": "语言",
        "settings_llm_base_url_label": "Base URL",
        "settings_llm_mode_always": "始终使用",
        "settings_llm_mode_ask": "每次询问",
        "settings_llm_mode_label": "模式",
        "settings_llm_mode_never": "不使用",
        "settings_llm_model_label": "模型",
        "settings_llm_provider_label": "提供方",
        "settings_llm_section_title": "LLM 修正",
        "settings_load_failed_message": "无法加载设置: {error}",
        "settings_menu": "设置",
        "settings_model_download_button": "下载",
        "settings_model_download_done_message": "下载完成",
        "settings_model_busy_message": "另一个下载正在进行中。请在完成后重试。",
        "settings_model_refresh_failed_message": "无法刷新模型状态。",
        "settings_model_download_error_message": "下载失败：{error}",
        "settings_model_downloaded_badge": "已下载",
        "settings_model_label": "模型",
        "settings_model_preparing_message": "正在准备下载…",
        "settings_model_retry_button": "重试",
        "settings_models_section_title": "模型",
        "settings_no_models_message": "此应用版本没有可用的语音模型。",
        "settings_open_config_button": "打开配置文件",
        "settings_output_mode_help": "选择将转写文本传递给当前应用的方式。",
        "settings_output_mode_label": "输出方式",
        "save_in_progress": "正在保存并更新界面…",
        "save_retry_button": "重试保存",
        "save_retry_hint": "请检查输入值或错误详情，然后重试保存。",
        "save_refresh_failed_message": "已保存，但无法更新界面: {error}",
        "save_error_unknown": "未提供错误详情。",
        "settings_save_button": "保存",
        "settings_save_failed_message": "无法保存设置: {error}",
        "settings_saved_message": "设置已保存。请重启语音输入以应用更改。",
        "settings_validation_error": "无法保存设置。请检查: {fields}",
        "settings_window_title": "设置",
        "setup_item_accessibility": "辅助功能",
        "setup_item_hotkey": "热键",
        "setup_item_input_monitoring": "输入监控",
        "setup_item_language": "语言",
        "setup_item_microphone": "麦克风",
        "setup_item_separator": "、",
        "setup_required_menu": "需要完成初始设置（{items}）",
        "start_dictation_button": "开始语音输入",
        "status_granted": "已允许",
        "status_waiting": "等待中",
        "stop_dictation_button": "停止语音输入",
        "voice_input_started_message": "已开始语音输入。",
        "webui_error_title": "错误",
    },
}


def strings_for(lang: str) -> dict[str, str]:
    """Return app strings for ``lang`` with English fallback per key."""
    english = ONBOARDING_STRINGS["en"]
    localized = ONBOARDING_STRINGS.get(lang.lower(), english)
    return {key: localized.get(key, english[key]) for key in english}
