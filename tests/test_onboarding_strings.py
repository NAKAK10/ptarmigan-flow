from __future__ import annotations

from ptarmigan_flow import onboarding_strings

REQUIRED_KEYS = {
    "accessibility_body",
    "accessibility_title",
    "allow_button",
    "all_permissions_granted_started_message",
    "app_setup_title",
    "choose_language_body",
    "choose_language_title",
    "config_opened_message",
    "daemon_not_running_message",
    "daemon_start_failed_message",
    "dictation_running_menu",
    "dictation_stopped_message",
    "dictation_stopped_menu",
    "download_complete_message",
    "download_failed_message",
    "download_in_progress_message",
    "download_preparing_message",
    "dictionary_add_exact_button",
    "dictionary_add_regex_button",
    "dictionary_canonical_label",
    "dictionary_candidates_patterns_label",
    "dictionary_default_candidate",
    "dictionary_default_pattern",
    "dictionary_delete_button",
    "dictionary_editor_title",
    "dictionary_exact_rules_title",
    "dictionary_invalid_rule_message",
    "dictionary_load_failed_message",
    "dictionary_new_exact_rule",
    "dictionary_new_regex_rule",
    "dictionary_no_rules",
    "dictionary_regex_rules_title",
    "dictionary_save_button",
    "dictionary_save_failed_message",
    "dictionary_saved_message",
    "done_body",
    "done_title",
    "edit_dictionary_menu",
    "grant_permissions_message",
    "input_monitoring_body",
    "input_monitoring_title",
    "language_chinese",
    "language_english",
    "language_japanese",
    "language_save_failed_message",
    "language_saved_message",
    "login_at_startup_button",
    "login_at_startup_menu",
    "login_disable_failed_message",
    "login_disabled_message",
    "login_enable_failed_message",
    "login_enabled_message",
    "microphone_body",
    "microphone_title",
    "model_unavailable_message",
    "open_config_advanced_button",
    "open_system_settings_button",
    "output_clipboard_paste",
    "output_direct_typing",
    "quit_menu",
    "restart_app_button",
    "restart_failed_message",
    "restart_required_note",
    "settings_button",
    "settings_hotkey_label",
    "settings_language_label",
    "settings_load_failed_message",
    "settings_menu",
    "settings_model_label",
    "settings_no_models_message",
    "settings_output_mode_label",
    "settings_save_button",
    "settings_save_failed_message",
    "settings_saved_message",
    "settings_validation_error",
    "settings_window_title",
    "start_dictation_button",
    "stop_dictation_button",
    "voice_input_started_message",
}


def test_supported_languages_have_identical_required_key_sets() -> None:
    key_sets = {
        language: set(onboarding_strings.strings_for(language))
        for language in ("en", "ja", "zh")
    }

    assert key_sets["en"] == REQUIRED_KEYS
    assert key_sets["ja"] == REQUIRED_KEYS
    assert key_sets["zh"] == REQUIRED_KEYS


def test_strings_for_unknown_language_falls_back_to_english() -> None:
    assert onboarding_strings.strings_for("de") == onboarding_strings.strings_for("en")


def test_strings_for_missing_key_falls_back_to_english(monkeypatch) -> None:
    localized = {
        language: dict(onboarding_strings.ONBOARDING_STRINGS[language])
        for language in ("en", "ja", "zh")
    }
    del localized["ja"]["restart_required_note"]
    monkeypatch.setattr(onboarding_strings, "ONBOARDING_STRINGS", localized)

    strings = onboarding_strings.strings_for("ja")

    assert strings["restart_required_note"] == localized["en"]["restart_required_note"]
    assert strings["app_setup_title"] == "PtarmiganFlow セットアップ"


def test_required_japanese_translations_are_exact() -> None:
    strings = onboarding_strings.strings_for("ja")

    assert strings["start_dictation_button"] == "音声入力を開始"
    assert strings["stop_dictation_button"] == "音声入力を停止"
    assert strings["dictation_running_menu"] == "音声入力 実行中"
    assert strings["dictation_stopped_menu"] == "音声入力 停止中"
    assert strings["settings_menu"] == "設定"
    assert strings["edit_dictionary_menu"] == "変換辞書を編集"
    assert strings["quit_menu"] == "終了"
    assert strings["grant_permissions_message"] == (
        "音声入力を開始する前にすべての権限を許可してください。"
    )
    assert strings["daemon_not_running_message"] == "音声入力デーモンはまだ起動していません。"
    assert strings["daemon_start_failed_message"] == "音声入力を開始できませんでした: {error}"
    assert strings["download_preparing_message"] == "モデルを準備中..."
    assert strings["download_in_progress_message"] == "モデルをダウンロード中... {percent}"
    assert strings["download_complete_message"] == "ダウンロードが完了しました。"
    assert strings["download_failed_message"] == (
        "モデルのダウンロードに失敗しました: {error}"
    )
    assert strings["voice_input_started_message"] == "音声入力を開始しました。"
    assert strings["all_permissions_granted_started_message"] == (
        "すべての権限が許可されました。音声入力を開始しました。"
    )
    assert strings["dictation_stopped_message"] == "音声入力を停止しました。"
    assert strings["restart_failed_message"] == "アプリを再起動できませんでした。"
    assert strings["language_saved_message"] == "{path} に言語設定を保存しました。"
    assert strings["language_save_failed_message"] == "言語設定を保存できませんでした: {error}"
    assert strings["config_opened_message"] == "設定ファイルを開きました: {path}"
    assert strings["login_at_startup_menu"] == "ログイン時に起動"
    assert strings["login_enabled_message"] == "ログイン時に起動する設定を有効にしました。"
    assert strings["login_disabled_message"] == "ログイン時に起動する設定を無効にしました。"
    assert strings["login_enable_failed_message"] == (
        "ログイン時に起動する設定を有効にできませんでした。"
    )
    assert strings["login_disable_failed_message"] == (
        "ログイン時に起動する設定を無効にできませんでした。"
    )
    assert strings["dictionary_editor_title"] == "変換辞書"
    assert strings["dictionary_exact_rules_title"] == "完全一致ルール"
    assert strings["dictionary_regex_rules_title"] == "正規表現ルール"
    assert strings["dictionary_canonical_label"] == "正規表記"
    assert strings["dictionary_candidates_patterns_label"] == "候補/パターン(カンマ区切り)"
    assert strings["dictionary_no_rules"] == "ルールがありません"
    assert strings["dictionary_add_exact_button"] == "完全一致を追加"
    assert strings["dictionary_add_regex_button"] == "正規表現を追加"
    assert strings["dictionary_save_button"] == "保存"
    assert strings["dictionary_delete_button"] == "削除"
    assert strings["dictionary_saved_message"] == (
        "保存しました。反映には音声入力を再起動してください。"
    )
    assert strings["dictionary_load_failed_message"] == "変換辞書を読み込めませんでした: {error}"
    assert strings["dictionary_save_failed_message"] == "変換辞書を保存できませんでした: {error}"
    assert strings["dictionary_invalid_rule_message"] == (
        "無効な辞書ルール: [{section}] {key} {pattern}: {message}"
    )
    assert strings["model_unavailable_message"] == (
        "このアプリ版では設定中のモデル {model} は利用できません。"
        "設定からモデルを選択してください。"
    )


def test_chinese_translations_are_simplified_and_complete() -> None:
    strings = onboarding_strings.strings_for("zh")

    assert strings["start_dictation_button"] == "开始语音输入"
    assert strings["stop_dictation_button"] == "停止语音输入"
    assert strings["dictation_running_menu"] == "语音输入 运行中"
    assert strings["dictation_stopped_menu"] == "语音输入 已停止"
    assert strings["settings_menu"] == "设置"
    assert strings["edit_dictionary_menu"] == "编辑转换词典"
    assert strings["dictionary_editor_title"] == "转换词典"
    assert strings["dictionary_saved_message"] == "已保存。请重启语音输入以应用更改。"
    assert strings["download_preparing_message"] == "正在准备模型..."
    assert strings["download_in_progress_message"] == "正在下载模型... {percent}"
    assert strings["download_complete_message"] == "下载完成。"
    assert strings["download_failed_message"] == "模型下载失败: {error}"
    assert strings["model_unavailable_message"] == (
        "此应用版本无法使用当前设置的模型 {model}。请在设置中选择模型。"
    )
