from __future__ import annotations

from ptarmigan_flow import onboarding_strings

REQUIRED_KEYS = {
    "app_setup_title",
    "choose_language_title",
    "choose_language_body",
    "microphone_title",
    "microphone_body",
    "accessibility_title",
    "accessibility_body",
    "input_monitoring_title",
    "input_monitoring_body",
    "allow_button",
    "open_system_settings_button",
    "restart_app_button",
    "restart_required_note",
    "done_title",
    "done_body",
    "start_dictation_button",
    "stop_dictation_button",
    "open_config_button",
    "login_at_startup_button",
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

    assert strings["app_setup_title"] == "PtarmiganFlow セットアップ"
    assert strings["choose_language_title"] == "言語を選択"
    assert strings["choose_language_body"] == "文字起こしに使う言語を選択して設定に保存します。"
    assert strings["microphone_title"] == "マイクへのアクセス"
    assert strings["microphone_body"] == (
        "ホットキーを押している間、PtarmiganFlow が音声を取得することを許可します。"
    )
    assert strings["accessibility_title"] == "アクセシビリティへのアクセス"
    assert strings["accessibility_body"] == (
        "ディクテーション結果を入力するため、PtarmiganFlow が"
        "アクティブなテキスト欄を操作することを許可します。"
    )
    assert strings["input_monitoring_title"] == "入力監視"
    assert strings["input_monitoring_body"] == (
        "プッシュトゥトーク用のホットキーを検出するため、"
        "PtarmiganFlow に入力監視を許可します。"
    )
    assert strings["allow_button"] == "許可する"
    assert strings["open_system_settings_button"] == "システム設定を開く"
    assert strings["restart_app_button"] == "アプリを再起動"
    assert strings["restart_required_note"] == (
        "システム設定で許可した直後は、反映のためアプリの再起動が必要な場合があります。"
    )
    assert strings["done_title"] == "ディクテーションの準備完了"
    assert strings["done_body"] == (
        "セットアップが完了しました。ディクテーションを開始するか、設定ファイルを開けます。"
    )
    assert strings["start_dictation_button"] == "ディクテーション開始"
    assert strings["stop_dictation_button"] == "ディクテーション停止"
    assert strings["open_config_button"] == "設定を開く"
    assert strings["login_at_startup_button"] == "ログイン時に起動"


def test_chinese_translations_are_simplified_and_complete() -> None:
    strings = onboarding_strings.strings_for("zh")

    assert strings["app_setup_title"] == "PtarmiganFlow 设置"
    assert strings["choose_language_title"] == "选择语言"
    assert strings["microphone_title"] == "麦克风访问权限"
    assert strings["accessibility_title"] == "辅助功能访问权限"
    assert strings["input_monitoring_title"] == "输入监控"
    assert strings["restart_app_button"] == "重新启动应用"
    assert strings["restart_required_note"] == (
        "如果刚刚在系统设置中授予了此权限，可能需要重新启动应用才能生效。"
    )
