"""Localized strings for the native macOS onboarding wizard."""

from __future__ import annotations

ONBOARDING_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "app_setup_title": "PtarmiganFlow setup",
        "choose_language_title": "Choose Language",
        "choose_language_body": "Select the transcription language to save into your config.",
        "microphone_title": "Microphone Access",
        "microphone_body": "Allow PtarmiganFlow to capture audio while you hold the hotkey.",
        "accessibility_title": "Accessibility Access",
        "accessibility_body": (
            "Allow PtarmiganFlow to control the active text field for dictation output."
        ),
        "input_monitoring_title": "Input Monitoring",
        "input_monitoring_body": "Allow PtarmiganFlow to detect the push-to-talk hotkey.",
        "allow_button": "Allow",
        "open_system_settings_button": "Open System Settings",
        "restart_app_button": "Restart App",
        "restart_required_note": (
            "If you just granted this in System Settings, you may need to restart the app to "
            "apply it."
        ),
        "done_title": "Ready to Dictate",
        "done_body": "Setup is complete. Start dictation now or open the config file.",
        "start_dictation_button": "Start Dictation",
        "stop_dictation_button": "Stop Dictation",
        "open_config_button": "Open Config",
        "login_at_startup_button": "Login at Startup",
    },
    "ja": {
        "app_setup_title": "PtarmiganFlow セットアップ",
        "choose_language_title": "言語を選択",
        "choose_language_body": "文字起こしに使う言語を選択して設定に保存します。",
        "microphone_title": "マイクへのアクセス",
        "microphone_body": (
            "ホットキーを押している間、PtarmiganFlow が音声を取得することを許可します。"
        ),
        "accessibility_title": "アクセシビリティへのアクセス",
        "accessibility_body": (
            "ディクテーション結果を入力するため、PtarmiganFlow が"
            "アクティブなテキスト欄を操作することを許可します。"
        ),
        "input_monitoring_title": "入力監視",
        "input_monitoring_body": (
            "プッシュトゥトーク用のホットキーを検出するため、"
            "PtarmiganFlow に入力監視を許可します。"
        ),
        "allow_button": "許可する",
        "open_system_settings_button": "システム設定を開く",
        "restart_app_button": "アプリを再起動",
        "restart_required_note": (
            "システム設定で許可した直後は、反映のためアプリの再起動が必要な場合があります。"
        ),
        "done_title": "ディクテーションの準備完了",
        "done_body": (
            "セットアップが完了しました。ディクテーションを開始するか、設定ファイルを開けます。"
        ),
        "start_dictation_button": "ディクテーション開始",
        "stop_dictation_button": "ディクテーション停止",
        "open_config_button": "設定を開く",
        "login_at_startup_button": "ログイン時に起動",
    },
    "zh": {
        "app_setup_title": "PtarmiganFlow 设置",
        "choose_language_title": "选择语言",
        "choose_language_body": "选择用于转写的语言，并保存到配置中。",
        "microphone_title": "麦克风访问权限",
        "microphone_body": "允许 PtarmiganFlow 在按住热键时采集音频。",
        "accessibility_title": "辅助功能访问权限",
        "accessibility_body": "允许 PtarmiganFlow 控制当前文本字段以输入听写结果。",
        "input_monitoring_title": "输入监控",
        "input_monitoring_body": "允许 PtarmiganFlow 检测一键通话热键。",
        "allow_button": "允许",
        "open_system_settings_button": "打开系统设置",
        "restart_app_button": "重新启动应用",
        "restart_required_note": (
            "如果刚刚在系统设置中授予了此权限，可能需要重新启动应用才能生效。"
        ),
        "done_title": "已准备好听写",
        "done_body": "设置已完成。现在可以开始听写或打开配置文件。",
        "start_dictation_button": "开始听写",
        "stop_dictation_button": "停止听写",
        "open_config_button": "打开配置",
        "login_at_startup_button": "登录时启动",
    },
}


def strings_for(lang: str) -> dict[str, str]:
    """Return onboarding strings for ``lang`` with English fallback per key."""
    english = ONBOARDING_STRINGS["en"]
    localized = ONBOARDING_STRINGS.get(lang.lower(), english)
    return {key: localized.get(key, english[key]) for key in english}
