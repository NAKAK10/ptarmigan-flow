from moonshine_flow.domain.transcription_session import append_only_delta


def test_append_only_delta_appends_when_strict_prefix() -> None:
    assert append_only_delta("hello", "hello world") == " world"


def test_append_only_delta_tolerates_small_non_monotonic_tail() -> None:
    # previous tail "います" rewritten to "ってください"
    previous = "同じ情報が2度入力される場合があるのでその対策を行います"
    current = "同じ情報が2度入力される場合があるのでその対策を行ってください"
    assert append_only_delta(previous, current) == "ってください"


def test_append_only_delta_keeps_phrase_overlap_without_aggressive_trim() -> None:
    previous = "同じ情報が2度入力される場合があるのでその対策を行います"
    current = (
        "同じ情報が2度入力される場合があるのでその対策を行います"
        "場合があるのでその対策を行ってください"
    )
    assert append_only_delta(previous, current) == "場合があるのでその対策を行ってください"


def test_append_only_delta_keeps_short_overlap() -> None:
    previous = "alphabet"
    current = "alphabetbetamax"
    # short overlap("bet") is below dedup threshold and should not be trimmed.
    assert append_only_delta(previous, current) == "betamax"
