from __future__ import annotations

from ptarmigan_flow.web_ui import choose_visible_frame, window_origin_for_display

# Main 1512x982 display with a 25pt menu bar, and a secondary display to its left.
MAIN = ((0.0, 0.0, 1512.0, 982.0), (0.0, 0.0, 1512.0, 957.0))
LEFT = ((-1920.0, 0.0, 1920.0, 1080.0), (-1920.0, 0.0, 1920.0, 1055.0))
SCREENS = [MAIN, LEFT]
WINDOW_SIZE = (860.0, 640.0)


def test_choose_visible_frame_picks_screen_under_mouse() -> None:
    assert choose_visible_frame((-500.0, 300.0), SCREENS, MAIN) == LEFT[1]
    assert choose_visible_frame((700.0, 300.0), SCREENS, LEFT) == MAIN[1]


def test_choose_visible_frame_falls_back_to_main_screen() -> None:
    assert choose_visible_frame((99999.0, 99999.0), SCREENS, MAIN) == MAIN[1]
    assert choose_visible_frame((99999.0, 99999.0), SCREENS, None) == MAIN[1]
    assert choose_visible_frame((0.0, 0.0), [], None) is None


def test_window_on_other_screen_is_centered_on_mouse_screen() -> None:
    window = (-1390.0, 220.0, *WINDOW_SIZE)

    origin = window_origin_for_display(window, (700.0, 400.0), SCREENS, MAIN)

    assert origin == ((1512.0 - 860.0) / 2, (957.0 - 640.0) / 2)


def test_window_moves_to_secondary_screen_when_mouse_is_there() -> None:
    window = (326.0, 158.0, *WINDOW_SIZE)

    origin = window_origin_for_display(window, (-100.0, 50.0), SCREENS, MAIN)

    assert origin == (-1920.0 + (1920.0 - 860.0) / 2, (1055.0 - 640.0) / 2)


def test_window_already_on_mouse_screen_is_left_alone() -> None:
    window = (10.0, 10.0, *WINDOW_SIZE)

    assert window_origin_for_display(window, (1000.0, 900.0), SCREENS, MAIN) is None


def test_window_partially_off_mouse_screen_is_recentered() -> None:
    window = (1000.0, 100.0, *WINDOW_SIZE)

    assert window_origin_for_display(window, (100.0, 100.0), SCREENS, MAIN) == (
        (1512.0 - 860.0) / 2,
        (957.0 - 640.0) / 2,
    )


def test_window_larger_than_screen_keeps_title_bar_visible() -> None:
    small = ((0.0, 0.0, 800.0, 600.0), (0.0, 0.0, 800.0, 575.0))
    window = (-2000.0, 0.0, *WINDOW_SIZE)

    origin = window_origin_for_display(window, (10.0, 10.0), [small], small)

    assert origin == (0.0, 575.0 - 640.0)
