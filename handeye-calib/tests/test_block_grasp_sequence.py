import pytest

from block_grasp_sequence import run_block_sequence


def callbacks(calls, failures=None):
    failures = failures or {}

    def action(name):
        def invoke():
            calls.append(name)
            if name in failures:
                raise failures[name]
        return invoke

    return {
        "confirm_pump_off": action("pump_off"),
        "move_pre": action("move_pre"),
        "move_contact": action("move_contact"),
        "pump_on": action("pump_on"),
        "retreat": action("retreat"),
    }


def run(calls, dry_run=False, stop_at_pre_grasp=False, failures=None):
    actions = callbacks(calls, failures)
    return run_block_sequence(
        dry_run=dry_run,
        stop_at_pre_grasp=stop_at_pre_grasp,
        log=lambda message: calls.append(("log", message)),
        **actions
    )


def action_calls(calls):
    return [item for item in calls if isinstance(item, str)]


def log_messages(calls):
    return [item[1] for item in calls if isinstance(item, tuple)]


def test_dry_run_calls_no_motion_or_pump():
    calls = []
    assert run(calls, dry_run=True) == "dry_run"
    assert action_calls(calls) == []


def test_stop_at_pre_grasp_moves_once_without_pump_or_contact():
    calls = []
    assert run(calls, stop_at_pre_grasp=True) == "pre_grasp"
    assert action_calls(calls) == ["move_pre"]


def test_full_grasp_has_one_strict_action_order_and_leaves_pump_on():
    calls = []
    assert run(calls) == "grasped"
    assert action_calls(calls) == [
        "move_pre", "pump_off", "move_contact", "pump_on", "retreat"
    ]


def test_pre_grasp_failure_stops_without_pump_state_warning():
    calls = []
    error = RuntimeError("pre failed")
    with pytest.raises(RuntimeError, match="pre failed"):
        run(calls, failures={"move_pre": error})
    assert action_calls(calls) == ["move_pre"]
    assert not any("UNKNOWN" in message for message in log_messages(calls))


def test_confirm_pump_off_exception_reports_unknown_and_aborts_before_contact():
    calls = []
    error = RuntimeError("off response timeout")
    with pytest.raises(RuntimeError, match="off response timeout"):
        run(calls, failures={"pump_off": error})
    assert action_calls(calls) == ["move_pre", "pump_off"]
    messages = log_messages(calls)
    assert any("UNKNOWN" in message for message in messages)
    assert any("contact" in message.lower() for message in messages)


def test_contact_failure_does_not_attempt_pump_or_extra_motion():
    calls = []
    error = RuntimeError("contact failed")
    with pytest.raises(RuntimeError, match="contact failed"):
        run(calls, failures={"move_contact": error})
    assert action_calls(calls) == ["move_pre", "pump_off", "move_contact"]


def test_pump_on_exception_reports_unknown_possibly_on_and_does_not_retreat():
    calls = []
    error = RuntimeError("on response timeout")
    with pytest.raises(RuntimeError, match="on response timeout"):
        run(calls, failures={"pump_on": error})
    assert action_calls(calls) == ["move_pre", "pump_off", "move_contact", "pump_on"]
    messages = " ".join(log_messages(calls))
    assert "UNKNOWN" in messages
    assert "may be ON" in messages


def test_retreat_failure_is_critical_and_sends_no_recovery_action():
    calls = []
    error = RuntimeError("retreat failed")
    with pytest.raises(RuntimeError, match="retreat failed"):
        run(calls, failures={"retreat": error})
    assert action_calls(calls) == [
        "move_pre", "pump_off", "move_contact", "pump_on", "retreat"
    ]
    message = " ".join(log_messages(calls))
    assert "CRITICAL" in message
    assert "pump may remain ON" in message
