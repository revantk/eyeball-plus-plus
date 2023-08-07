import pytest
from datetime import datetime, timedelta
from eyeball_pp.system_state import SystemState, get_recent_system_states


def test_get_recent_system_states_no_args():
    """
    Test get_recent_system_states with no arguments.
    The function should return system states for the last 14 days.
    """
    system_states = get_recent_system_states()
    assert len(system_states) == 14 // 2 + 1 if 14 % 2 != 0 else 14 // 2
    assert all((isinstance(state, SystemState) for state in system_states))
    assert all((state.start_time <= state.end_time for state in system_states))


def test_get_recent_system_states_with_since_till():
    """
    Test get_recent_system_states with since and till arguments.
    The function should return system states that cover the time period between since and till.
    """
    since = datetime.now() - timedelta(days=5)
    till = datetime.now()
    system_states = get_recent_system_states(since=since, till=till)
    assert len(system_states) == 5 // 2 + 1 if 5 % 2 != 0 else 5 // 2
    assert all((isinstance(state, SystemState) for state in system_states))
    assert all((state.start_time <= state.end_time for state in system_states))


def test_get_recent_system_states_with_rerun_ids():
    """
    Test get_recent_system_states with rerun_ids argument.
    The function should return system states that cover the time period of the rerun_ids.
    """
    rerun_ids = [
        datetime.now().isoformat(),
        (datetime.now() - timedelta(days=1)).isoformat(),
    ]
    system_states = get_recent_system_states(rerun_ids=rerun_ids)
    assert (
        len(system_states) == 14 // 2 + 1
        if 14 % 2 != 0
        else 14 // 2 + 2 * len(rerun_ids)
    )
    assert all((isinstance(state, SystemState) for state in system_states))
    assert all((state.start_time <= state.end_time for state in system_states))


def test_get_recent_system_states_with_since_till_rerun_ids():
    """
    Test get_recent_system_states with since, till, and rerun_ids arguments.
    The function should return system states that cover the time period between since and till and the time period of the rerun_ids.
    """
    since = datetime.now() - timedelta(days=5)
    till = datetime.now()
    rerun_ids = [
        (datetime.now() - timedelta(days=2)).isoformat(),
        (datetime.now() - timedelta(days=3)).isoformat(),
    ]
    system_states = get_recent_system_states(
        since=since, till=till, rerun_ids=rerun_ids
    )
    assert (
        len(system_states) == 5 // 2 + 1 if 5 % 2 != 0 else 5 // 2 + 2 * len(rerun_ids)
    )
    assert all((isinstance(state, SystemState) for state in system_states))
    assert all((state.start_time <= state.end_time for state in system_states))
