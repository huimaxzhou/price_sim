"""Focused serialization failure-policy tests."""

import dataclasses
from collections.abc import Callable

from factor_sim.engine import simulate
from factor_sim.io import save_result


def test_save_refuses_existing_directory(
    request_factory: Callable[..., object], tmp_path: object
) -> None:
    request = request_factory()
    result = simulate(request)
    target = tmp_path / "existing"
    target.mkdir()

    try:
        save_result(result, target)
    except FileExistsError:
        pass
    else:
        raise AssertionError("save_result must refuse an existing target")


def test_save_rejects_inconsistent_events(
    request_factory: Callable[..., object], tmp_path: object
) -> None:
    result = simulate(request_factory())
    event_type = type(result.asset_state_events)
    assert event_type is tuple
    altered = dataclasses.replace(result, asset_state_events=(object(),))

    try:
        save_result(altered, tmp_path / "run")
    except ValueError:
        pass
    else:
        raise AssertionError("events under None handlers must be rejected")
