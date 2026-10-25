"""Focused tests for version-one fixed asset state."""

import numpy as np

from factor_sim.config import InvalidReturnPolicy, resolve_policy
from factor_sim.models import SimulationPolicy
from factor_sim.pricing import (
    apply_corporate_action_policy,
    apply_new_listing_policy,
)


def test_none_handlers_preserve_returns_and_universe() -> None:
    policy = resolve_policy(
        SimulationPolicy(invalid_return_policy=InvalidReturnPolicy.FAIL)
    )
    returns = np.zeros((1, 1, 2))
    asset_ids = ("a", "b")

    adjusted, events, _ = apply_corporate_action_policy(returns, resolved_policy=policy)
    retained, _ = apply_new_listing_policy(asset_ids, resolved_policy=policy)

    assert adjusted is returns
    assert events == ()
    assert retained is asset_ids
