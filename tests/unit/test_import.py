"""Bootstrap smoke tests."""

import factor_sim
import factor_sim.config
import factor_sim.diagnostics
import factor_sim.engine
import factor_sim.errors
import factor_sim.io
import factor_sim.models
import factor_sim.pricing
import factor_sim.sampling
import factor_sim.validation


def test_package_skeleton_imports() -> None:
    """Every planned module imports without exposing unfinished root APIs."""

    assert factor_sim.__all__ == ()
