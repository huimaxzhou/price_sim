"""Numerical covariance-root tests for implementation-plan Step 5."""

from __future__ import annotations

import numpy as np
import pytest

from factor_sim.errors import NumericalError
from factor_sim.sampling import covariance_root

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0.04, 0.01], [0.01, 0.09]]),
        np.array([[0.04, 0.04], [0.04, 0.04]]),
    ],
)
def test_covariance_root_reconstructs_positive_and_singular_psd(
    matrix: np.ndarray,
) -> None:
    original = matrix.copy()
    root, diagnostics = covariance_root(
        matrix,
        matrix_name="factor_covariance",
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )

    np.testing.assert_allclose(root @ root.T, matrix, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(matrix, original)
    assert diagnostics["method"] == "symmetric_eigendecomposition"


def test_tolerance_bound_negative_eigenvalue_is_zeroed_and_reported() -> None:
    matrix = np.diag([0.04, -5e-13])
    root, diagnostics = covariance_root(
        matrix,
        matrix_name="near_psd",
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )

    np.testing.assert_allclose(root @ root.T, np.diag([0.04, 0.0]), atol=1e-15)
    assert diagnostics["minimum_eigenvalue"] == pytest.approx(-5e-13)
    assert diagnostics["zeroed_eigenvalue_count"] == 1
    assert diagnostics["zeroed_eigenvalue_minimum"] == pytest.approx(-5e-13)


def test_small_asymmetry_is_canonicalized_without_mutating_input() -> None:
    matrix = np.array([[0.04, 0.01 + 2e-13], [0.01, 0.09]])
    original = matrix.copy()
    canonical = (matrix + matrix.T) / 2.0
    root, diagnostics = covariance_root(
        matrix,
        matrix_name="near_symmetric",
        symmetry_tolerance=1e-12,
        positive_semidefinite_tolerance=1e-12,
    )

    np.testing.assert_allclose(root @ root.T, canonical, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(matrix, original)
    assert diagnostics["canonicalized"] is True
    assert diagnostics["maximum_asymmetry"] == pytest.approx(2e-13)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "indefinite"),
        (np.array([[1.0, 1e-3], [0.0, 1.0]]), "asymmetric"),
    ],
)
def test_material_numerical_failures_are_not_repaired(
    matrix: np.ndarray, message: str
) -> None:
    original = matrix.copy()
    with pytest.raises(NumericalError, match=message):
        covariance_root(
            matrix,
            matrix_name="factor_covariance",
            symmetry_tolerance=1e-12,
            positive_semidefinite_tolerance=1e-12,
        )
    np.testing.assert_array_equal(matrix, original)
