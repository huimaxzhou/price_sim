"""Return mapping, version-one handlers, and arithmetic price evolution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import InvalidReturnPolicy, PriceReturnConvention
from .errors import NumericalError, PolicyExecutionError
from .models import AssetStateEvent, DateLike, ResolvedSimulationPolicy, RiskState

FloatArray = NDArray[np.float64]


def _owned_float64(value: object) -> FloatArray:
    return np.array(value, dtype=np.float64, copy=True, order="C")


def map_modeled_returns(
    modeled_returns: FloatArray,
    *,
    risk_state: RiskState,
    resolved_policy: ResolvedSimulationPolicy,
) -> tuple[FloatArray, Mapping[str, Any]]:
    """Apply the supported raw arithmetic identity mapping."""

    if modeled_returns.ndim != 3:
        raise NumericalError("modeled_returns must have shape (M, H, N)")
    compatible = (
        resolved_policy.price_return_convention is PriceReturnConvention.RAW
        and risk_state.return_convention == "arithmetic"
        and risk_state.return_basis == "raw_price"
    )
    if not compatible:
        raise PolicyExecutionError(
            "raw identity mapping requires raw arithmetic price returns",
            code="incompatible_return_mapping",
            context={
                "return_convention": risk_state.return_convention,
                "return_basis": risk_state.return_basis,
                "price_return_convention": str(resolved_policy.price_return_convention),
            },
        )
    mapped = _owned_float64(modeled_returns)
    return mapped, {
        "mapping": "identity",
        "return_type": "arithmetic",
        "return_basis": "raw_price",
        "adjustment": "unadjusted",
    }


def apply_currency_policy(
    price_link_returns: FloatArray,
    *,
    resolved_policy: ResolvedSimulationPolicy,
    asset_currencies: tuple[str, ...],
) -> tuple[FloatArray, Mapping[str, Any]]:
    """Apply the explicit version-one no-conversion policy."""

    if resolved_policy.currency_handling is not None:
        raise PolicyExecutionError(
            "version one supports only currency_handling=None",
            code="unsupported_currency_handler",
        )
    return price_link_returns, {
        "handler": None,
        "converted": False,
        "price_currencies": asset_currencies,
    }


def apply_corporate_action_policy(
    price_link_returns: FloatArray,
    *,
    resolved_policy: ResolvedSimulationPolicy,
) -> tuple[FloatArray, tuple[AssetStateEvent, ...], Mapping[str, Any]]:
    """Apply the explicit version-one no-corporate-action policy."""

    if resolved_policy.corporate_action_adjustment is not None:
        raise PolicyExecutionError(
            "version one supports only corporate_action_adjustment=None",
            code="unsupported_corporate_action_handler",
        )
    return price_link_returns, (), {"handler": None, "adjusted": False}


def apply_new_listing_policy(
    asset_ids: tuple[str, ...],
    *,
    resolved_policy: ResolvedSimulationPolicy,
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    """Apply the explicit version-one fixed-universe policy."""

    if resolved_policy.new_listing is not None:
        raise PolicyExecutionError(
            "version one supports only new_listing=None",
            code="unsupported_new_listing_handler",
        )
    return asset_ids, {"handler": None, "universe_changed": False}


def _location(
    *,
    path_id: int,
    step_index: int,
    asset_index: int,
    asset_ids: tuple[str, ...],
    future_dates: tuple[DateLike, ...],
) -> dict[str, Any]:
    return {
        "path_id": path_id,
        "step_index": step_index,
        "future_date": future_dates[step_index],
        "asset_id": asset_ids[asset_index],
    }


def evolve_arithmetic_prices(
    initial_prices: FloatArray,
    price_link_returns: FloatArray,
    *,
    asset_ids: tuple[str, ...],
    future_dates: tuple[DateLike, ...],
    invalid_return_policy: InvalidReturnPolicy,
) -> FloatArray:
    """Compound arithmetic returns, looping over dates only."""

    if initial_prices.ndim != 1 or price_link_returns.ndim != 3:
        raise NumericalError(
            "initial_prices and price_link_returns must have shapes (N,) and (M,H,N)"
        )
    path_count, step_count, asset_count = price_link_returns.shape
    if initial_prices.shape != (asset_count,):
        raise NumericalError("initial_prices shape must match the return asset axis")
    if len(asset_ids) != asset_count:
        raise NumericalError("asset_ids length must match the return asset axis")
    if len(future_dates) != step_count:
        raise NumericalError("future_dates length must match the return date axis")
    if invalid_return_policy is not InvalidReturnPolicy.FAIL:
        raise PolicyExecutionError(
            "version one supports only the fail invalid-return policy",
            code="unsupported_invalid_return_policy",
        )
    if not np.isfinite(initial_prices).all() or np.any(initial_prices <= 0.0):
        raise NumericalError("initial_prices must be finite and strictly positive")
    if not np.isfinite(price_link_returns).all():
        raise NumericalError("price_link_returns contain nonfinite values")

    prices = np.empty((path_count, step_count + 1, asset_count), dtype=np.float64)
    prices[:, 0, :] = initial_prices
    for step_index in range(step_count):
        period_returns = price_link_returns[:, step_index, :]
        invalid = period_returns <= -1.0
        if np.any(invalid):
            path_id, asset_index = np.argwhere(invalid)[0]
            context = _location(
                path_id=int(path_id),
                step_index=step_index,
                asset_index=int(asset_index),
                asset_ids=asset_ids,
                future_dates=future_dates,
            )
            context["return_value"] = float(period_returns[path_id, asset_index])
            raise PolicyExecutionError(
                "arithmetic price-link return must be greater than -1",
                code="invalid_arithmetic_return",
                context=context,
            )

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            next_prices = prices[:, step_index, :] * (1.0 + period_returns)
        invalid_price = ~np.isfinite(next_prices) | (next_prices <= 0.0)
        if np.any(invalid_price):
            path_id, asset_index = np.argwhere(invalid_price)[0]
            context = _location(
                path_id=int(path_id),
                step_index=step_index,
                asset_index=int(asset_index),
                asset_ids=asset_ids,
                future_dates=future_dates,
            )
            context["prior_price"] = float(prices[path_id, step_index, asset_index])
            context["return_value"] = float(period_returns[path_id, asset_index])
            raise NumericalError(
                "arithmetic price evolution produced a nonfinite or nonpositive price",
                code="nonfinite_price",
                context=context,
            )
        prices[:, step_index + 1, :] = next_prices
    return prices
