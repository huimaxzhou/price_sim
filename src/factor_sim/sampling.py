"""Covariance roots, reproducible streams, and Gaussian innovation sampling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .config import (
    FactorReturnDistribution,
    JointDistributionPolicy,
    SerialCorrelationPolicy,
    SpecificReturnDistribution,
)
from .errors import ConfigurationError, NumericalError, ValidationIssue
from .models import InnovationBatch, ResolvedSimulationPolicy

FloatArray = NDArray[np.float64]
Sampler = Callable[..., InnovationBatch]


@dataclass(frozen=True, slots=True, kw_only=True)
class RandomStreams:
    """Path-stable Generator objects plus auditable derivation metadata."""

    joint_policy: JointDistributionPolicy
    factor_generators: tuple[Generator, ...]
    specific_generators: tuple[Generator, ...]
    joint_generators: tuple[Generator, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def path_count(self) -> int:
        if self.joint_policy is JointDistributionPolicy.JOINT:
            return len(self.joint_generators)
        return len(self.factor_generators)


def _configuration_error(
    code: str,
    field_path: str,
    message: str,
    context: Mapping[str, Any] | None = None,
) -> ConfigurationError:
    return ConfigurationError(
        (
            ValidationIssue(
                code=code,
                field_path=field_path,
                message=message,
                context=context,
            ),
        )
    )


def covariance_root(
    matrix: FloatArray,
    *,
    matrix_name: str,
    symmetry_tolerance: float,
    positive_semidefinite_tolerance: float,
) -> tuple[FloatArray, Mapping[str, Any]]:
    """Return a symmetric PSD root and normalization diagnostics.

    For ``A = Q diag(lambda) Q.T``, the returned root is
    ``Q diag(sqrt(max(lambda, 0))) Q.T``. Singular PSD matrices are valid.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a NumPy array")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NumericalError(f"{matrix_name} must be a square matrix")
    if not np.isfinite(matrix).all():
        raise NumericalError(f"{matrix_name} contains nonfinite values")
    if symmetry_tolerance < 0 or positive_semidefinite_tolerance < 0:
        raise ValueError("numerical tolerances must be nonnegative")

    maximum_asymmetry = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    if maximum_asymmetry > symmetry_tolerance:
        raise NumericalError(
            f"{matrix_name} is asymmetric: maximum asymmetry "
            f"{maximum_asymmetry:.17g} exceeds tolerance "
            f"{symmetry_tolerance:.17g}"
        )

    canonical = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(canonical)
    minimum_eigenvalue = float(eigenvalues.min(initial=np.inf))
    if minimum_eigenvalue < -positive_semidefinite_tolerance:
        raise NumericalError(
            f"{matrix_name} is indefinite: minimum eigenvalue "
            f"{minimum_eigenvalue:.17g} is below -tolerance "
            f"{-positive_semidefinite_tolerance:.17g}"
        )

    negative = eigenvalues < 0.0
    zeroed_count = int(np.count_nonzero(negative))
    zeroed_minimum = (
        float(eigenvalues[negative].min(initial=np.inf)) if zeroed_count else None
    )
    clipped = np.maximum(eigenvalues, 0.0)
    root = np.asarray(
        (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T,
        dtype=np.float64,
    )
    root = np.array(root, dtype=np.float64, copy=True, order="C")
    diagnostics: Mapping[str, Any] = MappingProxyType(
        {
            "matrix_name": matrix_name,
            "maximum_asymmetry": maximum_asymmetry,
            "canonicalized": maximum_asymmetry > 0.0,
            "minimum_eigenvalue": minimum_eigenvalue,
            "zeroed_eigenvalue_count": zeroed_count,
            "zeroed_eigenvalue_minimum": zeroed_minimum,
            "method": "symmetric_eigendecomposition",
        }
    )
    return root, diagnostics


def build_random_streams(
    *,
    seed: int,
    path_count: int,
    joint_policy: JointDistributionPolicy,
    bit_generator_name: str,
    stream_policy: str,
) -> RandomStreams:
    """Build one stable child stream per path under the resolved policy."""

    if bit_generator_name != "PCG64":
        raise _configuration_error(
            "unsupported_rng_bit_generator",
            "rng_bit_generator",
            "version one registers only PCG64",
            {"value": bit_generator_name},
        )
    if stream_policy != "per_path":
        raise _configuration_error(
            "unsupported_rng_stream_policy",
            "rng_stream_policy",
            "version one registers only per_path stream allocation",
            {"value": stream_policy},
        )
    if type(joint_policy) is not JointDistributionPolicy:
        raise _configuration_error(
            "invalid_policy_type",
            "policy.return_joint_distribution",
            "the stream builder requires a JointDistributionPolicy member",
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise _configuration_error(
            "invalid_random_seed",
            "random_seed",
            "the seed must be a nonnegative integer",
        )
    if (
        isinstance(path_count, bool)
        or not isinstance(path_count, int)
        or path_count <= 0
    ):
        raise _configuration_error(
            "nonpositive_dimension",
            "path_count",
            "path_count must be a positive integer",
        )

    master = np.random.SeedSequence(seed)
    path_sequences = master.spawn(path_count)
    path_metadata: list[Mapping[str, Any]] = []
    factor_generators: list[Generator] = []
    specific_generators: list[Generator] = []
    joint_generators: list[Generator] = []

    if joint_policy is JointDistributionPolicy.INDEPENDENT:
        for path_id, path_sequence in enumerate(path_sequences):
            factor_sequence, specific_sequence = path_sequence.spawn(2)
            factor_generators.append(Generator(np.random.PCG64(factor_sequence)))
            specific_generators.append(Generator(np.random.PCG64(specific_sequence)))
            path_metadata.append(
                MappingProxyType(
                    {
                        "path_id": path_id,
                        "path_spawn_key": path_sequence.spawn_key,
                        "factor_spawn_key": factor_sequence.spawn_key,
                        "specific_spawn_key": specific_sequence.spawn_key,
                    }
                )
            )
        derivation = "SeedSequence -> path -> factor,specific"
    else:
        for path_id, path_sequence in enumerate(path_sequences):
            joint_generators.append(Generator(np.random.PCG64(path_sequence)))
            path_metadata.append(
                MappingProxyType(
                    {
                        "path_id": path_id,
                        "path_spawn_key": path_sequence.spawn_key,
                        "joint_spawn_key": path_sequence.spawn_key,
                    }
                )
            )
        derivation = "SeedSequence -> path -> joint"

    metadata = {
        "seed": seed,
        "bit_generator": bit_generator_name,
        "configured_stream_policy": stream_policy,
        "stream_policy": derivation,
        "path_streams": tuple(path_metadata),
        "numpy_version": np.__version__,
    }
    return RandomStreams(
        joint_policy=joint_policy,
        factor_generators=tuple(factor_generators),
        specific_generators=tuple(specific_generators),
        joint_generators=tuple(joint_generators),
        metadata=metadata,
    )


def _check_sampler_inputs(
    *,
    factor_covariance: FloatArray,
    specific_variances: FloatArray,
    step_count: int,
    rng_streams: RandomStreams,
    expected_policy: JointDistributionPolicy,
) -> tuple[int, int, int]:
    if not isinstance(rng_streams, RandomStreams):
        raise TypeError("rng_streams must be RandomStreams")
    if rng_streams.joint_policy is not expected_policy:
        raise _configuration_error(
            "incompatible_rng_stream_policy",
            "rng_streams.joint_policy",
            "the RNG stream bundle does not match the selected sampler",
        )
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count <= 0
    ):
        raise _configuration_error(
            "nonpositive_dimension",
            "step_count",
            "step_count must be a positive integer",
        )
    if (
        factor_covariance.ndim != 2
        or factor_covariance.shape[0] != factor_covariance.shape[1]
    ):
        raise NumericalError("factor_covariance must be square")
    if specific_variances.ndim != 1:
        raise NumericalError("specific_variances must be one-dimensional")
    if not np.isfinite(specific_variances).all() or np.any(specific_variances < 0.0):
        raise NumericalError("specific_variances must be finite and nonnegative")
    return rng_streams.path_count, step_count, factor_covariance.shape[0]


def sample_independent_gaussian(
    *,
    factor_covariance: FloatArray,
    specific_variances: FloatArray,
    step_count: int,
    rng_streams: RandomStreams,
    symmetry_tolerance: float,
    positive_semidefinite_tolerance: float,
) -> InnovationBatch:
    """Draw independent Gaussian factor and specific blocks.

    Outputs have shapes ``(M, H, K)`` and ``(M, H, N)``. Python loops only over
    paths so each Generator retains its stable path assignment.
    """

    path_count, step_count, factor_count = _check_sampler_inputs(
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
        step_count=step_count,
        rng_streams=rng_streams,
        expected_policy=JointDistributionPolicy.INDEPENDENT,
    )
    root, root_diagnostics = covariance_root(
        factor_covariance,
        matrix_name="factor_covariance",
        symmetry_tolerance=symmetry_tolerance,
        positive_semidefinite_tolerance=positive_semidefinite_tolerance,
    )
    specific_scale = np.sqrt(specific_variances)
    asset_count = specific_variances.shape[0]
    factor_innovations = np.empty(
        (path_count, step_count, factor_count), dtype=np.float64
    )
    specific_innovations = np.empty(
        (path_count, step_count, asset_count), dtype=np.float64
    )
    for path_id, (factor_rng, specific_rng) in enumerate(
        zip(
            rng_streams.factor_generators,
            rng_streams.specific_generators,
            strict=True,
        )
    ):
        factor_innovations[path_id] = (
            factor_rng.standard_normal((step_count, factor_count)) @ root.T
        )
        specific_innovations[path_id] = (
            specific_rng.standard_normal((step_count, asset_count)) * specific_scale
        )

    metadata = dict(rng_streams.metadata)
    metadata.update(
        {
            "sampler_name": "independent_gaussian",
            "sampler_version": "1",
            "factor_distribution": "Gaussian",
            "specific_distribution": "Gaussian",
            "joint_policy": "independent",
            "factor_covariance_root": root_diagnostics,
            "zero_specific_variance_count": int(
                np.count_nonzero(specific_variances == 0.0)
            ),
        }
    )
    return InnovationBatch(
        factor_innovations=factor_innovations,
        specific_innovations=specific_innovations,
        sampler_metadata=metadata,
    )


def sample_joint_gaussian(
    *,
    factor_covariance: FloatArray,
    specific_variances: FloatArray,
    step_count: int,
    rng_streams: RandomStreams,
    symmetry_tolerance: float,
    positive_semidefinite_tolerance: float,
) -> InnovationBatch:
    """Draw one block-diagonal Gaussian vector per path and future step."""

    path_count, step_count, factor_count = _check_sampler_inputs(
        factor_covariance=factor_covariance,
        specific_variances=specific_variances,
        step_count=step_count,
        rng_streams=rng_streams,
        expected_policy=JointDistributionPolicy.JOINT,
    )
    asset_count = specific_variances.shape[0]
    joint_count = factor_count + asset_count
    joint_covariance = np.zeros((joint_count, joint_count), dtype=np.float64)
    joint_covariance[:factor_count, :factor_count] = factor_covariance
    specific_indices = np.arange(asset_count) + factor_count
    joint_covariance[specific_indices, specific_indices] = specific_variances
    root, root_diagnostics = covariance_root(
        joint_covariance,
        matrix_name="joint_covariance",
        symmetry_tolerance=symmetry_tolerance,
        positive_semidefinite_tolerance=positive_semidefinite_tolerance,
    )

    factor_innovations = np.empty(
        (path_count, step_count, factor_count), dtype=np.float64
    )
    specific_innovations = np.empty(
        (path_count, step_count, asset_count), dtype=np.float64
    )
    for path_id, joint_rng in enumerate(rng_streams.joint_generators):
        joint_draws = joint_rng.standard_normal((step_count, joint_count)) @ root.T
        factor_innovations[path_id] = joint_draws[:, :factor_count]
        specific_innovations[path_id] = joint_draws[:, factor_count:]

    metadata = dict(rng_streams.metadata)
    metadata.update(
        {
            "sampler_name": "block_diagonal_gaussian",
            "sampler_version": "1",
            "factor_distribution": "Gaussian",
            "specific_distribution": "Gaussian",
            "joint_policy": "joint",
            "block_dimensions": {
                "factor": factor_count,
                "specific": asset_count,
                "joint": joint_count,
            },
            "joint_covariance_root": root_diagnostics,
        }
    )
    return InnovationBatch(
        factor_innovations=factor_innovations,
        specific_innovations=specific_innovations,
        sampler_metadata=metadata,
    )


_SAMPLER_REGISTRY: Mapping[str, Sampler] = MappingProxyType(
    {
        "independent_gaussian": sample_independent_gaussian,
        "block_diagonal_gaussian": sample_joint_gaussian,
    }
)


def sample_innovations(
    *,
    resolved_policy: ResolvedSimulationPolicy,
    factor_covariance: FloatArray,
    specific_variances: FloatArray,
    step_count: int,
    rng_streams: RandomStreams,
    symmetry_tolerance: float,
    positive_semidefinite_tolerance: float,
) -> InnovationBatch:
    """Dispatch to the sampler selected by the typed resolved policy."""

    if (
        resolved_policy.factor_return_distribution
        is not FactorReturnDistribution.GAUSSIAN
        or resolved_policy.specific_return_distribution
        is not SpecificReturnDistribution.GAUSSIAN
        or resolved_policy.return_serial_correlation is not SerialCorrelationPolicy.NONE
    ):
        raise _configuration_error(
            "unsupported_policy_value",
            "policy",
            "the sampling registry supports Gaussian, serially independent "
            "innovations only",
        )

    common = {
        "factor_covariance": factor_covariance,
        "specific_variances": specific_variances,
        "step_count": step_count,
        "rng_streams": rng_streams,
        "symmetry_tolerance": symmetry_tolerance,
        "positive_semidefinite_tolerance": positive_semidefinite_tolerance,
    }
    if resolved_policy.return_joint_distribution is JointDistributionPolicy.INDEPENDENT:
        sampler_name = "independent_gaussian"
    elif resolved_policy.return_joint_distribution is JointDistributionPolicy.JOINT:
        generator = resolved_policy.joint_generator
        if generator is None:
            raise _configuration_error(
                "missing_joint_generator",
                "policy.joint_generator",
                "joint sampling requires its grouped generator configuration",
            )
        if generator.name != "block_diagonal_gaussian" or generator.version != "1":
            raise _configuration_error(
                "unsupported_joint_generator",
                "policy.joint_generator",
                "the resolved joint generator is not registered",
                {"name": generator.name, "version": generator.version},
            )
        sampler_name = generator.name
    else:
        raise _configuration_error(
            "unsupported_policy_value",
            "policy.return_joint_distribution",
            "the resolved joint policy is not supported",
        )
    return _SAMPLER_REGISTRY[sampler_name](**common)
