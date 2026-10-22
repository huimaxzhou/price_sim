# Baseline Factor-Risk Equity Return and Price Simulation Specification

## 1. Purpose and scope

This document specifies a baseline engine that simulates daily equity returns and converts them into price paths. The risk input may come from a Barra model or from a custom factor risk model, provided that it satisfies the interface and covariance assumptions stated here.

The baseline is deliberately conditional and static:

- the asset universe and factor set are fixed for a run;
- the latest point-in-time exposure matrix, factor covariance matrix, and specific covariance matrix are held fixed over the simulation horizon;
- the engine simulates one-day returns for future trading dates; and
- expected returns, innovation distributions, temporal dependence, and the mapping from modeled returns to prices are simulation policies rather than outputs of the risk model.

A factor covariance risk model does **not** by itself specify expected returns, a probability distribution, serial dependence, corporate actions, or a complete stochastic price process. The simulator must not infer any of those items from the covariance inputs. Section 4.3 defines simulator-policy defaults; those defaults are implementation choices and must not be represented as properties of the risk model.

Rolling re-estimation of exposures or risk, endogenous corporate actions, changing asset universes, and regime-switching dynamics are outside this baseline specification.

The words **must**, **must not**, **should**, and **may** are normative. A requirement described as an **external policy** is part of the simulator configuration, not an implication of the factor risk model.

## 2. Baseline model and compatibility requirements

Let the simulation origin be the end of trading day $t_0$. Let $h=1,\ldots,H$ index future simulated trading days, so the simulated date at step $h$ is $t_h$. For a fixed universe of $N$ assets and $K$ factors, the return model is

$$
\mathbf r_h
=
\mathbf X_0\mathbf f_h
+
\boldsymbol\epsilon_h.
$$

Here, $\mathbf r_h\in\mathbb R^N$ is the vector of modeled one-day asset returns, $\mathbf X_0\in\mathbb R^{N\times K}$ is the exposure matrix known at the simulation origin, $\mathbf f_h\in\mathbb R^K$ is the factor-return vector, and $\boldsymbol\epsilon_h\in\mathbb R^N$ is the specific-return vector. In plain English, each asset return is the sum of the returns attributable to its factor exposures and an asset-specific residual return.

The baseline uses

$$
\begin{aligned}
\mathbf f_h &= \boldsymbol\mu^f_h + \boldsymbol\eta^f_h,\\
\boldsymbol\epsilon_h &= \boldsymbol\mu^\epsilon_h + \boldsymbol\eta^\epsilon_h,
\end{aligned}
$$

where $\boldsymbol\mu^f_h\in\mathbb R^K$ and $\boldsymbol\mu^\epsilon_h\in\mathbb R^N$ are externally supplied one-day mean vectors, while $\boldsymbol\eta^f_h\in\mathbb R^K$ and $\boldsymbol\eta^\epsilon_h\in\mathbb R^N$ are zero-mean innovations. In plain English, the risk model supplies covariance structure, while the caller separately supplies or explicitly approves the drift assumptions.

For every future step $h$, the innovations must satisfy

$$
\begin{aligned}
\mathbb E[\boldsymbol\eta^f_h] &= \mathbf 0,\\
\mathbb E[\boldsymbol\eta^\epsilon_h] &= \mathbf 0,\\
\operatorname{Cov}(\boldsymbol\eta^f_h) &= \mathbf F_0,\\
\operatorname{Cov}(\boldsymbol\eta^\epsilon_h) &= \mathbf D_0,\\
\operatorname{Cov}(\boldsymbol\eta^f_h,\boldsymbol\eta^\epsilon_h) &= \mathbf 0_{K\times N}.
\end{aligned}
$$

Here, $\mathbf F_0\in\mathbb R^{K\times K}$ is the one-day factor covariance matrix and $\mathbf D_0\in\mathbb R^{N\times N}$ is the one-day specific covariance matrix, both known at the simulation origin. In plain English, factor and specific innovations may follow any authorized distribution, but their one-day means and covariance targets must match the supplied inputs, and their contemporaneous cross-covariance must be zero.

Conditional on the frozen exposure matrix, the resulting one-day asset moments are

$$
\mathbb E[\mathbf r_h]
=
\mathbf X_0\boldsymbol\mu^f_h
+
\boldsymbol\mu^\epsilon_h
$$

and

$$
\boldsymbol\Sigma_0
:=
\operatorname{Cov}(\mathbf r_h)
=
\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}
+
\mathbf D_0.
$$

In plain English, expected asset returns are determined by the external mean inputs, while shared factor exposures and specific risk determine the target contemporaneous covariance.

This baseline makes the following compatibility choices explicit:

1. $\mathbf D_0$ is diagonal. A custom model with correlated residual returns is not compatible with this baseline unless an authorized upstream transformation produces a diagonal specific covariance and records that transformation.
2. Factor and specific innovations are contemporaneously uncorrelated. This condition is weaker than statistical independence.
3. $\mathbf X_0$, $\mathbf F_0$, and $\mathbf D_0$ are frozen for all $H$ simulated days.
4. The matrices supplied to the engine are final one-day risk forecasts. Historical estimation, volatility adjustment, covariance shrinkage, and positive-semidefinite repair are upstream responsibilities unless separately specified.
5. No innovation distribution and no serial-dependence rule are implied by the covariance equations. The Gaussian and no-serial-correlation defaults in Section 4.3 are simulator policies.

## 3. Notation, dimensions, and status

| Symbol | Dimension | Meaning | Time status | Source/status |
|---|---:|---|---|---|
| $N$ | scalar | Number of assets | Fixed for a run | Input metadata |
| $K$ | scalar | Number of factors | Fixed for a run | Input metadata |
| $H$ | scalar | Number of future trading days | Fixed for a run | Simulation configuration |
| $M$ | scalar | Number of Monte Carlo paths | Fixed for a run | Simulation configuration |
| $\mathbf X_0$ | $N\times K$ | Asset-by-factor exposure matrix | Known at $t_0$; frozen | Risk-model input |
| $\mathbf F_0$ | $K\times K$ | One-day factor covariance matrix | Known at $t_0$; frozen | Risk-model input |
| $\mathbf D_0$ | $N\times N$ | One-day diagonal specific covariance matrix | Known at $t_0$; frozen | Risk-model input |
| $\boldsymbol\Sigma_0$ | $N\times N$ | One-day modeled asset covariance | Derived; frozen | Computed |
| $\boldsymbol\mu^f_h$ | $K$ | Mean factor return for step $h$ | Fixed or scheduled | External policy |
| $\boldsymbol\mu^\epsilon_h$ | $N$ | Mean specific return for step $h$ | Fixed or scheduled | External policy |
| $\boldsymbol\eta^f_h$ | $K$ | Zero-mean factor innovation | Simulated | Simulation output |
| $\boldsymbol\eta^\epsilon_h$ | $N$ | Zero-mean specific innovation | Simulated | Simulation output |
| $\mathbf f_h$ | $K$ | Factor return | Simulated | Simulation output |
| $\boldsymbol\epsilon_h$ | $N$ | Specific return | Simulated | Simulation output |
| $\mathbf r_h$ | $N$ | Modeled asset return | Simulated | Simulation output |
| $\mathbf q_h$ | $N$ | Arithmetic return used in the price link, if applicable | Derived from $\mathbf r_h$ | Price-policy output |
| $\mathbf g_h$ | $N$ | Log return used in the price link, if applicable | Derived from $\mathbf r_h$ | Price-policy output |
| $\mathbf P_h$ | $N$ | Simulated prices at step $h$ | Simulated | Simulation output |
| $\mathbf P_0$ | $N$ | Strictly positive initial prices at $t_0$ | Observed or supplied | Price input |

For asset $i$, let $\mathbf x_i^{\mathsf T}\in\mathbb R^{1\times K}$ denote row $i$ of $\mathbf X_0$. The target covariance between assets $i$ and $j$ is

$$
\Sigma_{ij,0}
=
\mathbf x_i^{\mathsf T}\mathbf F_0\mathbf x_j
+
\mathbf 1_{\{i=j\}}D_{ii,0},
$$

where $\mathbf 1_{\{i=j\}}$ equals one when $i=j$ and zero otherwise. In plain English, two different assets covary through their common factors; an asset's own variance also includes its specific variance.

## 4. Input contract

### 4.1 Run identity and calendar

The run configuration must provide:

- a unique `run_id`;
- a risk-model name and version, without requiring that the model be a Barra product;
- simulation origin $t_0$;
- an ordered list of $H$ future trading dates;
- ordered, unique `asset_ids` of length $N$;
- ordered, unique `factor_ids` of length $K$;
- number of paths $M\ge 1$; and
- an explicit trading calendar and time zone.

The future-date list, rather than calendar-day arithmetic, determines the number and order of simulated return periods.

### 4.2 Frozen risk state

The simulator must receive:

- $\mathbf X_0\in\mathbb R^{N\times K}$, aligned to `asset_ids` and `factor_ids`;
- $\mathbf F_0\in\mathbb R^{K\times K}$, aligned to `factor_ids` on both axes; and
- either $\mathbf D_0\in\mathbb R^{N\times N}$ or a vector of specific variances $\mathbf d_0\in\mathbb R^N$, aligned to `asset_ids`, with $\mathbf D_0=\operatorname{diag}(\mathbf d_0)$.

The simulator may also receive factor metadata aligned to `factor_ids`. Supported optional fields include:

- `factor_class`, such as `country`, `industry`, `style`, `market`, `currency`, or `model_specific`;
- `exposure_class`, when the producing model uses that name instead of `factor_class`;
- `factor_name` and `factor_description`; and
- model-specific factor attributes stored without changing the numerical exposure matrix.

If both `factor_class` and `exposure_class` are supplied for the same factor, they must agree under an explicitly documented mapping. Missing optional factor metadata is not a validation error. Supplied metadata must remain aligned with the ordered `factor_ids`, be persisted in the output, and must not change the interpretation of $\mathbf X_0$ silently.

Each state object must carry:

- `as_of_date`;
- `known_at` timestamp;
- return horizon;
- observation frequency;
- modeled-return convention, such as arithmetic or log;
- return basis, such as price, total, or excess return;
- covariance or exposure units, as applicable;
- annualization state; and
- provenance sufficient to identify the producing model run or source file.

The engine must require `known_at <= simulation_origin`. Factor covariance, specific covariance, mean vectors, and modeled returns must use the same return convention and return basis. All covariance inputs must already be expressed in one-trading-day return units before simulation begins. If an upstream conversion or return-basis transformation was applied, its method and assumptions must be recorded. 

### 4.3 Simulation-policy inputs and defaults

The simulator accepts the following policy fields. If a field is omitted, the engine must resolve it to the stated default, record `value_source = default`, and persist the resolved value in the run manifest. If the caller supplies a value, the engine must record `value_source = explicit`.

| Policy field | Default | Baseline meaning |
|---|---|---|
| `factor_exposure_dynamics` | `frozen` | Use $\mathbf X_h=\mathbf X_0$ for every simulated day. |
| `factor_volatility_dynamics` | `frozen` | Keep every factor's marginal volatility implied by $\mathbf F_0$ unchanged. |
| `factor_correlation_dynamics` | `frozen` | Keep the factor-correlation structure implied by $\mathbf F_0$ unchanged. Together with frozen factor volatility, this gives $\mathbf F_h=\mathbf F_0$. |
| `specific_risk_dynamics` | `frozen` | Use $\mathbf D_h=\mathbf D_0$ for every simulated day. |
| `factor_return_distribution` | `Gaussian` | Draw factor innovations from a multivariate Gaussian distribution with zero mean and covariance $\mathbf F_0$. |
| `specific_return_distribution` | `Gaussian` | Draw specific innovations from a multivariate Gaussian distribution with zero mean and covariance $\mathbf D_0$. |
| `return_serial_correlation` | `none` | Conditional on the configured means and frozen risk state, innovation draws are independent across simulated days. |
| `return_joint_distribution` | `independent` | Select `independent` to sample the factor and specific innovation blocks independently, or `joint` to use one registered joint generator subject to the covariance requirements below. |
| `price_return_convention` | `raw` | Treat a compatible modeled return as a raw simple arithmetic price return and use it directly in the arithmetic price link. |
| `currency_handling` | `None` | Perform no currency conversion inside the simulator; each price path remains in the asset's declared quote currency. |
| `corporate_action_adjustment` | `None` | Apply no dividend, split, merger, delisting, or other corporate-action adjustment inside the simulator. |
| `new_listing` | `None` | Add no assets after the simulation origin; the ordered asset universe remains fixed. |

A fully defaulted policy object is equivalent to:

```yaml
policies:
  factor_exposure_dynamics: frozen
  factor_volatility_dynamics: frozen
  factor_correlation_dynamics: frozen
  specific_risk_dynamics: frozen
  factor_return_distribution: Gaussian
  specific_return_distribution: Gaussian
  return_serial_correlation: none
  return_joint_distribution: independent
  price_return_convention: raw
  currency_handling: null
  corporate_action_adjustment: null
  new_listing: null
```

In serialized configuration, YAML `null`, JSON `null`, and the specification label `None` represent the same absence of an optional handler. Implementations must choose one canonical internal representation and record it.

For this baseline, the four dynamics policies support `frozen`. A non-frozen value is valid only when a separately specified and registered dynamics implementation defines its parameters, state transition, timing, and validation rules. Similarly, a non-`None` currency, corporate-action, or new-listing policy requires a separately specified handler. The baseline engine must reject an unsupported nondefault value rather than silently treating it as the default.

When both factor-volatility and factor-correlation dynamics are `frozen`, the engine copies $\mathbf F_0$ directly and need not decompose it into volatilities and correlations. This avoids an undefined correlation calculation for any factor whose variance is zero.

The `independent` joint-distribution default applies between the factor and specific innovation **blocks**. It does not imply that factor components or asset returns are mutually independent: factor dependence is governed by $\mathbf F_0$, and shared exposures transmit that dependence to assets. Under diagonal $\mathbf D_0$ and the Gaussian default, specific innovation components are mutually independent as a mathematical consequence of that selected distribution and covariance, not merely because the covariance is diagonal.

If `return_joint_distribution = joint`, the caller must provide a named and parameterized joint generator for

$$
\mathbf z_h
=
\begin{bmatrix}
\boldsymbol\eta^f_h\\
\boldsymbol\eta^\epsilon_h
\end{bmatrix}
\in\mathbb R^{K+N}
$$

with target moments

$$
\mathbb E[\mathbf z_h]=\mathbf 0,
\qquad
\operatorname{Cov}(\mathbf z_h)
=
\begin{bmatrix}
\mathbf F_0 & \mathbf 0_{K\times N}\\
\mathbf 0_{N\times K} & \mathbf D_0
\end{bmatrix}.
$$

In plain English, `joint` allows the two innovation blocks to be produced by one joint sampling procedure, but the baseline still requires zero factor-specific cross-covariance so that $\boldsymbol\Sigma_0=\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}+\mathbf D_0$ remains valid. A non-Gaussian joint generator may create dependence that is not captured by covariance. Under one multivariate Gaussian joint generator with the block-diagonal covariance above, the two blocks are independent, so `joint` and `independent` differ operationally but not distributionally.

If a proposed joint generator has nonzero cross-covariance $\mathbf C_0:=\operatorname{Cov}(\boldsymbol\eta^f_h,\boldsymbol\eta^\epsilon_h)\in\mathbb R^{K\times N}$, the resulting asset covariance would instead be

$$
\operatorname{Cov}(\mathbf r_h)
=
\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}
+
\mathbf D_0
+
\mathbf X_0\mathbf C_0
+
\mathbf C_0^{\mathsf T}\mathbf X_0^{\mathsf T}.
$$

In plain English, nonzero factor-specific cross-covariance changes the defining covariance equation. It is outside this baseline and must be rejected unless a separately authorized generalized model replaces the baseline invariant throughout the implementation and validation contract.

The `raw` price-return default means the simple arithmetic return

$$
q_{i,h}
=
\frac{P_{i,h}-P_{i,h-1}}{P_{i,h-1}}.
$$

In plain English, `raw` means an unadjusted one-period price change divided by the prior price; it does not mean a log return or total return. This default may be used as an identity mapping only when the risk-model metadata declares the modeled asset return to be a raw arithmetic price return. Otherwise, the caller must provide a compatible return mapping.

### 4.4 Stochastic policy details

After applying the defaults in Section 4.3, the stochastic configuration must contain:

- $\boldsymbol\mu^f_h\in\mathbb R^K$ for every step $h$, or one vector declared constant across steps;
- $\boldsymbol\mu^\epsilon_h\in\mathbb R^N$ for every step $h$, or one vector declared constant across steps;
- the resolved factor and specific return distributions;
- the resolved contemporaneous factor-specific joint-distribution policy;
- the resolved return serial-correlation policy; and
- random-number-generator identity, seed, and stream-allocation policy.

An explicitly supplied zero mean is valid. An omitted mean is a configuration error. Unlike the means, the distribution and serial-correlation fields use the defaults in Section 4.3 when omitted.

When the resolved joint-distribution policy is `joint`, the configuration must additionally supply the joint generator's identity, version, parameters, marginal-distribution contract, and random-stream requirements.

Under all stochastic defaults, the innovations satisfy

$$
\boldsymbol\eta^f_h
\sim
\mathcal N(\mathbf 0,\mathbf F_0),
\qquad
\boldsymbol\eta^\epsilon_h
\sim
\mathcal N(\mathbf 0,\mathbf D_0),
$$

with the two vectors independent at each step and the innovation pairs independent across different steps. In plain English, the default process uses Gaussian one-day shocks, preserves the covariance within each block, creates no factor-specific dependence, and creates no serial dependence.

An override may use another named and fully parameterized innovation generator and may draw factor and specific innovations jointly or separately. Regardless of implementation, emitted innovations must target the moments in Section 2 and satisfy the resolved joint and serial policies. Covariance matching alone does not establish that the draws have the intended tails, skewness, or temporal behavior.

### 4.5 Return and price policy

The caller must provide $\mathbf P_0\in\mathbb R^N_{>0}$ and risk-model return-basis metadata. The following must be declared explicitly or resolved through Section 4.3:

- whether $\mathbf r_h$ represents arithmetic or log returns;
- whether it represents raw price, adjusted price, total, or excess returns;
- the transformation from $\mathbf r_h$ to the return used by the price link, with the `raw` identity default allowed only for a compatible raw arithmetic price return;
- whether the price series is raw, split-adjusted, or total-return adjusted;
- the resolved currency-handling policy;
- the resolved corporate-action-adjustment policy;
- the resolved new-listing policy; and
- the policy for invalid arithmetic returns at or below $-100\%$.

Under `currency_handling = None`, the simulator performs no FX conversion or currency aggregation. Each asset must carry an `asset_currency`, and each emitted price remains in that currency. A supported non-`None` currency handler must specify at least the target or base currency, FX quote convention, FX return or price inputs, timestamp alignment, conversion timing, and behavior for missing FX observations. Currency conversion is distinct from exposure to currency factors in $\mathbf X_0$; the simulator must not treat one as a substitute for the other.

Under the remaining defaults, the simulator performs no corporate-action adjustment and introduces no new listings. Therefore, the supplied prices and modeled returns must already have the intended adjustment basis. If the requested price-path interpretation requires corporate actions or universe changes, the caller must provide a supported nondefault handler or process them outside this baseline engine.

If a nondefault delisting or corporate-action handler can change whether an asset remains live, it must emit an `asset_state` event containing:

- `asset_id`;
- `path_id`;
- `effective_date`;
- `status`; and
- `reason`.

The handler must define the allowed `status` values and their price semantics. At minimum, it must distinguish a live asset from a terminal or delisted asset. The simulator must not encode terminal status solely through a zero, missing, or nonfinite price.

### 4.6 Numerical and audit policy

The configuration must declare:

- symmetry tolerance $\tau_{\mathrm{sym}}\ge 0$;
- positive-semidefinite tolerance $\tau_{\mathrm{PSD}}\ge 0$;
- equation tolerance $\tau_{\mathrm{eq}}\ge 0$;
- statistical tolerances or confidence levels for Monte Carlo validation;
- floating-point precision and numerical platform, where reproducibility requires them; and
- whether any numerical repair is permitted.

No symmetry, diagonal, or positive-semidefinite repair may occur silently. If repair is authorized upstream, both the original and repaired objects and the repair method must be retained.

## 5. Simulation procedure

### 5.1 Initialize and validate the frozen state

At $t_0$, load $\mathbf X_0$, $\mathbf F_0$, $\mathbf D_0$, $\mathbf P_0$, ordered identifiers, asset currencies, optional factor metadata, dates, units, and supplied policies. Resolve every omitted policy field using Section 4.3 and persist whether each value was explicit or defaulted. Then compute

$$
\boldsymbol\Sigma_0
=
\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}
+
\mathbf D_0.
$$

In plain English, the engine constructs the asset covariance target once from the frozen risk state.

The run must fail before sampling if required metadata, mean vectors, random-number controls, or nondefault-policy parameters are absent; a policy value is unsupported; identifiers are misaligned; units are incompatible; prices are nonpositive; or matrix checks fail.

### 5.2 Draw factor and specific innovations

For every path $m=1,\ldots,M$ and future step $h=1,\ldots,H$, draw

$$
\boldsymbol\eta^{f,(m)}_h\in\mathbb R^K,
\qquad
\boldsymbol\eta^{\epsilon,(m)}_h\in\mathbb R^N
$$

from the resolved innovation policy. The policy must target the contemporaneous moments in Section 2 and the resolved temporal-dependence behavior. Under the default `none` serial-correlation policy, draws for different simulated days are independent. In plain English, freezing the risk matrices does not by itself require serial independence; independence across days comes from the separately recorded default or override.

Construct

$$
\begin{aligned}
\mathbf f^{(m)}_h
&=
\boldsymbol\mu^f_h+\boldsymbol\eta^{f,(m)}_h,\\
\boldsymbol\epsilon^{(m)}_h
&=
\boldsymbol\mu^\epsilon_h+\boldsymbol\eta^{\epsilon,(m)}_h.
\end{aligned}
$$

In plain English, each innovation is shifted by its externally specified one-day mean.

### 5.3 Construct modeled asset returns

For every path and future step, compute

$$
\mathbf r^{(m)}_h
=
\mathbf X_0\mathbf f^{(m)}_h
+
\boldsymbol\epsilon^{(m)}_h.
$$

In plain English, the same exposure matrix is applied on every simulated day, and the day's specific return is then added.

The frozen-state rule is

$$
\mathbf X_h=\mathbf X_0,
\qquad
\mathbf F_h=\mathbf F_0,
\qquad
\mathbf D_h=\mathbf D_0,
\qquad h=1,\ldots,H.
$$

In plain English, neither simulated returns nor elapsed time update the risk state in this baseline.

### 5.4 Map modeled returns to price-link returns

Apply the resolved return mapping to $\mathbf r^{(m)}_h$. The mapping must identify its output as either an arithmetic return $\mathbf q^{(m)}_h$ or a log return $\mathbf g^{(m)}_h$. Under the `raw` default, the mapping is the identity and the output is arithmetic only when the modeled-return metadata identifies $\mathbf r^{(m)}_h$ as a raw arithmetic price return. Otherwise, an explicit compatible mapping is required.

The engine must retain both the original modeled return and the mapped price-link return. In plain English, the simulator must not silently treat a total return, excess return, or differently adjusted return as an ordinary price return.

### 5.5 Evolve prices

For an arithmetic price-link return, update each asset $i=1,\ldots,N$ by

$$
P^{(m)}_{i,h}
=
P^{(m)}_{i,h-1}\left(1+q^{(m)}_{i,h}\right).
$$

In plain English, the prior price is multiplied by one plus the arithmetic return. A live price remains strictly positive only if $q^{(m)}_{i,h}>-1$.

If $q^{(m)}_{i,h}\le -1$, the engine must apply the explicitly configured invalid-return policy or fail. Rejection, resampling, truncation, clipping, flooring, or a terminal-state rule changes the emitted distribution and may change its covariance; the engine must record the action and revalidate the final emitted returns.

For a log price-link return, update by

$$
P^{(m)}_{i,h}
=
P^{(m)}_{i,h-1}\exp\!\left(g^{(m)}_{i,h}\right).
$$

In plain English, exponentiating a finite log return preserves a positive price, although the implementation must still detect numerical overflow, underflow, NaN, and infinity.

### 5.6 Repeat without updating risk state

Set the current price vector to $\mathbf P^{(m)}_h$, advance to step $h+1$, and repeat Sections 5.2–5.5. Do not update $\mathbf X_0$, $\mathbf F_0$, or $\mathbf D_0$.

## 6. Output contract

For each run, the engine must emit:

- factor innovations with shape $M\times H\times K$;
- specific innovations with shape $M\times H\times N$;
- factor returns with shape $M\times H\times K$;
- specific returns with shape $M\times H\times N$;
- modeled asset returns with shape $M\times H\times N$;
- mapped price-link returns with shape $M\times H\times N$, carrying return-type and adjustment tags;
- prices with shape $M\times(H+1)\times N$, including $\mathbf P_0$;
- asset currencies and the resolved currency-handling policy;
- supplied factor metadata aligned to the ordered factor IDs;
- when a nondefault delisting or corporate-action handler can change asset liveness, `asset_state` event records containing `asset_id`, `path_id`, `effective_date`, `status`, and `reason`;
- the exact frozen $\mathbf X_0$, $\mathbf F_0$, $\mathbf D_0$, and derived $\boldsymbol\Sigma_0$;
- ordered asset IDs, factor IDs, and simulated trading dates;
- the complete resolved-policy and assumption manifest, including each policy value and whether it was explicit or defaulted;
- random-number-generator, seed, version, and stream metadata;
- validation results and any numerical or price-safeguard events; and
- provenance for every risk-model input.

The engine should use distinct field names for modeled returns and mapped price-link returns even when the mapping is the identity.

## 7. Invariants and validation

### 7.1 Pre-simulation deterministic checks

- [ ] $\mathbf X_0$ has shape $N\times K$, $\mathbf F_0$ has shape $K\times K$, and $\mathbf D_0$ has shape $N\times N$.
- [ ] Asset and factor identifiers are unique, ordered, and aligned across all arrays.
- [ ] Supplied factor metadata is aligned to `factor_ids`; if both `factor_class` and `exposure_class` are present, their documented mapping is consistent.
- [ ] Every asset has an `asset_currency`; every nondefault currency policy provides its required FX inputs and parameters.
- [ ] All inputs are finite; initial prices are strictly positive; specific variances are nonnegative.
- [ ] $\mathbf F_0$ is symmetric within $\tau_{\mathrm{sym}}$ and positive semidefinite within $\tau_{\mathrm{PSD}}$.
- [ ] $D_{ij,0}=0$ for every $i\ne j$, and every diagonal entry of $\mathbf D_0$ is nonnegative.
- [ ] $\boldsymbol\Sigma_0=\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}+\mathbf D_0$ is symmetric and positive semidefinite within tolerance.
- [ ] All risk inputs were known by $t_0$, use one-day covariance units and the same return basis, and refer to the same ordered universe and factor set.
- [ ] Mean vectors, invalid-return policy, calendar, and random policy are present; all other simulation policies have been resolved from explicit values or documented defaults.
- [ ] Every nondefault policy value names a supported implementation and supplies its required parameters.
- [ ] A `joint` factor-specific generation policy supplies a registered joint generator whose marginal moments and zero cross-covariance target match Section 4.3.
- [ ] The `raw` price-return default is used only when the modeled-return metadata is a raw arithmetic price return.

### 7.2 Per-draw and per-path invariants

- [ ] Every modeled return satisfies $\mathbf r^{(m)}_h=\mathbf X_0\mathbf f^{(m)}_h+\boldsymbol\epsilon^{(m)}_h$ within $\tau_{\mathrm{eq}}$.
- [ ] The stored $\mathbf X_0$, $\mathbf F_0$, and $\mathbf D_0$ do not change across steps or paths.
- [ ] The resolved frozen dynamics policies remain unchanged and are applied on every step.
- [ ] Every mapped return is tagged as arithmetic or log and as price or total return.
- [ ] Every live price is finite and strictly positive.
- [ ] Every safeguard event is recorded and reflected in the final emitted return arrays.
- [ ] Every liveness-changing corporate-action or delisting event emits a complete `asset_state` record, and terminal status is not represented solely by an invalid or missing price.

### 7.3 Monte Carlo statistical checks

Using sufficiently many independent path replications at a fixed future step, verify that

$$
\widehat{\operatorname{Cov}}(\boldsymbol\eta^f_h)
\approx
\mathbf F_0,
\qquad
\widehat{\operatorname{Cov}}(\boldsymbol\eta^\epsilon_h)
\approx
\mathbf D_0,
$$

$$
\widehat{\operatorname{Cov}}(\boldsymbol\eta^f_h,\boldsymbol\eta^\epsilon_h)
\approx
\mathbf 0_{K\times N},
$$

and

$$
\widehat{\operatorname{Cov}}(\mathbf r_h)
\approx
\mathbf X_0\mathbf F_0\mathbf X_0^{\mathsf T}+\mathbf D_0.
$$

In plain English, empirical one-day moments should approach the configured targets as the number of paths grows. Tolerances or confidence intervals must account for Monte Carlo sample size and the authorized innovation distribution.

Also verify:

- [ ] empirical means of factor, specific, and asset returns against their configured targets;
- [ ] factor, specific, and asset marginal volatilities;
- [ ] selected or all cross-asset correlations;
- [ ] off-diagonal specific-return covariances against zero;
- [ ] zero serial dependence under the default `none` policy, or the configured temporal-dependence statistics under an override;
- [ ] factor-specific block independence under the default `independent` joint policy;
- [ ] the configured joint generator's marginal distributions, cross-covariance, and any additional dependence diagnostics under the `joint` policy; and
- [ ] post-safeguard moments whenever emitted returns differ from original draws.

Zero empirical cross-covariance alone is not proof of independence. Under the default `independent` policy, validate both the generator construction and suitable distribution-sensitive diagnostics; under an override that requires only zero covariance, do not relabel that weaker property as independence.

### 7.4 Engineering checks

- [ ] A fixed seed, generator version, stream policy, inputs, and platform reproduce identical outputs, or the documented deterministic tolerance is met.
- [ ] Omitting each policy field separately resolves it to the documented default and records `value_source = default`.
- [ ] Supplying each supported policy explicitly records `value_source = explicit` and does not change unrelated defaults.
- [ ] Separate paths do not accidentally reuse identical random streams.
- [ ] Singular positive-semidefinite covariance matrices are either supported or rejected with an explicit diagnostic.
- [ ] NaN, infinity, duplicate IDs, missing dates, incompatible units, and nonpositive prices fail validation.
- [ ] Arithmetic returns at or below $-100\%$, log-return overflow, and extreme exposures exercise the configured failure or safeguard behavior.
- [ ] Serialized outputs contain sufficient metadata to reproduce the run.
- [ ] `currency_handling = None` leaves every price in its declared asset currency; a nondefault handler reproduces its documented FX conversion convention.

## 8. Prohibited assumptions

Except where a documented Section 4.3 default supplies the authorization, the implementation must not:

- use an innovation distribution other than the resolved factor and specific distribution policies;
- assume zero factor or specific expected returns;
- infer independence solely from zero covariance; the default block independence comes from the resolved `return_joint_distribution` policy;
- use a serial-dependence process other than the resolved `return_serial_correlation` policy;
- use a joint generator with nonzero factor-specific cross-covariance while retaining the baseline asset-covariance identity;
- represent the Gaussian, independence, no-serial-correlation, or raw-return defaults as properties of the factor risk model;
- use the `raw` price-return default when the modeled-return metadata is incompatible with a raw arithmetic price return;
- apply square-root-of-time or linear covariance scaling;
- clip, floor, truncate, reject, or resample returns;
- repair a matrix or reorder identifiers silently;
- update exposures, factor volatility, factor correlation, or specific risk contrary to the resolved dynamics policies;
- invent factor classifications, asset currencies, FX paths, exchange-rate conventions, or currency conversions;
- create corporate-action adjustments or new listings without supported nondefault handlers;
- mix inputs from different model runs, universes, factor definitions, dates, horizons, or units without an explicit transformation record; or
- describe the output as a Barra expected-return model or a complete Barra price process merely because a Barra risk input was used.

## 9. Language-neutral pseudocode

### 9.1 One-period simulation

```text
FUNCTION simulate_one_period(
    frozen_state,
    period_policy,
    simulated_date,
    prior_price,
    prior_asset_state,
    rng_state
):
    X <- frozen_state.X                         # N x K
    F <- frozen_state.F                         # K x K, one-day covariance
    D <- frozen_state.D                         # N x N, diagonal, one-day covariance

    VALIDATE_FROZEN_STATE_AND_POLICIES(frozen_state, period_policy)

    eta_f, eta_epsilon, rng_state <- DRAW_INNOVATIONS(
        factor_distribution = period_policy.factor_return_distribution,
        specific_distribution = period_policy.specific_return_distribution,
        joint_policy = period_policy.return_joint_distribution,
        serial_policy = period_policy.return_serial_correlation,
        factor_covariance = F,
        specific_covariance = D,
        cross_covariance_target = ZERO_MATRIX(K, N),
        temporal_state = rng_state.temporal_state,
        rng_state = rng_state
    )

    f <- period_policy.factor_mean + eta_f
    epsilon <- period_policy.specific_mean + eta_epsilon
    r <- X * f + epsilon
    ASSERT_NEAR(r, X * f + epsilon, equation_tolerance)

    mapped_return, return_tag, asset_state_events <-
    MAP_MODELED_RETURN_TO_PRICE_LINK_RETURN(
        modeled_return = r,
        simulated_date = simulated_date,
        price_return_convention = period_policy.price_return_convention,
        mapping_policy = period_policy.return_mapping,
        currency_handling = period_policy.currency_handling,
        corporate_action_adjustment = period_policy.corporate_action_adjustment,
        prior_asset_state = prior_asset_state
    )

    IF return_tag.convention == ARITHMETIC:
        IF ANY(mapped_return <= -1):
            mapped_return <- APPLY_AUTHORIZED_POLICY_OR_FAIL(mapped_return)
        price <- prior_price * (1 + mapped_return)
    ELSE IF return_tag.convention == LOG:
        price <- prior_price * EXP(mapped_return)
    ELSE:
        FAIL("Price-link return convention is missing or unsupported")

    asset_state <- APPLY_ASSET_STATE_EVENTS(prior_asset_state, asset_state_events)
    VALIDATE_PRICE_AND_ASSET_STATE(price, asset_state)
    RETURN period_output, period_diagnostics, rng_state
END FUNCTION
```

### 9.2 Multi-period simulation

```text
FUNCTION simulate_paths(initial_state, policy, future_dates, number_of_paths, seed):
    resolved_policy <- APPLY_DOCUMENTED_DEFAULTS(policy)
    RECORD_EXPLICIT_VERSUS_DEFAULT_SOURCE(resolved_policy)
    VALIDATE_COMPLETE_CONFIGURATION(initial_state, resolved_policy, future_dates)
    REQUIRE_BASELINE_DYNAMICS_ARE_FROZEN(resolved_policy)
    REQUIRE_NEW_LISTING_POLICY_IS_NONE(resolved_policy)
    VALIDATE_CURRENCY_AND_CORPORATE_ACTION_HANDLERS(resolved_policy)

    X0 <- COPY(initial_state.X)
    F0 <- COPY(initial_state.F)
    D0 <- COPY(initial_state.D)
    Sigma0 <- X0 * F0 * TRANSPOSE(X0) + D0

    VALIDATE_RISK_MATRICES(X0, F0, D0, Sigma0)
    master_rng <- INITIALIZE_RECORDED_GENERATOR(seed)
    outputs <- EMPTY_COLLECTION()

    FOR path_id FROM 1 TO number_of_paths:
        price <- COPY(initial_state.P0)
        asset_state <- INITIALIZE_ALL_ASSETS_AS_LIVE(initial_state.asset_ids)
        rng_state <- ALLOCATE_REPRODUCIBLE_PATH_STREAM(master_rng, path_id)
        STORE_INITIAL_PRICE_AND_ASSET_STATE(outputs, path_id, price, asset_state)

        FOR h FROM 1 TO LENGTH(future_dates):
            period_policy <- RESOLVE_POLICY_FOR_STEP(resolved_policy, h)

            period_output, diagnostics, rng_state <- simulate_one_period(
                frozen_state = {X: X0, F: F0, D: D0},
                period_policy = period_policy,
                simulated_date = future_dates[h],
                prior_price = price,
                prior_asset_state = asset_state,
                rng_state = rng_state
            )

            APPEND(outputs, path_id, future_dates[h], period_output, diagnostics)
            price <- period_output.price
            asset_state <- period_output.asset_state
        END FOR
    END FOR

    ASSERT_FROZEN_STATE_UNCHANGED(X0, F0, D0)
    aggregate_diagnostics <- VALIDATE_MONTE_CARLO_MOMENTS(outputs, Sigma0, resolved_policy)
    RETURN outputs, aggregate_diagnostics, resolved_policy_manifest
END FUNCTION
```

## 10. Acceptance criteria

An implementation is mathematically consistent with this specification only if all of the following are true:

1. It accepts either Barra or custom risk inputs through the same documented, model-agnostic interface.
2. It rejects incompatible dimensions, identifier order, dates, horizons, units, nonfinite values, negative specific variances, and materially non-PSD covariance inputs.
3. It resolves the four dynamics policies to `frozen` by default and keeps $\mathbf X_0$, $\mathbf F_0$, and $\mathbf D_0$ unchanged over every simulated path and future date; unsupported non-frozen dynamics are rejected.
4. Every emitted modeled return satisfies $\mathbf r_h=\mathbf X_0\mathbf f_h+\boldsymbol\epsilon_h$ within the declared numerical tolerance.
5. Conditional Monte Carlo tests reproduce the configured mean vectors, factor covariance, diagonal specific covariance, factor-specific joint policy, serial policy, and resulting asset covariance within statistically justified tolerances; `joint` generation retains zero factor-specific cross-covariance under the baseline identity.
6. Omitted policy fields resolve exactly to the documented Gaussian, independent-block, no-serial-correlation, raw-return, frozen-dynamics, no-currency-handling, no-corporate-action, and no-new-listing defaults; required mean vectors, random controls, risk metadata, asset currencies, initial prices, calendar, and invalid-return policy remain mandatory.
7. It records whether every policy value was explicit or defaulted and does not present a simulator default as a property of the factor risk model.
8. Raw arithmetic and explicitly configured alternative return conventions produce prices using their respective declared links; the resolved currency policy is applied; and every emitted live price is finite and strictly positive.
9. A fixed seed and recorded random-stream policy make the run reproducible to the declared standard.
10. Outputs include the complete frozen state, generated arrays, ordered identifiers and dates, optional factor metadata, asset currencies, conditional `asset_state` events, diagnostics, provenance, and resolved-policy manifest needed to audit and reproduce the run.

