"""Hedonic (quality-adjusted) price index, in pure Python.

The problem this solves: the median ask for a reference moves when the *mix*
of listings changes, not only when the market moves. A month with more unworn
full-set examples looks like appreciation. Comparing raw medians across
periods therefore measures inventory composition as much as price.

The fix is the standard time-dummy hedonic regression:

    log(price_i) = a + SUM_k b_k * x_ik + SUM_t d_t * D_it + e_i

where x are quality characteristics (condition, papers, box, age, country,
seller type) and D are period dummies with one period omitted as the base.
The index is then exp(d_t), rescaled so the base period equals 100. Because
the quality characteristics are held in the model, d_t reflects the price of a
*constant-quality* watch over time, which is the thing you actually want.

No numpy: the design matrices here are small (hundreds of columns at most) and
avoiding the dependency keeps the whole project runnable with a bare Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# Ridge penalty applied to every non-intercept column. Dummy-heavy design
# matrices are frequently rank-deficient (a reference that appears in only one
# period is perfectly collinear with that period's dummy); a small penalty
# makes the solve stable instead of throwing.
DEFAULT_RIDGE = 1e-6


class SingularMatrixError(ValueError):
    """Raised when the normal equations cannot be solved even with ridge."""


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise SingularMatrixError(f"column {col} is numerically singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot_val
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = aug[row][n] - sum(aug[row][k] * out[k] for k in range(row + 1, n))
        out[row] = total / aug[row][row]
    return out


def invert(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse, used only when standard errors are requested."""
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise SingularMatrixError(f"column {col} is numerically singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        aug[col] = [v / pivot_val for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            aug[row] = [v - factor * p for v, p in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


@dataclass
class Feature:
    """One column-group in the design matrix.

    ``kind='categorical'`` expands to one dummy per level minus the base level.
    Missing values become their own ``'(unknown)'`` level rather than being
    imputed or dropped -- an unknown papers status is genuinely different
    information from a known-absent one, and conflating them biases the
    estimated papers premium.
    """

    name: str
    kind: str = "categorical"  # 'categorical' | 'numeric'
    getter: Callable[[dict[str, Any]], Any] | None = None
    base_level: Any = None

    def value(self, row: dict[str, Any]) -> Any:
        return self.getter(row) if self.getter else row.get(self.name)


@dataclass
class FitResult:
    columns: list[str]
    coefficients: list[float]
    std_errors: list[float] | None
    n_obs: int
    r_squared: float
    residual_sd: float
    dropped: list[str] = field(default_factory=list)

    def coef(self, column: str) -> float | None:
        try:
            return self.coefficients[self.columns.index(column)]
        except ValueError:
            return None

    def se(self, column: str) -> float | None:
        if self.std_errors is None:
            return None
        try:
            return self.std_errors[self.columns.index(column)]
        except ValueError:
            return None


def build_design(
    rows: Sequence[dict[str, Any]], features: Sequence[Feature]
) -> tuple[list[list[float]], list[str]]:
    """Expand rows into a design matrix with a leading intercept column."""
    columns: list[str] = ["(intercept)"]
    expansions: list[tuple[Feature, list[Any]]] = []

    for feature in features:
        if feature.kind == "numeric":
            columns.append(feature.name)
            expansions.append((feature, []))
            continue
        levels = sorted({_level(feature.value(r)) for r in rows}, key=str)
        if feature.base_level is not None and feature.base_level in levels:
            base = feature.base_level
        else:
            # Never make '(unknown)' the omitted base level: the intercept
            # would then mean "a watch whose papers status we don't know",
            # which makes every other coefficient awkward to interpret.
            known = [lv for lv in levels if lv != "(unknown)"]
            base = (known or levels or [None])[0]
        kept = [lv for lv in levels if lv != base]
        columns.extend(f"{feature.name}={lv}" for lv in kept)
        expansions.append((feature, kept))

    matrix: list[list[float]] = []
    for row in rows:
        vector = [1.0]
        for feature, kept in expansions:
            if feature.kind == "numeric":
                raw = feature.value(row)
                vector.append(float(raw) if raw is not None else 0.0)
            else:
                level = _level(feature.value(row))
                vector.extend(1.0 if level == lv else 0.0 for lv in kept)
        matrix.append(vector)
    return matrix, columns


def _level(value: Any) -> Any:
    return "(unknown)" if value is None or value == "" else value


def fit_ols(
    matrix: list[list[float]],
    target: list[float],
    columns: list[str],
    ridge: float = DEFAULT_RIDGE,
    with_se: bool = False,
) -> FitResult:
    """Fit (X'X + ridge*I) b = X'y, leaving the intercept unpenalised."""
    n = len(matrix)
    if n == 0:
        raise ValueError("no observations")
    p = len(matrix[0])
    if n <= p:
        raise ValueError(f"under-determined: {n} observations for {p} parameters")

    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for row, y in zip(matrix, target):
        for i in range(p):
            ri = row[i]
            if ri == 0.0:
                continue
            xty[i] += ri * y
            target_row = xtx[i]
            for j in range(i, p):
                target_row[j] += ri * row[j]
    for i in range(p):
        for j in range(i):
            xtx[i][j] = xtx[j][i]
    for i in range(1, p):
        xtx[i][i] += ridge

    beta = solve([row[:] for row in xtx], xty[:])

    mean_y = sum(target) / n
    sst = sum((y - mean_y) ** 2 for y in target)
    ssr = 0.0
    for row, y in zip(matrix, target):
        prediction = sum(c * v for c, v in zip(beta, row))
        ssr += (y - prediction) ** 2
    r_squared = 1.0 - ssr / sst if sst > 0 else 0.0
    dof = max(n - p, 1)
    residual_sd = math.sqrt(ssr / dof)

    std_errors = None
    if with_se:
        try:
            inverse = invert([row[:] for row in xtx])
            std_errors = [math.sqrt(max(inverse[i][i], 0.0)) * residual_sd for i in range(p)]
        except SingularMatrixError:
            std_errors = None

    return FitResult(
        columns=columns,
        coefficients=beta,
        std_errors=std_errors,
        n_obs=n,
        r_squared=r_squared,
        residual_sd=residual_sd,
    )


@dataclass
class IndexPoint:
    period: str
    value: float
    n_obs: int
    ci_low: float | None = None
    ci_high: float | None = None


def time_dummy_index(
    rows: Sequence[dict[str, Any]],
    features: Sequence[Feature],
    price_key: str = "price",
    period_key: str = "period",
    base_period: str | None = None,
    ridge: float = DEFAULT_RIDGE,
    with_ci: bool = True,
    min_obs_per_period: int = 3,
) -> tuple[list[IndexPoint], FitResult]:
    """Fit a quality-adjusted index, base period = 100.

    Periods with fewer than ``min_obs_per_period`` observations are dropped
    before fitting: a period represented by one listing produces a dummy that
    fits that listing's residual exactly and reports a wild index move.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[period_key]] = counts.get(row[period_key], 0) + 1
    usable = [r for r in rows if counts[r[period_key]] >= min_obs_per_period and r.get(price_key)]
    if not usable:
        raise ValueError("no periods meet min_obs_per_period")

    periods = sorted({r[period_key] for r in usable})
    base = base_period if base_period in periods else periods[0]

    period_feature = Feature(name="period", kind="categorical", base_level=base,
                             getter=lambda r: r[period_key])
    all_features = list(features) + [period_feature]

    matrix, columns = build_design(usable, all_features)
    target = [math.log(r[price_key]) for r in usable]
    fit = fit_ols(matrix, target, columns, ridge=ridge, with_se=with_ci)

    points: list[IndexPoint] = []
    for period in periods:
        n_obs = counts[period]
        if period == base:
            points.append(IndexPoint(period=period, value=100.0, n_obs=n_obs,
                                     ci_low=100.0, ci_high=100.0))
            continue
        column = f"period={period}"
        delta = fit.coef(column)
        if delta is None:
            continue
        value = 100.0 * math.exp(delta)
        low = high = None
        se = fit.se(column)
        if se is not None:
            low = 100.0 * math.exp(delta - 1.96 * se)
            high = 100.0 * math.exp(delta + 1.96 * se)
        points.append(IndexPoint(period=period, value=value, n_obs=n_obs,
                                 ci_low=low, ci_high=high))
    points.sort(key=lambda p: p.period)
    return points, fit


def standard_features() -> list[Feature]:
    """The quality controls that matter most for wristwatches.

    Age is entered numerically (a linear depreciation-ish term in log space)
    while everything else is categorical. Country is included because VAT and
    import treatment create persistent double-digit price differences for
    physically identical watches.
    """
    return [
        Feature("condition", "categorical", base_level="very_good"),
        Feature("has_papers", "categorical"),
        Feature("has_box", "categorical"),
        Feature("seller_country", "categorical"),
        Feature("seller_type", "categorical"),
        Feature("age_years", "numeric", getter=lambda r: r.get("age_years") or 0.0),
    ]


def annualised_return(points: Iterable[IndexPoint]) -> float | None:
    """CAGR implied by the first and last index points, assuming monthly periods."""
    ordered = sorted(points, key=lambda p: p.period)
    if len(ordered) < 2:
        return None
    first, last = ordered[0], ordered[-1]
    months = _month_diff(first.period, last.period)
    if months <= 0 or first.value <= 0:
        return None
    return (last.value / first.value) ** (12.0 / months) - 1.0


def _month_diff(a: str, b: str) -> int:
    ya, ma = int(a[:4]), int(a[5:7])
    yb, mb = int(b[:4]), int(b[5:7])
    return (yb - ya) * 12 + (mb - ma)
