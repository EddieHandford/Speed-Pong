"""Genetic-algorithm-tuned reference screen, validated walk-forward against
equal-weight buy-and-hold.

Deliberately last in this project's build order (see the top-level README's
"Not built yet" section, written before this module existed) because a GA
over a few thousand noisy listings is an efficient way to fit the past and
predict nothing. This module exists to build it the way that section insists
it must be if it gets built at all:

  - **walk-forward validated**: the weight vector is evolved on an earlier
    window of checkpoints and scored, unseen, on a later one. Nothing here
    is ever fit and evaluated on the same data.
  - **complexity-penalised**: fitness subtracts an L1 penalty on the weight
    vector, so the search prefers a smaller weight vector over an equally
    fit, sprawling one.
  - **costed**: every simulated position nets out round-trip friction
    through the same :class:`metrics.CostModel` used everywhere else in
    this project -- a GA free to ignore transaction costs would evolve
    high-turnover nonsense.
  - **honestly compared**: the out-of-sample result is always reported next
    to equal-weight buy-and-hold over the identical test window. Per the
    README, buy-and-hold will probably win. This module does not hide that
    outcome; :func:`run_ga` returns both numbers, and the CLI prints both.

What this evolves is not a trading strategy -- watches are not a
day-tradeable asset; a single round trip already costs ~18-20%, so anything
resembling frequent rebalancing is economically absurd for this asset class.
It evolves a *scoring function*: a weight vector over the same microstructure
features :func:`metrics.screen` already computes (plus one price-momentum
term), used to rank references at a decision date and select the top K to
hold for one horizon. That makes it a GA-tuned generalisation of
``metrics._composite_score``, not a market-timing system.
"""

from __future__ import annotations

import datetime as _dt
import random
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from . import hedonic, ingest, metrics

FEATURES = ("days_on_market", "cut_rate", "dispersion", "churn_ratio", "momentum")


@dataclass
class Checkpoint:
    period: str
    features: dict[str, dict[str, float]]      # reference -> {feature: z-scored value}
    forward_gross_return: dict[str, float]      # reference -> (index[T+h]/index[T]) - 1.0
    price_eur: dict[str, float]                 # reference -> median ask at T, for cost scaling


@dataclass
class Candidate:
    weights: dict[str, float]

    def score(self, features: dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * features.get(name, 0.0) for name in FEATURES)


@dataclass
class GAResult:
    best_weights: dict[str, float]
    fitness_by_generation: list[float]
    train_net_return: float | None
    test_net_return: float | None
    test_buy_and_hold_net_return: float | None
    n_train_checkpoints: int
    n_test_checkpoints: int
    top_k: int
    features: tuple[str, ...] = FEATURES


def _add_months(period: str, months: int) -> str:
    year, month = int(period[:4]), int(period[5:7])
    total = (year * 12 + (month - 1)) + months
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _period_as_of(period: str) -> str:
    """Approximate 'end of this month' as-of date -- the 28th, valid in every month."""
    return f"{period}-28"


def _zscore(values: dict[str, float | None]) -> dict[str, float]:
    """Missing values become 0.0 (the cross-sectional mean) -- neutral, not penalised."""
    present = [v for v in values.values() if v is not None]
    if len(present) < 2:
        return {k: 0.0 for k in values}
    mean = statistics.mean(present)
    stdev = statistics.pstdev(present) or 1.0
    return {k: (0.0 if v is None else (v - mean) / stdev) for k, v in values.items()}


def build_checkpoints(
    conn,
    references: Sequence[str],
    checkpoint_every_months: int = 3,
    horizon_months: int = 6,
    window_days: int = 90,
) -> list[Checkpoint]:
    """Assemble the walk-forward dataset: one hedonic fit per reference, then
    period-by-period, no-lookahead features paired with a forward return.

    One index fit per reference (not per GA generation, and not per
    checkpoint) -- cheap for a watchlist-sized universe and avoids refitting
    the same regression thousands of times inside the GA loop.
    """
    index_by_ref: dict[str, dict[str, float]] = {}
    for reference in references:
        rows = ingest.hedonic_rows(conn, reference=reference)
        if len(rows) < 12:
            continue
        try:
            points, _fit = hedonic.time_dummy_index(rows, hedonic.standard_features())
        except (ValueError, hedonic.SingularMatrixError):
            continue
        index_by_ref[reference] = {p.period: p.value for p in points}

    if not index_by_ref:
        return []

    all_periods = sorted({period for series in index_by_ref.values() for period in series})
    if len(all_periods) < 3:
        return []

    checkpoints: list[Checkpoint] = []
    period = all_periods[checkpoint_every_months]  # skip the first: no trailing momentum yet
    while True:
        forward_period = _add_months(period, horizon_months)
        trailing_period = _add_months(period, -checkpoint_every_months)
        if forward_period > all_periods[-1]:
            break

        as_of = _period_as_of(period)
        raw_days_on_market: dict[str, float | None] = {}
        raw_cut_rate: dict[str, float | None] = {}
        raw_dispersion: dict[str, float | None] = {}
        raw_churn: dict[str, float | None] = {}
        raw_momentum: dict[str, float | None] = {}
        price_eur: dict[str, float] = {}
        forward_gross_return: dict[str, float] = {}

        for reference, series in index_by_ref.items():
            m = metrics.reference_metrics(conn, reference, as_of=as_of, window_days=window_days)
            raw_days_on_market[reference] = m.median_days_on_market
            raw_cut_rate[reference] = m.cut_rate
            raw_dispersion[reference] = m.dispersion
            raw_churn[reference] = m.churn_ratio
            if m.median_ask_eur:
                price_eur[reference] = m.median_ask_eur

            if period in series and trailing_period in series and series[trailing_period]:
                raw_momentum[reference] = series[period] / series[trailing_period] - 1.0
            else:
                raw_momentum[reference] = None

            if period in series and forward_period in series and series[period]:
                forward_gross_return[reference] = series[forward_period] / series[period] - 1.0

        z_dom = _zscore(raw_days_on_market)
        z_cut = _zscore(raw_cut_rate)
        z_disp = _zscore(raw_dispersion)
        z_churn = _zscore(raw_churn)
        z_mom = _zscore(raw_momentum)

        features = {
            reference: {
                "days_on_market": z_dom[reference],
                "cut_rate": z_cut[reference],
                "dispersion": z_disp[reference],
                "churn_ratio": z_churn[reference],
                "momentum": z_mom[reference],
            }
            for reference in index_by_ref
        }

        checkpoints.append(Checkpoint(period, features, forward_gross_return, price_eur))
        period = _add_months(period, checkpoint_every_months)

    return checkpoints


def _portfolio_net_return(
    checkpoint: Checkpoint, selected: Sequence[str], horizon_months: int, costs: metrics.CostModel,
) -> float | None:
    """Equal-weight net-of-cost return across ``selected``, skipping references
    the checkpoint has no forward outcome for (illiquid-market data gap, not
    a lookahead problem -- the gap is discovered after the fact, same as
    real life)."""
    years = horizon_months / 12.0
    returns = []
    for reference in selected:
        gross = checkpoint.forward_gross_return.get(reference)
        price = checkpoint.price_eur.get(reference)
        if gross is None or not price:
            continue
        returns.append(metrics.net_of_costs(gross, price, years, costs))
    return statistics.mean(returns) if returns else None


def _select_top_k(candidate: Candidate, checkpoint: Checkpoint, top_k: int) -> list[str]:
    ranked = sorted(
        checkpoint.features.items(), key=lambda kv: candidate.score(kv[1]), reverse=True,
    )
    return [reference for reference, _features in ranked[:top_k]]


def _fitness(
    candidate: Candidate, checkpoints: Sequence[Checkpoint], top_k: int,
    horizon_months: int, costs: metrics.CostModel, complexity_penalty: float,
) -> float:
    per_checkpoint = []
    for checkpoint in checkpoints:
        selected = _select_top_k(candidate, checkpoint, top_k)
        net = _portfolio_net_return(checkpoint, selected, horizon_months, costs)
        if net is not None:
            per_checkpoint.append(net)
    mean_return = statistics.mean(per_checkpoint) if per_checkpoint else -1.0
    penalty = complexity_penalty * sum(abs(w) for w in candidate.weights.values())
    return mean_return - penalty


def _buy_and_hold_net_return(
    checkpoints: Sequence[Checkpoint], horizon_months: int, costs: metrics.CostModel,
) -> float | None:
    """Equal-weight across the *whole* universe at each checkpoint, GA-blind."""
    per_checkpoint = []
    for checkpoint in checkpoints:
        net = _portfolio_net_return(checkpoint, list(checkpoint.features), horizon_months, costs)
        if net is not None:
            per_checkpoint.append(net)
    return statistics.mean(per_checkpoint) if per_checkpoint else None


def _random_weights(rng: random.Random) -> dict[str, float]:
    return {name: rng.uniform(-1.0, 1.0) for name in FEATURES}


def _crossover(a: Candidate, b: Candidate, rng: random.Random) -> Candidate:
    return Candidate({name: (a.weights[name] + b.weights[name]) / 2.0 for name in FEATURES})


def _mutate(candidate: Candidate, rate: float, rng: random.Random) -> Candidate:
    weights = dict(candidate.weights)
    for name in FEATURES:
        if rng.random() < rate:
            weights[name] = max(-2.0, min(2.0, weights[name] + rng.gauss(0.0, 0.3)))
    return Candidate(weights)


def _tournament_select(population: list[Candidate], fitnesses: list[float], rng: random.Random) -> Candidate:
    i, j = rng.randrange(len(population)), rng.randrange(len(population))
    return population[i] if fitnesses[i] >= fitnesses[j] else population[j]


def run_ga(
    conn,
    references: Sequence[str],
    population_size: int = 40,
    generations: int = 30,
    top_k: int = 3,
    checkpoint_every_months: int = 3,
    horizon_months: int = 6,
    train_fraction: float = 0.7,
    complexity_penalty: float = 0.02,
    mutation_rate: float = 0.15,
    seed: int | None = None,
    costs: metrics.CostModel | None = None,
) -> GAResult:
    """Evolve a scoring-weight vector on an earlier window, then report its
    *unseen* performance on a later one, next to equal-weight buy-and-hold.

    ``top_k`` and the checkpoint/horizon spacing matter more than population
    size or generation count here: with a watchlist-sized universe (tens of
    references, not thousands of listings) and a handful of checkpoints,
    this is a small-sample estimate. ``n_train_checkpoints`` /
    ``n_test_checkpoints`` on the result say exactly how small -- read them
    before trusting the return numbers.
    """
    costs = costs or metrics.CostModel()
    rng = random.Random(seed)

    checkpoints = build_checkpoints(
        conn, references, checkpoint_every_months=checkpoint_every_months,
        horizon_months=horizon_months,
    )
    split = max(1, int(len(checkpoints) * train_fraction))
    train_checkpoints = checkpoints[:split]
    test_checkpoints = checkpoints[split:]

    if not train_checkpoints:
        return GAResult(
            best_weights={name: 0.0 for name in FEATURES}, fitness_by_generation=[],
            train_net_return=None, test_net_return=None, test_buy_and_hold_net_return=None,
            n_train_checkpoints=0, n_test_checkpoints=len(test_checkpoints), top_k=top_k,
        )

    population = [Candidate(_random_weights(rng)) for _ in range(population_size)]
    fitness_by_generation: list[float] = []
    best = population[0]
    best_fitness = float("-inf")

    for _generation in range(generations):
        fitnesses = [
            _fitness(c, train_checkpoints, top_k, horizon_months, costs, complexity_penalty)
            for c in population
        ]
        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best = population[gen_best_idx]
        fitness_by_generation.append(best_fitness)

        next_population = [population[gen_best_idx]]  # elitism: carry the generation's best
        while len(next_population) < population_size:
            parent_a = _tournament_select(population, fitnesses, rng)
            parent_b = _tournament_select(population, fitnesses, rng)
            child = _mutate(_crossover(parent_a, parent_b, rng), mutation_rate, rng)
            next_population.append(child)
        population = next_population

    train_net_return = _fitness(best, train_checkpoints, top_k, horizon_months, costs, 0.0)
    test_net_return = None
    test_buy_and_hold = None
    if test_checkpoints:
        selected_returns = []
        for checkpoint in test_checkpoints:
            selected = _select_top_k(best, checkpoint, top_k)
            net = _portfolio_net_return(checkpoint, selected, horizon_months, costs)
            if net is not None:
                selected_returns.append(net)
        test_net_return = statistics.mean(selected_returns) if selected_returns else None
        test_buy_and_hold = _buy_and_hold_net_return(test_checkpoints, horizon_months, costs)

    return GAResult(
        best_weights=best.weights,
        fitness_by_generation=fitness_by_generation,
        train_net_return=train_net_return,
        test_net_return=test_net_return,
        test_buy_and_hold_net_return=test_buy_and_hold,
        n_train_checkpoints=len(train_checkpoints),
        n_test_checkpoints=len(test_checkpoints),
        top_k=top_k,
    )
