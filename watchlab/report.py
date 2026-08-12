"""Multi-reference ranking: which references' quality-adjusted price moved most
(and least) over their own observed listings history.

Every command elsewhere in this project (``index``, ``screen``) looks at one
reference or ranks by supply-side metrics that never touch price. This is the
missing piece for a question like "top 10 / bottom 10 sub-GBP1000 watches by
return" -- run the hedonic index per reference, net it against round-trip
costs, and rank. It leans entirely on :mod:`hedonic` and :mod:`metrics`; there
is no separate estimator here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Sequence

from . import hedonic, ingest, metrics


@dataclass
class RankedReference:
    reference: str
    brand: str | None
    n_obs: int
    median_ask_eur: float | None
    gross_cagr: float | None = None
    net_cagr: float | None = None
    r_squared: float | None = None
    skip_reason: str | None = None


def rank_references(
    conn: sqlite3.Connection,
    references: Sequence[str],
    max_price_eur: float | None = None,
    min_obs: int = 12,
    costs: metrics.CostModel | None = None,
) -> list[RankedReference]:
    """Fit a hedonic index per reference and attach a net-of-cost CAGR.

    A reference without enough fresh listings to fit an index is *skipped*,
    not errored on -- for a curated watchlist, "no data yet" is the common
    case, not a bug, and the caller needs to see which references those are
    rather than have them silently vanish from the ranking.
    """
    costs = costs or metrics.CostModel()
    out = []
    for reference in references:
        ref_metrics = metrics.reference_metrics(conn, reference)
        median_ask = ref_metrics.median_ask_eur
        rows = ingest.hedonic_rows(conn, reference=reference)

        if max_price_eur is not None and median_ask is not None and median_ask > max_price_eur:
            out.append(RankedReference(
                reference, ref_metrics.brand, len(rows), median_ask,
                skip_reason=f"median ask EUR {median_ask:,.0f} over the EUR {max_price_eur:,.0f} ceiling",
            ))
            continue

        if len(rows) < min_obs:
            out.append(RankedReference(
                reference, ref_metrics.brand, len(rows), median_ask,
                skip_reason=f"only {len(rows)} fresh listings (need {min_obs})",
            ))
            continue

        try:
            points, fit = hedonic.time_dummy_index(rows, hedonic.standard_features())
        except (hedonic.SingularMatrixError, ValueError) as exc:
            out.append(RankedReference(
                reference, ref_metrics.brand, len(rows), median_ask, skip_reason=f"index failed: {exc}",
            ))
            continue

        gross = hedonic.annualised_return(points)
        net = None
        if gross is not None and median_ask:
            years = hedonic.month_diff(points[0].period, points[-1].period) / 12.0
            if years > 0:
                net = metrics.net_of_costs(gross, median_ask, years, costs)

        out.append(RankedReference(
            reference, ref_metrics.brand, len(rows), median_ask, gross, net, fit.r_squared,
        ))
    return out


def top_bottom(
    ranked: Sequence[RankedReference], n: int = 10
) -> tuple[list[RankedReference], list[RankedReference]]:
    """Split ranked (non-skipped) references into (top n, bottom n) by net CAGR.

    On a small watchlist the two lists can overlap -- with 12 scored
    references and n=10, the "bottom 10" and "top 10" share 8 entries. That's
    an honest reflection of a thin universe, not something to paper over.
    """
    scored = sorted(
        (r for r in ranked if r.net_cagr is not None), key=lambda r: r.net_cagr, reverse=True,
    )
    return scored[:n], list(reversed(scored[-n:]))
