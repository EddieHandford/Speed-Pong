"""Market-microstructure metrics computed from listing snapshots.

None of these require knowing a sale price, which is what makes them usable
on listings data. They describe *supply and seller behaviour*, and in illiquid
collectible markets those lead the price index rather than following it:

  supply          how many examples are competing for a buyer right now
  days_on_market  how long an example sits before it leaves the market
  cut_rate        share of live listings that have reduced their ask
  cut_depth       median size of those reductions
  dispersion      inter-quartile range over median, i.e. how much sellers
                  disagree about what the thing is worth
  churn           new listings vs delistings over the window

A rising ask index alongside rising supply, rising DOM and a rising cut rate
is a market topping out, not a market appreciating. That combination is the
main thing this module exists to make visible.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from . import db


@dataclass(frozen=True)
class CostModel:
    """Round-trip friction. Defaults are deliberately pessimistic.

    A watch bought retail-adjacent and sold through a marketplace loses roughly
    a fifth of its value to fees, shipping, insurance and the bid-ask spread
    before any tax. Any screen that ignores this will happily recommend
    positions with negative expected value.
    """

    buy_premium_pct: float = 0.02      # authentication / escrow / payment fees
    sell_commission_pct: float = 0.065  # marketplace seller commission
    spread_pct: float = 0.10            # realistic gap between ask and clearing bid
    shipping_insurance_eur: float = 250.0
    servicing_eur_per_year: float = 0.0  # set per-brand if you care

    def round_trip_pct(self) -> float:
        return self.buy_premium_pct + self.sell_commission_pct + self.spread_pct

    def hurdle_rate(self, price_eur: float, years: float = 1.0) -> float:
        """Gross appreciation needed over ``years`` just to break even."""
        if price_eur <= 0 or years <= 0:
            return float("inf")
        fixed = (self.shipping_insurance_eur * 2 + self.servicing_eur_per_year * years) / price_eur
        total = self.round_trip_pct() + fixed
        return (1.0 + total) ** (1.0 / years) - 1.0


@dataclass
class ReferenceMetrics:
    reference: str
    brand: str | None
    n_live: int
    n_delisted_window: int
    median_ask_eur: float | None
    dispersion: float | None
    median_days_on_market: float | None
    cut_rate: float | None
    median_cut_depth: float | None
    new_listings: int
    delistings: int
    churn_ratio: float | None
    liquidity_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso(day: _dt.date) -> str:
    return day.isoformat()


def _window_bounds(as_of: str | None, days: int) -> tuple[str, str]:
    end = _dt.date.fromisoformat(as_of) if as_of else _dt.date.today()
    return _iso(end - _dt.timedelta(days=days)), _iso(end)


def median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def dispersion(values: Sequence[float]) -> float | None:
    """Inter-quartile range divided by the median.

    Scale-free, so a €5k Tudor and a €150k Patek are comparable, and far more
    robust than standard deviation on data where one mispriced listing can sit
    three times above the rest.
    """
    if len(values) < 4:
        return None
    ordered = sorted(values)
    q1, q3 = _quantile(ordered, 0.25), _quantile(ordered, 0.75)
    med = statistics.median(ordered)
    return (q3 - q1) / med if med else None


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        raise ValueError("empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def live_asks_eur(conn: sqlite3.Connection, reference: str, as_of: str | None = None) -> list[float]:
    """Latest observed ask for every listing still live at ``as_of``, in EUR."""
    as_of = as_of or _iso(_dt.date.today())
    rows = conn.execute(
        """
        SELECT l.listing_id, l.currency, s.price_cents, s.observed_at
        FROM listings l
        JOIN listing_snapshots s ON s.listing_id = l.listing_id
        WHERE l.reference = ?
          AND s.observed_at <= ?
          AND (l.delisted_at IS NULL OR l.delisted_at > ?)
          AND s.observed_at = (
              SELECT MAX(s2.observed_at) FROM listing_snapshots s2
              WHERE s2.listing_id = l.listing_id AND s2.observed_at <= ?
          )
        """,
        (reference, as_of, as_of, as_of),
    ).fetchall()
    return [db.to_base(r["price_cents"], r["currency"], conn, r["observed_at"]) / 100.0 for r in rows]


def days_on_market(
    conn: sqlite3.Connection, reference: str, as_of: str | None = None, window_days: int = 180
) -> list[float]:
    """Lifespans of listings that disappeared inside the window.

    A disappearance is NOT a sale. Sellers delist, relist, and let listings
    expire. Treat this as a liquidity proxy -- how long the market takes to
    absorb an example -- and never as a transaction record.
    """
    start, end = _window_bounds(as_of, window_days)
    rows = conn.execute(
        """
        SELECT first_seen_at, delisted_at FROM listings
        WHERE reference = ? AND delisted_at IS NOT NULL
          AND delisted_at BETWEEN ? AND ?
        """,
        (reference, start, end),
    ).fetchall()
    out = []
    for row in rows:
        first = _dt.date.fromisoformat(row["first_seen_at"])
        gone = _dt.date.fromisoformat(row["delisted_at"])
        out.append((gone - first).days)
    return [float(d) for d in out if d >= 0]


def price_cuts(
    conn: sqlite3.Connection, reference: str, as_of: str | None = None, window_days: int = 90
) -> tuple[float | None, float | None]:
    """Return (share of listings that cut their ask, median cut depth).

    Computed against each listing's own maximum ask inside the window, so a
    seller who lists high and grinds down is captured even if they never post
    a single dramatic reduction.
    """
    start, end = _window_bounds(as_of, window_days)
    rows = conn.execute(
        """
        SELECT s.listing_id, s.price_cents, s.currency, s.observed_at
        FROM listing_snapshots s
        JOIN listings l ON l.listing_id = s.listing_id
        WHERE l.reference = ? AND s.observed_at BETWEEN ? AND ?
        ORDER BY s.listing_id, s.observed_at
        """,
        (reference, start, end),
    ).fetchall()

    by_listing: dict[str, list[float]] = {}
    for row in rows:
        eur = db.to_base(row["price_cents"], row["currency"], conn, row["observed_at"]) / 100.0
        by_listing.setdefault(row["listing_id"], []).append(eur)

    if not by_listing:
        return None, None

    depths = []
    for series in by_listing.values():
        if len(series) < 2:
            continue
        peak = max(series)
        last = series[-1]
        if peak > 0 and last < peak:
            depths.append((peak - last) / peak)

    tracked = sum(1 for s in by_listing.values() if len(s) >= 2)
    if tracked == 0:
        return None, None
    return len(depths) / tracked, median_or_none(depths)


def churn(
    conn: sqlite3.Connection, reference: str, as_of: str | None = None, window_days: int = 90
) -> tuple[int, int]:
    """(new listings, delistings) inside the window."""
    start, end = _window_bounds(as_of, window_days)
    new = conn.execute(
        "SELECT COUNT(*) AS n FROM listings WHERE reference = ? AND first_seen_at BETWEEN ? AND ?",
        (reference, start, end),
    ).fetchone()["n"]
    gone = conn.execute(
        "SELECT COUNT(*) AS n FROM listings WHERE reference = ? AND delisted_at BETWEEN ? AND ?",
        (reference, start, end),
    ).fetchone()["n"]
    return int(new), int(gone)


def reference_metrics(
    conn: sqlite3.Connection, reference: str, as_of: str | None = None, window_days: int = 90
) -> ReferenceMetrics:
    as_of = as_of or _iso(_dt.date.today())
    asks = live_asks_eur(conn, reference, as_of)
    doms = days_on_market(conn, reference, as_of, window_days=max(window_days, 180))
    cut_rate, cut_depth = price_cuts(conn, reference, as_of, window_days)
    new, gone = churn(conn, reference, as_of, window_days)

    brand_row = conn.execute(
        "SELECT brand FROM listings WHERE reference = ? AND brand IS NOT NULL LIMIT 1", (reference,)
    ).fetchone()

    median_dom = median_or_none(doms)
    # Fast turnover and thin inventory both signal a tight market. Scaled so
    # that ~30 days on market with ~10 live examples lands near 1.0.
    liquidity = None
    if median_dom and median_dom > 0 and asks:
        liquidity = round((30.0 / median_dom) * (10.0 / max(len(asks), 1)) ** 0.5, 3)

    return ReferenceMetrics(
        reference=reference,
        brand=brand_row["brand"] if brand_row else None,
        n_live=len(asks),
        n_delisted_window=len(doms),
        median_ask_eur=median_or_none(asks),
        dispersion=dispersion(asks),
        median_days_on_market=median_dom,
        cut_rate=cut_rate,
        median_cut_depth=cut_depth,
        new_listings=new,
        delistings=gone,
        churn_ratio=(new / gone) if gone else None,
        liquidity_score=liquidity,
    )


def all_references(conn: sqlite3.Connection, min_listings: int = 5) -> list[str]:
    rows = conn.execute(
        """
        SELECT reference, COUNT(*) AS n FROM listings
        WHERE reference IS NOT NULL
        GROUP BY reference HAVING n >= ?
        ORDER BY n DESC
        """,
        (min_listings,),
    ).fetchall()
    return [r["reference"] for r in rows]


def screen(
    conn: sqlite3.Connection,
    as_of: str | None = None,
    window_days: int = 90,
    min_listings: int = 5,
    costs: CostModel | None = None,
) -> list[dict[str, Any]]:
    """Rank references by a composite of supply pressure and liquidity.

    The score is intentionally NOT a return forecast. It is a screen: it
    surfaces references whose supply-side behaviour looks tight, so you have
    somewhere to point your own judgement. Anything claiming to be a forecast
    here would be lying about what listings data can support.
    """
    costs = costs or CostModel()
    out = []
    for reference in all_references(conn, min_listings):
        metrics = reference_metrics(conn, reference, as_of, window_days)
        row = metrics.to_dict()
        row["hurdle_1y"] = (
            round(costs.hurdle_rate(metrics.median_ask_eur, 1.0), 4)
            if metrics.median_ask_eur else None
        )
        row["score"] = _composite_score(metrics)
        out.append(row)
    out.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))
    return out


def _composite_score(m: ReferenceMetrics) -> float | None:
    """Blend the supply-side signals into one comparable number in [0, 1].

    Each component is mapped to [0, 1] where 1 is 'tight market'. Components
    that are unavailable are skipped rather than defaulted, and the result is
    the mean of whatever was available -- so a reference with only two usable
    components is scored on those two, and ``n_live`` tells you how much to
    trust it.
    """
    parts: list[float] = []
    if m.median_days_on_market is not None and m.median_days_on_market > 0:
        parts.append(_clamp(60.0 / m.median_days_on_market, 0.0, 2.0) / 2.0)
    if m.cut_rate is not None:
        parts.append(1.0 - _clamp(m.cut_rate, 0.0, 1.0))
    if m.dispersion is not None:
        parts.append(1.0 - _clamp(m.dispersion / 0.5, 0.0, 1.0))
    if m.churn_ratio is not None:
        # Fewer new listings than delistings means inventory is draining.
        parts.append(1.0 - _clamp(m.churn_ratio / 2.0, 0.0, 1.0))
    if not parts:
        return None
    return round(sum(parts) / len(parts), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def net_of_costs(gross_return: float, price_eur: float, years: float, costs: CostModel) -> float:
    """Convert a gross annualised return into a net one after round-trip costs."""
    gross_total = (1.0 + gross_return) ** years
    friction = costs.round_trip_pct() + (
        costs.shipping_insurance_eur * 2 + costs.servicing_eur_per_year * years
    ) / max(price_eur, 1.0)
    net_total = gross_total * (1.0 - friction)
    if net_total <= 0:
        return -1.0
    return net_total ** (1.0 / years) - 1.0
