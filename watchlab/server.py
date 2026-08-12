"""Local dashboard: stdlib HTTP server with a small JSON API.

Binds to 127.0.0.1 by default. There is no authentication and none is planned;
this is a tool that reads a local SQLite file, and it should not be exposed to
a network.
"""

from __future__ import annotations

import json
import os
import statistics
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import auctions, db, genetic, hedonic, ingest, metrics, report

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


def _index_payload(conn, reference: str) -> dict[str, Any]:
    """Hedonic index for one reference, alongside the naive median it corrects."""
    rows = ingest.hedonic_rows(conn, reference=reference)
    if len(rows) < 12:
        return {"reference": reference, "error": "not enough observations", "points": []}

    try:
        points, fit = hedonic.time_dummy_index(rows, hedonic.standard_features())
    except (ValueError, hedonic.SingularMatrixError) as exc:
        return {"reference": reference, "error": str(exc), "points": []}

    by_period: dict[str, list[float]] = {}
    for row in rows:
        by_period.setdefault(row["period"], []).append(row["price"])

    periods = [p.period for p in points]
    base_median = statistics.median(by_period[periods[0]]) if periods else None

    series = []
    for point in points:
        naive = None
        if base_median and point.period in by_period:
            naive = 100.0 * statistics.median(by_period[point.period]) / base_median
        series.append(
            {
                "period": point.period,
                "hedonic": round(point.value, 2),
                "naive": round(naive, 2) if naive is not None else None,
                "n": point.n_obs,
                "ci_low": round(point.ci_low, 2) if point.ci_low else None,
                "ci_high": round(point.ci_high, 2) if point.ci_high else None,
            }
        )

    cagr = hedonic.annualised_return(points)
    costs = metrics.CostModel()
    # Cost hurdles scale with price, so use the current level, not the average
    # across the whole history.
    median_price = statistics.median(by_period[periods[-1]]) if periods else 0.0
    return {
        "reference": reference,
        "points": series,
        "r_squared": round(fit.r_squared, 3),
        "n_obs": fit.n_obs,
        "cagr": round(cagr, 4) if cagr is not None else None,
        "hurdle_1y": round(costs.hurdle_rate(median_price, 1.0), 4),
        "net_cagr": (
            round(metrics.net_of_costs(cagr, median_price, 1.0, costs), 4)
            if cagr is not None else None
        ),
        "papers_premium": _premium(fit, "has_papers=1"),
        "condition_premiums": {
            column.split("=", 1)[1]: round(fit.coefficients[i], 4)
            for i, column in enumerate(fit.columns) if column.startswith("condition=")
        },
    }


def _auction_index_payload(conn, reference: str) -> dict[str, Any]:
    """Transaction-based index for one reference, alongside the naive median it corrects.

    Deliberately NOT merged into ``_index_payload``: auction periods are
    quarterly by default (auction volume is far lower than listing volume)
    while the ask-based index is monthly, and the two series come from
    different underlying data with different biases. Keeping them as
    separate payloads (and separate chart panels in the dashboard) avoids
    implying a period-by-period comparability that isn't there -- see
    auctions.py's module docstring and tests/test_auctions.py for why
    "transactions beat asks" is not a claim this project makes.
    """
    rows = auctions.auction_hedonic_rows(conn, reference=reference)
    if len(rows) < 12:
        return {"reference": reference, "error": "not enough transactions", "points": []}

    try:
        points, fit = hedonic.time_dummy_index(
            rows, hedonic.auction_features(), ridge=auctions.DEFAULT_AUCTION_RIDGE
        )
    except (ValueError, hedonic.SingularMatrixError) as exc:
        return {"reference": reference, "error": str(exc), "points": []}

    by_period: dict[str, list[float]] = {}
    houses: set[str] = set()
    for row in rows:
        by_period.setdefault(row["period"], []).append(row["price"])
        if row.get("house"):
            houses.add(row["house"])

    periods = [p.period for p in points]
    base_median = statistics.median(by_period[periods[0]]) if periods else None

    series = []
    for point in points:
        naive = None
        if base_median and point.period in by_period:
            naive = 100.0 * statistics.median(by_period[point.period]) / base_median
        series.append(
            {
                "period": point.period,
                "hedonic": round(point.value, 2),
                "naive": round(naive, 2) if naive is not None else None,
                "n": point.n_obs,
                "ci_low": round(point.ci_low, 2) if point.ci_low else None,
                "ci_high": round(point.ci_high, 2) if point.ci_high else None,
            }
        )

    cagr = hedonic.annualised_return(points)
    return {
        "reference": reference,
        "points": series,
        "r_squared": round(fit.r_squared, 3),
        "n_obs": fit.n_obs,
        "n_houses": len(houses),
        "cagr": round(cagr, 4) if cagr is not None else None,
    }


def _premium(fit: hedonic.FitResult, column: str) -> float | None:
    coefficient = fit.coef(column)
    return round(coefficient, 4) if coefficient is not None else None


def _report_payload(conn, query: dict[str, list[str]]) -> dict[str, Any]:
    """Net-of-cost hedonic return per reference, ranked -- the /report CLI command's data."""
    references = query.get("reference") or metrics.all_references(conn, min_listings=1)
    max_price = query.get("max_price_eur", [None])[0]
    shipping = float(query.get("shipping_insurance_eur", ["30"])[0])
    top_n = int(query.get("top", ["10"])[0])

    costs = metrics.CostModel(shipping_insurance_eur=shipping)
    ranked = report.rank_references(
        conn, references, max_price_eur=float(max_price) if max_price else None, costs=costs,
    )
    top, bottom = report.top_bottom(ranked, n=top_n)

    def _row(r: report.RankedReference) -> dict[str, Any]:
        return {
            "reference": r.reference, "brand": r.brand, "n_obs": r.n_obs,
            "median_ask_eur": r.median_ask_eur,
            "gross_cagr": round(r.gross_cagr, 4) if r.gross_cagr is not None else None,
            "net_cagr": round(r.net_cagr, 4) if r.net_cagr is not None else None,
            "r_squared": round(r.r_squared, 3) if r.r_squared is not None else None,
            "skip_reason": r.skip_reason,
        }

    return {
        "top": [_row(r) for r in top],
        "bottom": [_row(r) for r in bottom],
        "skipped": [_row(r) for r in ranked if r.net_cagr is None],
        "n_ranked": len(ranked) - sum(1 for r in ranked if r.net_cagr is None),
        "n_total": len(ranked),
    }


def _genetic_payload(conn, query: dict[str, list[str]]) -> dict[str, Any]:
    """Walk-forward GA screen vs equal-weight buy-and-hold -- see genetic.py before trusting this."""
    references = query.get("reference") or metrics.all_references(conn, min_listings=1)
    shipping = float(query.get("shipping_insurance_eur", ["30"])[0])
    seed = query.get("seed", [None])[0]

    costs = metrics.CostModel(shipping_insurance_eur=shipping)
    result = genetic.run_ga(
        conn, references,
        population_size=int(query.get("population", ["30"])[0]),
        generations=int(query.get("generations", ["25"])[0]),
        top_k=int(query.get("top_k", ["3"])[0]),
        checkpoint_every_months=int(query.get("checkpoint_months", ["3"])[0]),
        horizon_months=int(query.get("horizon_months", ["6"])[0]),
        complexity_penalty=float(query.get("complexity_penalty", ["0.02"])[0]),
        seed=int(seed) if seed else None,
        costs=costs,
    )
    return {
        "n_references": len(references),
        "top_k": result.top_k,
        "n_train_checkpoints": result.n_train_checkpoints,
        "n_test_checkpoints": result.n_test_checkpoints,
        "weights": {k: round(v, 4) for k, v in result.best_weights.items()},
        "fitness_by_generation": [round(f, 4) for f in result.fitness_by_generation],
        "train_net_return": (
            round(result.train_net_return, 4) if result.train_net_return is not None else None
        ),
        "test_net_return": (
            round(result.test_net_return, 4) if result.test_net_return is not None else None
        ),
        "test_buy_and_hold_net_return": (
            round(result.test_buy_and_hold_net_return, 4)
            if result.test_buy_and_hold_net_return is not None else None
        ),
    }


class Handler(BaseHTTPRequestHandler):
    db_path: str = db.DEFAULT_DB_PATH

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        return

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: str) -> None:
        try:
            with open(path, "rb") as handle:
                body = handle.read()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self._send_file(os.path.join(WEB_DIR, "dashboard.html"), "text/html; charset=utf-8")
            return

        if not parsed.path.startswith("/api/"):
            self.send_error(404)
            return

        with db.session(self.db_path) as conn:
            if parsed.path == "/api/screen":
                window = int(query.get("window", ["90"])[0])
                rows = metrics.screen(conn, window_days=window)
                brand = query.get("brand", [None])[0]
                if brand:
                    rows = [r for r in rows if (r.get("brand") or "").lower() == brand.lower()]
                max_price = query.get("max_price_eur", [None])[0]
                if max_price:
                    limit = float(max_price)
                    rows = [r for r in rows if r["median_ask_eur"] is None or r["median_ask_eur"] <= limit]
                self._send({"rows": rows, "costs": metrics.CostModel().__dict__})
            elif parsed.path == "/api/references":
                self._send({"references": metrics.all_references(conn)})
            elif parsed.path == "/api/index":
                reference = query.get("reference", [None])[0]
                if not reference:
                    self._send({"error": "reference is required"}, status=400)
                    return
                self._send(_index_payload(conn, reference))
            elif parsed.path == "/api/auction-index":
                reference = query.get("reference", [None])[0]
                if not reference:
                    self._send({"error": "reference is required"}, status=400)
                    return
                self._send(_auction_index_payload(conn, reference))
            elif parsed.path == "/api/report":
                self._send(_report_payload(conn, query))
            elif parsed.path == "/api/genetic":
                self._send(_genetic_payload(conn, query))
            else:
                self.send_error(404)


def serve(host: str = "127.0.0.1", port: int = 8765, db_path: str | None = None) -> None:
    Handler.db_path = db_path or db.DEFAULT_DB_PATH
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"watchlab dashboard on http://{host}:{port}  (db: {Handler.db_path})")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
