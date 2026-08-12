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

from . import auctions, db, hedonic, ingest, metrics

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
                self._send(
                    {
                        "rows": metrics.screen(conn, window_days=window),
                        "costs": metrics.CostModel().__dict__,
                    }
                )
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
