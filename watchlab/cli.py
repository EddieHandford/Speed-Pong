"""Command line entry point: ``python -m watchlab <command>``."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

from . import catalogue, db, hedonic, ingest, metrics
from .sources import chrono24, synthetic


def cmd_demo(args: argparse.Namespace) -> int:
    """Build a database from the simulator so the dashboard has something to show."""
    end = _dt.date.today()
    start = end - _dt.timedelta(days=args.days)
    with db.session(args.db) as conn:
        conn.execute("DELETE FROM listing_snapshots")
        conn.execute("DELETE FROM listings")
        synthetic.seed_fx(conn, start.isoformat())
        synthetic.generate(conn, start, end, seed=args.seed)
        n = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        s = conn.execute("SELECT COUNT(*) FROM listing_snapshots").fetchone()[0]
    print(f"simulated {n} listings / {s} snapshots over {args.days} days into {args.db}")
    print("note: this is SIMULATED data for validating the pipeline, not a real market.")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Show what each parsing strategy extracts from saved HTML pages."""
    total = 0
    for path, html in chrono24.iter_saved_pages(args.files):
        jsonld = chrono24.parse_jsonld(html)
        dom = chrono24.parse_dom(html)
        print(f"\n{path}  ({len(html):,} bytes)")
        print(f"  json-ld strategy: {len(jsonld)} listings")
        print(f"  dom strategy:     {len(dom)} listings")
        sample = (jsonld or dom)[: args.sample]
        for item in sample:
            print(f"    - {item.listing_id}  {(item.title or '')[:64]!r}  {item.price_text}")
        if not jsonld and not dom:
            print("    nothing matched. Fix SELECTORS in watchlab/sources/chrono24.py,")
            print("    or check whether the page is a bot-protection challenge.")
        total += len(jsonld or dom)
    print(f"\ntotal parsed: {total}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Parse saved pages and write them into the database as one observation."""
    observed = args.date or _dt.date.today().isoformat()
    raw = []
    for _, html in chrono24.iter_saved_pages(args.files):
        raw.extend(chrono24.parse_listing_html(html))
    if not raw:
        print("no listings parsed -- run 'calibrate' first", file=sys.stderr)
        return 1

    with db.session(args.db) as conn:
        report = ingest.upsert_listings(conn, raw, observed_at=observed)
        if args.complete_crawl:
            report.delisted = ingest.mark_delisted(
                conn, [item.listing_id for item in raw], observed
            )
    print(report.summary())
    if not args.complete_crawl:
        print("(no delisting pass: use --complete-crawl only when the crawl covered "
              "everything you track, or days-on-market will be corrupted)")
    return 0


def cmd_catalogue_import(args: argparse.Namespace) -> int:
    """Load reference-catalogue records (JSON or CSV) into refs/ref_aliases.

    See watchlab/catalogue.py for the expected record shape. This is provider
    agnostic: point it at a file in that shape, regardless of where the file
    came from.
    """
    try:
        records = catalogue.load_file(args.file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"could not read {args.file}: {exc}", file=sys.stderr)
        return 1

    with db.session(args.db) as conn:
        report = catalogue.upsert_catalogue(conn, records)
    print(report.summary())
    for error in report.errors[:20]:
        print(f"  ! {error}", file=sys.stderr)
    if len(report.errors) > 20:
        print(f"  ... and {len(report.errors) - 20} more errors", file=sys.stderr)
    return 1 if report.errors and report.refs_inserted == 0 and report.refs_updated == 0 else 0


def cmd_index(args: argparse.Namespace) -> int:
    with db.session(args.db) as conn:
        rows = ingest.hedonic_rows(conn, reference=args.reference, brand=args.brand)
        if len(rows) < 12:
            print(f"only {len(rows)} observations -- not enough to index", file=sys.stderr)
            return 1
        points, fit = hedonic.time_dummy_index(rows, hedonic.standard_features())
        cagr = hedonic.annualised_return(points)

        print(f"{args.reference or args.brand or 'all'}: "
              f"{fit.n_obs} fresh listings, R^2 {fit.r_squared:.3f}")
        print(f"{'period':<9}{'index':>9}{'n':>7}   95% CI")
        for point in points:
            ci = (f"  {point.ci_low:.1f}-{point.ci_high:.1f}"
                  if point.ci_low is not None else "")
            print(f"{point.period:<9}{point.value:>9.1f}{point.n_obs:>7}{ci}")
        if cagr is not None:
            costs = metrics.CostModel()
            level = points[-1].value
            print(f"\ngross CAGR {cagr:+.1%}")
            print(f"cost hurdle to break even: {costs.hurdle_rate(level * 100, 1.0):.1%} "
                  "(scaled to index level; use --json for the real figure)")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    with db.session(args.db) as conn:
        rows = metrics.screen(conn, window_days=args.window, min_listings=args.min_listings)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    header = f"{'reference':<22}{'score':>7}{'live':>6}{'med ask':>11}{'DOM':>6}{'cut%':>7}{'disp':>7}"
    print(header)
    print("-" * len(header))
    for row in rows[: args.limit]:
        ask = row["median_ask_eur"]
        ask_text = f"EUR{ask:,.0f}" if ask else "-"
        cut = row["cut_rate"]
        cut_text = f"{cut * 100:.0f}%" if cut is not None else "-"
        print(
            f"{row['reference']:<22}"
            f"{_fmt(row['score'], 3):>7}"
            f"{row['n_live']:>6}"
            f"{ask_text:>11}"
            f"{_fmt(row['median_days_on_market'], 0):>6}"
            f"{cut_text:>7}"
            f"{_fmt(row['dispersion'], 2):>7}"
        )
    print("\nThis is a supply-side screen, not a return forecast.")
    return 0


def _fmt(value, places: int) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server

    if not os.path.exists(args.db):
        print(f"no database at {args.db} -- run 'python -m watchlab demo' first", file=sys.stderr)
        return 1
    server.serve(host=args.host, port=args.port, db_path=args.db)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watchlab", description="Quality-adjusted wristwatch price indices."
    )
    parser.add_argument("--db", default=db.DEFAULT_DB_PATH, help="SQLite path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="populate a simulated market")
    demo.add_argument("--days", type=int, default=730)
    demo.add_argument("--seed", type=int, default=20260812)
    demo.set_defaults(func=cmd_demo)

    calibrate = subparsers.add_parser(
        "calibrate", help="inspect what the parsers extract from saved HTML"
    )
    calibrate.add_argument("files", nargs="+")
    calibrate.add_argument("--sample", type=int, default=5)
    calibrate.set_defaults(func=cmd_calibrate)

    ingest_cmd = subparsers.add_parser("ingest", help="load saved HTML pages into the database")
    ingest_cmd.add_argument("files", nargs="+")
    ingest_cmd.add_argument("--date", help="observation date (default: today)")
    ingest_cmd.add_argument(
        "--complete-crawl", action="store_true",
        help="mark unseen live listings as delisted; only safe on a full crawl",
    )
    ingest_cmd.set_defaults(func=cmd_ingest)

    catalogue_cmd = subparsers.add_parser(
        "catalogue", help="manage the reference catalogue (refs / ref_aliases)"
    )
    catalogue_sub = catalogue_cmd.add_subparsers(dest="catalogue_command", required=True)
    catalogue_import = catalogue_sub.add_parser(
        "import", help="load a JSON or CSV file of reference records"
    )
    catalogue_import.add_argument("file")
    catalogue_import.set_defaults(func=cmd_catalogue_import)

    index_cmd = subparsers.add_parser("index", help="print a hedonic index")
    index_cmd.add_argument("--reference")
    index_cmd.add_argument("--brand")
    index_cmd.set_defaults(func=cmd_index)

    screen_cmd = subparsers.add_parser("screen", help="rank references by supply-side metrics")
    screen_cmd.add_argument("--window", type=int, default=90)
    screen_cmd.add_argument("--min-listings", type=int, default=5)
    screen_cmd.add_argument("--limit", type=int, default=30)
    screen_cmd.add_argument("--json", action="store_true")
    screen_cmd.set_defaults(func=cmd_screen)

    serve_cmd = subparsers.add_parser("serve", help="run the local dashboard")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
