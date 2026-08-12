"""Command line entry point: ``python -m watchlab <command>``."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

from . import auctions, catalogue, db, genetic, hedonic, ingest, metrics, report
from .sources import auction_houses, chrono24, synthetic
from .sources import ebay as _ebay
from .sources import thewatchapi as _thewatchapi


def cmd_demo(args: argparse.Namespace) -> int:
    """Build a database from the simulator so the dashboard has something to show."""
    end = _dt.date.today()
    start = end - _dt.timedelta(days=args.days)
    universe = synthetic.BUDGET_UNIVERSE if args.universe == "budget" else None
    with db.session(args.db) as conn:
        conn.execute("DELETE FROM listing_snapshots")
        conn.execute("DELETE FROM listings")
        synthetic.seed_fx(conn, start.isoformat())
        synthetic.generate(conn, start, end, universe=universe, seed=args.seed)
        n = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        s = conn.execute("SELECT COUNT(*) FROM listing_snapshots").fetchone()[0]
    print(f"simulated {n} listings / {s} snapshots over {args.days} days into {args.db} "
          f"({args.universe} universe)")
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


def _thewatchapi_token(args: argparse.Namespace) -> str | None:
    token = args.token or os.environ.get("THEWATCHAPI_TOKEN")
    if not token:
        print(
            "no API token: pass --token or set THEWATCHAPI_TOKEN. "
            "Never commit a token to the repo.", file=sys.stderr,
        )
        return None
    return token


def _thewatchapi_client(args: argparse.Namespace) -> _thewatchapi.Client | None:
    token = _thewatchapi_token(args)
    if not token:
        return None
    cache_dir = args.cache_dir or os.path.join(os.path.dirname(args.db) or ".", ".thewatchapi_cache")
    return _thewatchapi.Client(token=token, cache_dir=cache_dir)


def cmd_thewatchapi_sync_brand(args: argparse.Namespace) -> int:
    """Cheap path: reference/list for a brand -> refs table (brand + reference only)."""
    client = _thewatchapi_client(args)
    if client is None:
        return 1
    try:
        records = _thewatchapi.sync_brand_references(client, args.brand)
    except _thewatchapi.ThewatchapiError as exc:
        print(f"thewatchapi error: {exc}", file=sys.stderr)
        return 1
    with db.session(args.db) as conn:
        report = catalogue.upsert_catalogue(conn, records)
    print(f"{args.brand}: {report.summary()}")
    return 0


def cmd_thewatchapi_enrich(args: argparse.Namespace) -> int:
    """HIGH USAGE: one model/search call per --reference. Never loops a whole brand."""
    client = _thewatchapi_client(args)
    if client is None:
        return 1
    records = []
    for reference in args.reference:
        try:
            record = _thewatchapi.enrich_reference(client, args.brand, reference)
        except _thewatchapi.ThewatchapiError as exc:
            print(f"  ! {args.brand} {reference}: {exc}", file=sys.stderr)
            continue
        if record is None:
            print(f"  ! {args.brand} {reference}: no match", file=sys.stderr)
            continue
        records.append(_thewatchapi.to_catalogue_record(record))
    if not records:
        print("nothing enriched", file=sys.stderr)
        return 1
    with db.session(args.db) as conn:
        report = catalogue.upsert_catalogue(conn, records)
    print(report.summary())
    return 0


def cmd_thewatchapi_price_history(args: argparse.Namespace) -> int:
    """Standard-plan-and-above endpoint. Stores to provider_price_series, not index_points."""
    client = _thewatchapi_client(args)
    if client is None:
        return 1

    scope_type, scope_value, fetch = next(
        (kind, value, fn) for kind, value, fn in (
            ("brand", args.brand, _thewatchapi.brand_price_history),
            ("model", args.model, _thewatchapi.model_price_history),
            ("reference", args.reference, _thewatchapi.reference_price_history),
        ) if value
    )
    try:
        payload = fetch(client, scope_value, date_from=args.date_from, date_to=args.date_to)
    except _thewatchapi.ThewatchapiError as exc:
        print(f"thewatchapi error: {exc}", file=sys.stderr)
        return 1

    with db.session(args.db) as conn:
        n = ingest.store_provider_price_series(conn, "thewatchapi", scope_type, scope_value, payload)
    print(f"{scope_type}={scope_value}: {n} price points stored "
          "(indicative asking prices -- see provider_price_series docstring)")
    return 0


def cmd_auctions_import(args: argparse.Namespace) -> int:
    """Load auction lots (JSON or CSV) into auction_results. The path to prefer."""
    try:
        records = auctions.load_file(args.file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"could not read {args.file}: {exc}", file=sys.stderr)
        return 1
    with db.session(args.db) as conn:
        catalogue_table = ingest.load_catalogue(conn)
        report = auctions.upsert_auction_results(conn, records, catalogue=catalogue_table)
    print(report.summary())
    for error in report.errors[:20]:
        print(f"  ! {error}", file=sys.stderr)
    return 1 if report.errors and report.inserted == 0 and report.updated == 0 else 0


def cmd_auctions_calibrate(args: argparse.Namespace) -> int:
    """Show what JSON-LD extraction finds on saved auction results pages.

    Weaker evidence than the same command for Chrono24: schema.org's Offer
    type models something currently for sale, a poor semantic fit for a lot
    that has already closed. Expect this to find nothing on many real pages
    -- that's the honest failure mode, not a bug, and it means 'auctions
    import' with hand-compiled data is very likely the path you actually want.
    """
    total = 0
    for path, html in chrono24.iter_saved_pages(args.files):
        lots = auction_houses.parse_jsonld(html, house=args.house)
        print(f"\n{path}  ({len(html):,} bytes): {len(lots)} lots via JSON-LD")
        for lot in lots[: args.sample]:
            print(f"    - lot {lot.lot_number}  {(lot.raw_title or '')[:64]!r}  {lot.price_text}")
        if not lots:
            print("    nothing matched -- likely no schema.org markup on this page. "
                  "Consider 'auctions import' with hand-compiled data instead.")
        total += len(lots)
    print(f"\ntotal parsed: {total}")
    return 0


def cmd_auctions_ingest(args: argparse.Namespace) -> int:
    """Parse saved auction pages via JSON-LD and load them into auction_results."""
    lots = []
    for _, html in chrono24.iter_saved_pages(args.files):
        lots.extend(auction_houses.parse_jsonld(html, house=args.house))
    if not lots:
        print("no lots parsed -- run 'auctions calibrate' first, or use 'auctions import' "
              "with hand-compiled data instead", file=sys.stderr)
        return 1
    records = [auctions.from_raw_lot(lot) for lot in lots]
    with db.session(args.db) as conn:
        catalogue_table = ingest.load_catalogue(conn)
        report = auctions.upsert_auction_results(conn, records, catalogue=catalogue_table)
    print(report.summary())
    return 0


def cmd_auctions_index(args: argparse.Namespace) -> int:
    """Print the transaction-based hedonic index for a reference or brand."""
    with db.session(args.db) as conn:
        rows = auctions.auction_hedonic_rows(
            conn, reference=args.reference, brand=args.brand, period_months=args.period_months
        )
        if len(rows) < 12:
            print(f"only {len(rows)} transactions -- not enough to index", file=sys.stderr)
            return 1
        points, fit = hedonic.time_dummy_index(
            rows, hedonic.auction_features(), ridge=auctions.DEFAULT_AUCTION_RIDGE
        )
        cagr = hedonic.annualised_return(points)

        print(f"{args.reference or args.brand or 'all'} (transactions): "
              f"{fit.n_obs} sales, R^2 {fit.r_squared:.3f}, {args.period_months}-month periods")
        print(f"{'period':<9}{'index':>9}{'n':>7}   95% CI")
        for point in points:
            ci = (f"  {point.ci_low:.1f}-{point.ci_high:.1f}"
                  if point.ci_low is not None else "")
            print(f"{point.period:<9}{point.value:>9.1f}{point.n_obs:>7}{ci}")
        if cagr is not None:
            print(f"\ngross CAGR {cagr:+.1%} (realised prices -- no ask/transaction gap to worry "
                  "about here, but still gross of round-trip costs)")
    return 0


def _ebay_client(args: argparse.Namespace) -> _ebay.Client | None:
    client_id = args.client_id or os.environ.get("EBAY_CLIENT_ID")
    client_secret = args.client_secret or os.environ.get("EBAY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "no eBay credentials: pass --client-id/--client-secret or set "
            "EBAY_CLIENT_ID/EBAY_CLIENT_SECRET. Never commit them to the repo.", file=sys.stderr,
        )
        return None
    cache_dir = args.cache_dir or os.path.join(os.path.dirname(args.db) or ".", ".ebay_cache")
    return _ebay.Client(
        client_id=client_id, client_secret=client_secret,
        marketplace_id=args.marketplace, cache_dir=cache_dir,
    )


def cmd_ebay_ingest(args: argparse.Namespace) -> int:
    """Search eBay's Browse API for a watchlist and ingest results directly.

    No calibrate step: this is a real REST API returning structured JSON, not
    scraped HTML with guessed selectors, so there is nothing to calibrate.
    """
    client = _ebay_client(args)
    if client is None:
        return 1

    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            queries = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        queries = args.query

    try:
        raw = _ebay.search_watchlist(client, queries, limit=args.limit, filter=args.filter)
    except _ebay.EbayError as exc:
        print(f"eBay error: {exc}", file=sys.stderr)
        return 1

    if not raw:
        print("no listings found for this watchlist", file=sys.stderr)
        return 1

    observed = args.date or _dt.date.today().isoformat()
    with db.session(args.db) as conn:
        ingest_report = ingest.upsert_listings(conn, raw, observed_at=observed, source="ebay")
    print(ingest_report.summary())
    return 0


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


def cmd_report(args: argparse.Namespace) -> int:
    """Rank a watchlist of references by net-of-cost hedonic return."""
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            references = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        references = args.reference

    # metrics.CostModel's default shipping_insurance_eur=250 is calibrated for
    # watches worth thousands: against a sub-1000 watch, EUR500 round-trip
    # shipping/insurance alone can exceed the item's value, clipping every net
    # CAGR to the -100% floor and making the ranking meaningless. --shipping-
    # insurance-eur lets a cheap-watch report use a realistic figure instead.
    costs = metrics.CostModel(shipping_insurance_eur=args.shipping_insurance_eur)
    with db.session(args.db) as conn:
        ranked = report.rank_references(
            conn, references, max_price_eur=args.max_price_eur, min_obs=args.min_obs, costs=costs,
        )
    top, bottom = report.top_bottom(ranked, n=args.top)
    skipped = [r for r in ranked if r.net_cagr is None]

    ceiling_text = f" (ceiling EUR {args.max_price_eur:,.0f})" if args.max_price_eur else ""
    print(f"{len(ranked) - len(skipped)}/{len(ranked)} references indexed{ceiling_text}")

    def _table(title: str, rows: list) -> None:
        print(f"\n{title}")
        header = f"{'reference':<24}{'brand':<14}{'net CAGR':>10}{'gross':>9}{'n':>6}   median ask"
        print(header)
        print("-" * len(header))
        for r in rows:
            ask = f"EUR {r.median_ask_eur:,.0f}" if r.median_ask_eur else "-"
            print(
                f"{r.reference:<24}{(r.brand or '-'):<14}"
                f"{r.net_cagr:>+10.1%}{r.gross_cagr:>+9.1%}{r.n_obs:>6}   {ask}"
            )

    if top:
        _table(f"TOP {len(top)} (best net-of-cost return)", top)
    if bottom:
        _table(f"BOTTOM {len(bottom)} (worst net-of-cost return)", bottom)
    if not top and not bottom:
        print("\nnothing indexed -- every reference was skipped, see below")

    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for r in skipped:
            print(f"  {r.reference:<24}{r.skip_reason}")

    print(
        "\nNet-of-cost, quality-adjusted return over each reference's own observed window -- "
        "not a forecast. A small watchlist makes this noisy; treat direction, not precision, "
        "as the signal."
    )
    return 0


def cmd_genetic_run(args: argparse.Namespace) -> int:
    """Evolve a scoring-weight vector, walk-forward validated, vs buy-and-hold.

    See watchlab/genetic.py's module docstring before trusting the output --
    this is a small-sample estimate over a watchlist-sized universe, and
    buy-and-hold is expected to win more often than not.
    """
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            references = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        references = args.reference

    # See cmd_report's identical comment: metrics.CostModel's 250EUR-per-leg
    # default assumes a watch worth thousands. Left at that default, every
    # sub-1000EUR position nets to the same -100% floor and the GA has
    # nothing to differentiate on.
    costs = metrics.CostModel(shipping_insurance_eur=args.shipping_insurance_eur)
    with db.session(args.db) as conn:
        result = genetic.run_ga(
            conn, references, population_size=args.population, generations=args.generations,
            top_k=args.top_k, checkpoint_every_months=args.checkpoint_months,
            horizon_months=args.horizon_months, complexity_penalty=args.complexity_penalty,
            seed=args.seed, costs=costs,
        )

    print(f"{result.n_train_checkpoints} train checkpoints, {result.n_test_checkpoints} test "
          f"checkpoints, top-{result.top_k} of {len(references)} references, "
          f"{args.horizon_months}-month horizon")
    if result.n_train_checkpoints == 0:
        print("not enough history to build even one checkpoint -- see build_checkpoints' "
              "12-observation-per-reference floor", file=sys.stderr)
        return 1

    print("\nevolved weights (feature -> signed weight; sign is what the GA learned, not assumed):")
    for name, weight in result.best_weights.items():
        print(f"  {name:<16}{weight:+.3f}")

    print(f"\ntrain net CAGR-equivalent (in-sample, for reference only): "
          f"{_fmt_pct(result.train_net_return)}")
    print(f"test net return, GA-selected top-{result.top_k}  (out-of-sample): "
          f"{_fmt_pct(result.test_net_return)}")
    print(f"test net return, equal-weight buy-and-hold (out-of-sample): "
          f"{_fmt_pct(result.test_buy_and_hold_net_return)}")
    if result.test_net_return is not None and result.test_buy_and_hold_net_return is not None:
        winner = "GA screen" if result.test_net_return > result.test_buy_and_hold_net_return else "buy-and-hold"
        print(f"\n{winner} wins on this run -- on {result.n_test_checkpoints} test checkpoint(s), "
              "which is not enough to generalise. This is a screen to point judgement at, "
              "never a return forecast.")
    return 0


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1%}"


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
    demo.add_argument(
        "--universe", choices=["default", "budget"], default="default",
        help="'default' = luxury validation set, 'budget' = everyday sub-EUR1000 watches",
    )
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

    def _token_arg(p):
        p.add_argument("--token", help="thewatchapi token (default: $THEWATCHAPI_TOKEN)")
        p.add_argument("--cache-dir", help="response cache dir (default: <db dir>/.thewatchapi_cache)")

    twa = subparsers.add_parser(
        "thewatchapi", help="pull reference/price data from thewatchapi.com"
    )
    twa_sub = twa.add_subparsers(dest="thewatchapi_command", required=True)

    twa_sync = twa_sub.add_parser(
        "sync-brand", help="cheap: reference/list -> refs table (brand + reference only)"
    )
    twa_sync.add_argument("brand")
    _token_arg(twa_sync)
    twa_sync.set_defaults(func=cmd_thewatchapi_sync_brand)

    twa_enrich = twa_sub.add_parser(
        "enrich", help="HIGH USAGE: model/search per reference for case/movement/years"
    )
    twa_enrich.add_argument("--brand", required=True)
    twa_enrich.add_argument("--reference", nargs="+", required=True)
    _token_arg(twa_enrich)
    twa_enrich.set_defaults(func=cmd_thewatchapi_enrich)

    twa_price = twa_sub.add_parser(
        "price-history", help="Standard plan+: indicative asking-price series"
    )
    scope = twa_price.add_mutually_exclusive_group(required=True)
    scope.add_argument("--brand")
    scope.add_argument("--model")
    scope.add_argument("--reference")
    twa_price.add_argument("--date-from")
    twa_price.add_argument("--date-to")
    _token_arg(twa_price)
    twa_price.set_defaults(func=cmd_thewatchapi_price_history)

    auctions_cmd = subparsers.add_parser(
        "auctions", help="load and index auction transactions (real sale prices, not asks)"
    )
    auctions_sub = auctions_cmd.add_subparsers(dest="auctions_command", required=True)

    auctions_import = auctions_sub.add_parser(
        "import", help="load a JSON or CSV file of lots -- the path to prefer"
    )
    auctions_import.add_argument("file")
    auctions_import.set_defaults(func=cmd_auctions_import)

    auctions_calibrate = auctions_sub.add_parser(
        "calibrate", help="inspect what JSON-LD extraction finds on saved auction pages"
    )
    auctions_calibrate.add_argument("house")
    auctions_calibrate.add_argument("files", nargs="+")
    auctions_calibrate.add_argument("--sample", type=int, default=5)
    auctions_calibrate.set_defaults(func=cmd_auctions_calibrate)

    auctions_ingest = auctions_sub.add_parser(
        "ingest", help="parse saved auction pages via JSON-LD and load them"
    )
    auctions_ingest.add_argument("house")
    auctions_ingest.add_argument("files", nargs="+")
    auctions_ingest.set_defaults(func=cmd_auctions_ingest)

    auctions_index = auctions_sub.add_parser(
        "index", help="print the transaction-based hedonic index"
    )
    auctions_index.add_argument("--reference")
    auctions_index.add_argument("--brand")
    auctions_index.add_argument(
        "--period-months", type=int, default=3,
        help="index period width in months; 3=quarterly (default, matches typical auction "
             "cadence), 1=monthly for a reference that trades often enough to support it",
    )
    auctions_index.set_defaults(func=cmd_auctions_index)

    def _ebay_credential_args(p):
        p.add_argument("--client-id", help="eBay application client id (default: $EBAY_CLIENT_ID)")
        p.add_argument("--client-secret", help="eBay application client secret (default: $EBAY_CLIENT_SECRET)")
        p.add_argument("--cache-dir", help="response cache dir (default: <db dir>/.ebay_cache)")
        p.add_argument(
            "--marketplace", default="EBAY_GB",
            help="eBay marketplace id, sets both currency and site searched (default: EBAY_GB)",
        )

    ebay_cmd = subparsers.add_parser("ebay", help="search eBay's Browse API and ingest results")
    ebay_sub = ebay_cmd.add_subparsers(dest="ebay_command", required=True)

    ebay_ingest = ebay_sub.add_parser(
        "ingest", help="search a watchlist via item_summary/search and ingest the results"
    )
    ebay_ref_group = ebay_ingest.add_mutually_exclusive_group(required=True)
    ebay_ref_group.add_argument("--query", nargs="+", help="search query strings, e.g. 'Seiko SRPD55K1'")
    ebay_ref_group.add_argument("--file", help="text file, one search query per line ('#' comments ok)")
    ebay_ingest.add_argument("--date", help="observation date (default: today)")
    ebay_ingest.add_argument("--limit", type=int, default=50, help="results per query (max 200)")
    ebay_ingest.add_argument(
        "--filter", help="Browse API filter string, e.g. 'buyingOptions:{FIXED_PRICE}'"
    )
    _ebay_credential_args(ebay_ingest)
    ebay_ingest.set_defaults(func=cmd_ebay_ingest)

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

    genetic_cmd = subparsers.add_parser(
        "genetic", help="GA-tuned reference screen, walk-forward validated vs buy-and-hold"
    )
    genetic_sub = genetic_cmd.add_subparsers(dest="genetic_command", required=True)
    genetic_run = genetic_sub.add_parser(
        "run", help="evolve a scoring weight vector and report it against buy-and-hold"
    )
    genetic_ref_group = genetic_run.add_mutually_exclusive_group(required=True)
    genetic_ref_group.add_argument("--reference", nargs="+", help="reference numbers to include")
    genetic_ref_group.add_argument("--file", help="text file, one reference per line ('#' comments ok)")
    genetic_run.add_argument("--top-k", type=int, default=3, help="how many references the GA screen selects")
    genetic_run.add_argument("--population", type=int, default=40)
    genetic_run.add_argument("--generations", type=int, default=30)
    genetic_run.add_argument("--checkpoint-months", type=int, default=3)
    genetic_run.add_argument("--horizon-months", type=int, default=6)
    genetic_run.add_argument(
        "--complexity-penalty", type=float, default=0.02,
        help="L1 penalty on the weight vector; higher prefers fewer/smaller weights",
    )
    genetic_run.add_argument("--seed", type=int, help="fixed seed for a reproducible run")
    genetic_run.add_argument(
        "--shipping-insurance-eur", type=float, default=30.0,
        help="round-trip shipping+insurance per leg; default (30) suits sub-1000EUR watches",
    )
    genetic_run.set_defaults(func=cmd_genetic_run)

    report_cmd = subparsers.add_parser(
        "report", help="rank a watchlist of references by net-of-cost hedonic return"
    )
    report_ref_group = report_cmd.add_mutually_exclusive_group(required=True)
    report_ref_group.add_argument("--reference", nargs="+", help="reference numbers to include")
    report_ref_group.add_argument("--file", help="text file, one reference per line ('#' comments ok)")
    report_cmd.add_argument(
        "--max-price-eur", type=float, help="drop references whose median ask exceeds this"
    )
    report_cmd.add_argument("--min-obs", type=int, default=12)
    report_cmd.add_argument("--top", type=int, default=10)
    report_cmd.add_argument(
        "--shipping-insurance-eur", type=float, default=30.0,
        help="round-trip shipping+insurance per leg; default (30) suits sub-1000EUR watches, "
             "raise it for a luxury watchlist (metrics.CostModel's own default is 250)",
    )
    report_cmd.set_defaults(func=cmd_report)

    serve_cmd = subparsers.add_parser("serve", help="run the local dashboard")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
