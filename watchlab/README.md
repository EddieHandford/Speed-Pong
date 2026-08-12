# watchlab

A local dashboard for wristwatch reference prices: quality-adjusted price
indices and supply-side metrics, computed from marketplace listings.

Pure standard library. No `pip install`, no `numpy`, no `pandas`. Python 3.11+.

```bash
python3 -m watchlab demo        # build a simulated market to look at
python3 -m watchlab serve       # dashboard on http://127.0.0.1:8765
python3 -m unittest discover -s tests -t .
```

## Read this before you trust a number

**These are asking prices, not transactions.** Chrono24 and every other
listings site publish what sellers *want*. Most of those asks never clear. A
listing disappearing is not a sale — sellers delist, relist, and let listings
expire constantly. Nothing here is a record of what anyone actually paid.

**Costs eat most of what looks like a return.** A round trip through a
marketplace costs roughly 18–20% before tax: escrow and authentication,
seller commission, the gap between ask and clearing bid, shipping and
insurance. The dashboard prints a `cost hurdle` next to every index precisely
so a 10%-a-year "winner" is visibly a loss. It usually is.

**A screen is not a forecast.** The composite score ranks references by how
tight their supply side looks. That is a place to point your judgement, not a
prediction. Nothing in this repository forecasts a price, and any component
that claimed to would be lying about what listings data can support.

## What it actually does

### Normalisation (`normalize.py`)

Turns `"ROLEX SUBMARINER DATE 126610 LN 2023 B+P"` into a canonical record.
This is the largest and least glamorous part of the project, and the one that
silently ruins everything downstream when it is wrong — a title resolved to
the wrong reference does not raise, it just pollutes a price series with a
different watch.

Resolution runs catalogue-first, then brand-specific patterns, then a generic
fallback, and it strips the reference out of the title *before* looking for a
production year so reference digits can never be misread as a year. Unknown
box/papers status stays `None` rather than becoming `0`; silence about papers
is not evidence of absence, and conflating the two biases the estimated
papers premium.

### The hedonic index (`hedonic.py`)

The core problem with a median ask: it moves when the *mix* of listings
changes, not only when the market does. A month with more unworn full-set
examples reads as appreciation that never happened.

The fix is a time-dummy hedonic regression — condition, papers, box, age,
country and seller type enter as controls, period dummies carry the index, and
the index is the exponentiated period coefficient. That gives the price of a
*constant-quality* watch over time.

`tests/test_hedonic.py` asserts this works, against a simulated market whose
true price path is known. Measured across the ten-reference default universe,
RMSE against the true index:

| Reference | True CAGR | Hedonic | Naive median | Hedonic RMSE | Naive RMSE |
|---|---|---|---|---|---|
| 126610LN | +13.4% | +11.4% | +15.6% | 3.31 | 5.94 |
| 15500ST | −13.1% | −9.4% | −2.2% | 6.40 | 15.79 |
| 5711/1A-010 | −7.6% | −7.3% | −4.8% | 2.74 | 9.55 |
| SBGA211 | +4.9% | +8.0% | +15.2% | 5.68 | 18.05 |

The hedonic index wins on 9 of 10 references, and note the *direction* of the
naive error: it overstates almost everywhere. That is composition drift, and
it is the single most common way an amateur watch-investing analysis fools
itself.

### Fresh asks only

By default the index uses each listing's **first** observation. In the
reference simulation the median live listing is ~49 days old, so an index
built from all live listings measures the market of seven weeks ago, smeared.
Worse, the staleness is not random — overpriced listings are exactly the ones
that fail to sell and keep accumulating observations, so stale asks are
systematically high asks. A first observation is a fresh quote.

### Supply-side metrics (`metrics.py`)

None of these need a sale price, which is what makes them usable:

- `days_on_market` — how long examples sit before leaving the market
- `cut_rate` / `cut_depth` — how many sellers are reducing, and by how much
- `dispersion` — IQR over median; how much sellers disagree on value
- `churn_ratio` — new listings against delistings
- `n_live` — current inventory

A rising ask index *alongside* rising supply, rising days-on-market and a
rising cut rate is a market topping out, not a market appreciating. Making
that combination visible is the point of the table.

### The transaction-based index (`auctions.py`)

Everything above is asks. `auction_results` is the one table with transaction
truth in it — a "price realised" is what a real buyer actually paid, on a
known date. `auction_hedonic_rows()` feeds this straight into the same
`hedonic.time_dummy_index()` machinery used for listings, with
`hedonic.auction_features()` swapping in `house` as a quality control (which
room a lot sells through has a real, documented effect on hammer price)
instead of seller country/type.

```bash
python3 -m watchlab auctions import lots.json      # the path to prefer
python3 -m watchlab auctions index --reference 126610LN
```

The dashboard shows this alongside the ask-based index for the same
reference, in its own panel with its own chart, not merged into one --
different period width, different bias profile, different volume, and
merging them would visually imply a comparability that isn't there. A short
compare line under the panel states both CAGRs side by side with that same
caveat, rather than picking a winner.

Auction volume is much lower than listing volume — a reference might see a
few dozen sales a year across every house combined, not thousands of
listings a month — which changes two things about the estimation, both
handled by default rather than left as a footgun:

- **Periods are quarterly, not monthly** (`period_months=3` by default in
  `auction_hedonic_rows`), because monthly bins on auction-scale volume run
  out of observations per period fast enough to make the regression
  under-determined.
- **Ridge is higher** (`DEFAULT_AUCTION_RIDGE = 0.02` vs the near-zero
  default): fewer observations per parameter means individual coefficients
  are noisier even though they're unbiased, and a moderate ridge trades a
  little bias for a real reduction in that variance. Checked empirically
  across the synthetic universe and several seeds, not picked to fit one case.

**What this table does *not* let you claim:** that a transaction-based index
always beats an ask-based one. It's less *biased* (no seller markup, no
listing staleness), but it's also noisier from lower volume, and in a
small-sample regime lower bias doesn't guarantee lower error against the
truth — checked in `tests/test_auctions.py` across several synthetic seeds,
auctions beat listings on RMSE only about half the time. The reliable,
reproducible result is the one auctions and listings both make separately:
hedonic beats naive-median-of-the-same-data, because the composition-drift
problem is real in auction lot mix too, not just in Chrono24 listings.

## Getting real data in

For anything sub-£1000, prefer eBay's Browse API (below) — it's a free,
structured REST API with no scraping, no ToS conflict, and no selectors to
keep fixing. Chrono24's saved-page workflow is what to reach for when a
watch's real market lives there instead (higher-end pieces, mostly).

### eBay Browse API (`sources/ebay.py`)

A real, documented REST API, free to register for, with a 5,000-call/day
default quota — no scraping, no ToS conflict, no HTML selectors to keep
fixing, and no manual page-saving. Built against the documented OAuth2
client-credentials grant and the `item_summary/search` resource. Set
`EBAY_CLIENT_ID` / `EBAY_CLIENT_SECRET` (a free application keyset from
https://developer.ebay.com — production keys, not sandbox, since sandbox has
no real listings; never commit them to the repo) and:

```bash
python3 -m watchlab ebay ingest --query "Seiko SRPD55K1" "Casio GA-2100-1A1"
# or, for a longer list:
python3 -m watchlab ebay ingest --file watchlists/budget_under_1000_queries.txt --date 2026-08-19
```

Goes straight into `listings`/`listing_snapshots` via the same
`ingest.upsert_listings` path `chrono24.py` uses (`source="ebay"`), so
`index`, `screen` and `report` all work on it unmodified. Condition is
mapped from eBay's documented condition strings on a best-effort basis; an
unrecognised string is left `None` rather than guessed, same rule
`normalize.py` already applies to box/papers.

### Chrono24 listings (`sources/chrono24.py`)

`parse_jsonld` has been fixed against a real saved search-results page
(2026-08-12): Chrono24's `@graph` carries a single `AggregateOffer` node with
a bare `offers` array, not the Product/IndividualProduct wrapper originally
guessed. The DOM fallback (`SELECTORS`) remains unverified for whatever page
shape doesn't carry that JSON-LD.

Chrono24's terms prohibit automated collection and the site runs bot
protection, so a plain HTTP client will mostly collect challenge pages. The
supported workflow avoids the issue entirely:

```bash
# save pages from a normal browser session, then:
python3 -m watchlab calibrate saved/*.html   # see what each strategy extracts
python3 -m watchlab ingest saved/*.html --date 2026-08-12
```

`PoliteFetcher` exists for the case where you have permission: it checks
robots.txt (and treats an unreadable robots.txt as *disallowed*, not as
consent), rate-limits, and caches every page to disk so nothing is ever
requested twice.

**`--complete-crawl` is dangerous.** It marks every unseen live listing as
delisted. On a partial crawl that silently corrupts every days-on-market
number in the database. Only pass it when the crawl genuinely covered
everything you track.

### Auction results (`sources/auction_houses.py`, `auctions.py`)

For anything involving *history* or *actual clearing prices*, auction results
beat listings outright: Phillips, Christie's, Sotheby's, Bonhams and
Antiquorum publish dated, itemised, genuinely-sold prices going back years,
publicly, and have every incentive to keep doing so — publishing results is
core to how they attract future consignments. Four independent houses means
no single point of failure the way a scraper aimed at one marketplace has.

Two ways in, same split as the reference catalogue:

```bash
# prefer this: a documented JSON/CSV shape, works with anything you can get
# as structured data -- a research export, a hand-compiled spreadsheet
python3 -m watchlab auctions import lots.json

# saved-page HTML parsing, unverified against live pages (no network access
# to any auction house from this sandbox) -- and weaker evidence here than
# it was for Chrono24: schema.org's Offer type models something currently
# for sale, a poor semantic fit for a lot that has already closed
python3 -m watchlab auctions calibrate "Phillips" saved/*.html
python3 -m watchlab auctions ingest "Phillips" saved/*.html
```

Fields explicitly supplied in an import always win over what
`normalize.parse_title` infers from the lot title — an auction house's own
condition report is more authoritative than a regex guess.

### thewatchapi.com (`sources/thewatchapi.py`)

An actual documented REST API with a free tier, unlike Chrono24. Built
against the real docs and their example responses, so unlike `chrono24.py`
nothing here is a guess. Set `THEWATCHAPI_TOKEN` (never commit a token to the
repo) and:

```bash
# cheap: reference/list, "All plans" -- one call per brand
python3 -m watchlab thewatchapi sync-brand Rolex

# HIGH USAGE: model/search, one call per reference -- opt in explicitly,
# never loop this across a brand's whole catalogue
python3 -m watchlab thewatchapi enrich --brand Rolex --reference 116520 116500LN

# Standard plan+: indicative ASKING price series (not transactions --
# same caveat as everything else here), stored separately from the
# hedonic index so the two are never confused
python3 -m watchlab thewatchapi price-history --reference 116520
```

`sync-brand` populates `refs` with brand + reference only, which is already
enough to lift `normalize.parse_title` from regex-guessing to catalogue
matching (confidence 0.4 → 0.7). `enrich` adds case size and production
years for a specific watchlist, at a real credit cost — do this selectively.
`price-history` writes into `provider_price_series`, deliberately **not**
`index_points`: thewatchapi's series is a pre-aggregated asking-price
average with no visibility into whether it controls for a changing listing
mix, so it is a cross-check against the hedonic index, never a substitute
for it.

## Layout

```
watchlab/
  db.py                schema, connection, FX conversion
  normalize.py         title -> canonical record
  catalogue.py         provider-agnostic reference-catalogue importer
  auctions.py          provider-agnostic auction-lot importer + transaction index rows
  hedonic.py           OLS/ridge, time-dummy index (no numpy)
  metrics.py           microstructure metrics, cost model
  ingest.py            raw listings -> normalised rows, lifecycle
  report.py            multi-reference ranking: net-of-cost return, top/bottom N
  server.py            stdlib HTTP server + JSON API
  cli.py               python -m watchlab ...
  web/dashboard.html   the dashboard
  sources/
    chrono24.py        listing scraper (JSON-LD verified 2026-08-12; DOM fallback unverified)
    auction_houses.py  auction lot HTML parser (UNVERIFIED -- no live access)
    thewatchapi.py      catalogue + price-history client (verified vs real docs)
    ebay.py             Browse API client (verified vs real docs; no scraping, free tier)
    synthetic.py         simulated market + auction sales, both with a known true path
tests/                 148 tests
```

## Not built yet

- The genetic algorithm. Deliberately last. A GA over a few thousand noisy
  observations is an extremely efficient way to find a rule that fits the
  past perfectly and predicts nothing. If it gets built, it needs walk-forward
  validation, a complexity penalty, transaction costs inside the fitness
  function, and an honest comparison against equal-weight buy-and-hold — which
  will probably win.
