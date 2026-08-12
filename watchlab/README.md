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

## Getting real data in

`sources/chrono24.py` is written but **unverified** — it was built without
live access to the site, so the CSS selectors in `SELECTORS` are guesses. It
tries schema.org JSON-LD first, which is a published standard and much less
likely to drift than class names.

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

### Better sources than Chrono24

For anything involving *history* or *actual clearing prices*, auction results
beat listings outright: Phillips, Christie's, Sotheby's, Bonhams and
Antiquorum publish dated, itemised, genuinely-sold prices going back years,
publicly, with far cleaner legal footing. The `auction_results` table is in
the schema and waiting for an ingester. That is the right next thing to build
— and the only way to backtest anything before you have collected two years
of listings yourself.

## Layout

```
watchlab/
  db.py                schema, connection, FX conversion
  normalize.py         title -> canonical record
  hedonic.py           OLS/ridge, time-dummy index (no numpy)
  metrics.py           microstructure metrics, cost model
  ingest.py            raw listings -> normalised rows, lifecycle
  server.py            stdlib HTTP server + JSON API
  cli.py               python -m watchlab ...
  web/dashboard.html   the dashboard
  sources/
    chrono24.py        listing scraper (UNVERIFIED selectors)
    synthetic.py       simulated market with a known true path
tests/                 51 tests
```

## Not built yet

- Auction-results ingester (the real historical dataset)
- The genetic algorithm. Deliberately last. A GA over a few thousand noisy
  observations is an extremely efficient way to find a rule that fits the
  past perfectly and predicts nothing. If it gets built, it needs walk-forward
  validation, a complexity penalty, transaction costs inside the fitness
  function, and an honest comparison against equal-weight buy-and-hold — which
  will probably win.
