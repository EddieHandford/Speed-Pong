# eBay Browse API setup — pick up here

Status so far:
- [x] eBay developer account created
- [x] Marketplace Account Deletion exemption submitted ("does not persist eBay
      user/personal data" — accurate, since `sources/ebay.py` is app-only
      client-credentials auth and only stores public listing data)
- [ ] Production keyset's Client ID / Client Secret in hand
- [ ] Credentials set as env vars (never committed to the repo)
- [ ] First real ingest run
- [ ] Verify listings landed in the dashboard/report

## Next steps (on the Macbook)

1. Pull this branch:
   ```bash
   git fetch origin claude/ebay-developer-setup-eut8en
   git checkout claude/ebay-developer-setup-eut8en
   git pull origin claude/ebay-developer-setup-eut8en
   ```

2. Grab the **production** (not sandbox) Client ID and Client Secret from
   the keyset on https://developer.ebay.com and export them — don't put
   them in any file that gets committed:
   ```bash
   export EBAY_CLIENT_ID="..."
   export EBAY_CLIENT_SECRET="..."
   ```

3. Run the ingest against the existing watchlist:
   ```bash
   python3 -m watchlab ebay ingest --file watchlists/budget_under_1000_queries.txt
   ```
   or ad hoc:
   ```bash
   python3 -m watchlab ebay ingest --query "Seiko SRPD55K1" "Casio GA-2100-1A1"
   ```

4. Confirm it worked:
   ```bash
   python3 -m watchlab report --db <your.db>
   ```
   or open `watchlab/web/dashboard.html` and check eBay-sourced listings
   show up (`source="ebay"`).

## Notes

- Quota is 5,000 calls/day by default; one call per watchlist line, so a
  personal-size watchlist is nowhere near the limit.
- The OAuth app token is refetched per process (~2hr expiry) and never
  written to disk. Search *results* are cached to `.ebay_cache/` next to the
  db, since those count against quota — safe to delete that dir to force
  fresh pulls.
- If the exemption gets rejected instead of approved, the fallback is a
  small webhook (challenge-response verification + a 200-OK handler) — not
  built yet, ask Claude to add it to `watchlab/server.py` if it comes to
  that.
