# Instructions for Claude Code

This repository is Bruno del Castillo's yield-curve forecasting project: a
dynamic Nelson–Siegel model with macro factors, estimated with a Kalman filter,
for USD, EUR and GBP, using public data only. It is a public portfolio piece
for quant internship applications, so code quality and explanation matter.

**Read `README.md` in full before doing anything.** It is the complete
handover: setup, data pipeline, data quirks, the model specification, the
evaluation protocol, the roadmap and the decisions already made.

## Where the project stands

- The data pipeline is finished and verified (README §2).
- Roadmap step 1 (two-step Diebold–Li baseline, `src/yck/model/dns.py`) is done.
- Roadmap step 2 (`notebooks/01_two_step_usd.ipynb`, recursive backtest in
  `src/yck/eval/backtest.py`) is done; first results are in README §2.
- Roadmap step 3 (`src/yck/model/kalman.py`, tested against statsmodels) is done.
- Roadmap steps 4 and 4b (ML DNS with EM + exact score, 1 bp noise floor,
  multistart, parallel cached refits, `python -m yck backtest`) are done;
  results are in README §2 and `notebooks/02_kalman_usd.ipynb`.
- Roadmap step 5 is done except the optional SPF survey benchmark: forward
  benchmark (`curves_ext_{CCY}.parquet`, `eval.backtest.forward_forecasts`)
  and DM/CW tests (`eval/stats.py`). USD: nothing beats the random walk
  significantly at h = 6, 12; no yields-only model beats forwards.
- Roadmap steps 6 and 7 are done: yields + macro (real-time panels, DRA
  model, policy-rate-only variant, shrinkage, impulse responses) for USD, and
  EUR/GBP with the cross-currency comparison (notebooks 03, 04; README
  §7.4). Since 11 Nov 2026 μ is fixed at the real-time sample mean and
  estimates are required to be stationary (README §6.3). Result: no model
  beats forwards significantly in any currency; the policy rate helps only
  in USD; EUR loses to the random walk through mean reversion. Next: the
  accuracy improvements in README §8 step 7b (diagnosis and literature in
  §7.6; start with AR(1) measurement errors and a shifting endpoint for the
  level), then the trading evaluation (step 8), then the write-up (step 9).
- Backtest outputs go in `data/results/` (git-ignored), fits are cached in
  `data/results/cache/`. `python -m yck backtest --ccy USD EUR GBP --all`
  runs the whole standard grid (24 backtests, 488 refits) in ~1.5 min from
  scratch on 8 cores, ~10 s from cache, in one longest-first pool. Bump
  `FIT_CACHE_VERSION` in `eval/backtest.py` whenever the estimator changes,
  and compare new outputs with the old ones (they should be identical unless
  the estimator was meant to change). Parallel code must live in importable
  modules (macOS uses `spawn`), not in scripts piped through stdin. Profile
  before optimising (README §8, steps 4c and 4d).

## First-time setup check

Run these from the repo root and fix anything that fails before starting work:

```bash
python3 -m pip install -e ".[dev]"
test -f .env || echo "MISSING .env: ask Bruno for his FRED key, then: echo 'FRED_API_KEY=<key>' > .env"
python3 -m pytest -q                  # expect 133 passed
python3 -m yck fetch                  # builds data/processed/ (data is not in git)
cat data/processed/coverage.md
```

## Rules

- Never commit `.env` or any API key. The repository is public.
- Public data only (no Bloomberg or other licensed sources).
- Never interpolate or fill missing curve points; the Kalman filter handles
  missing observations (README §5.2, §6.3).
- Series IDs and URLs go in `config/data_sources.yaml`, not in code.
- New parsers and model components get offline tests; keep `pytest` green.
- Out-of-sample forecasts use filtered states and real-time / lagged macro
  (README §7.3); no look-ahead.
- Update README §2 (status table) and §8 (roadmap) as steps are completed.
- Commit in small, descriptive commits and push to `main`. The repository may
  live in OneDrive; if git reports lock-file errors, see README §11.
