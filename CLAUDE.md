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
- The next task is **roadmap step 5**: forward-rate panel in the data
  pipeline and forward benchmark (README §7.4), then Diebold–Mariano and
  Clark–West tests. Then trading evaluation (§7.5).
- Backtest outputs go in `data/results/` (git-ignored), fits are cached in
  `data/results/cache/`. Bump `FIT_CACHE_VERSION` in `eval/backtest.py`
  whenever the estimator changes. Parallel code must live in importable
  modules (macOS uses `spawn`), not in scripts piped through stdin.

## First-time setup check

Run these from the repo root and fix anything that fails before starting work:

```bash
python3 -m pip install -e ".[dev]"
test -f .env || echo "MISSING .env: ask Bruno for his FRED key, then: echo 'FRED_API_KEY=<key>' > .env"
python3 -m pytest -q                  # expect 55 passed
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
