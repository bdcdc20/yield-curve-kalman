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
- The next task is **roadmap step 2: `notebooks/01_two_step_usd.ipynb`**
  (README §8): USD factor plots, fit RMSE, first comparison with the random walk.

## First-time setup check

Run these from the repo root and fix anything that fails before starting work:

```bash
python3 -m pip install -e ".[dev]"
test -f .env || echo "MISSING .env: ask Bruno for his FRED key, then: echo 'FRED_API_KEY=<key>' > .env"
python3 -m pytest -q                  # expect 27 passed
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
