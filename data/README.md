`raw/` holds downloads exactly as received (cached; `--refresh` re-downloads).
`processed/` holds the monthly panels built from them. Neither is committed:
run `python -m yck fetch` to rebuild both.
