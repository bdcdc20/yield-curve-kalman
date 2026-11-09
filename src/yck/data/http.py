"""Small HTTP helper: one session, retries, and a raw-file cache under data/raw."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

USER_AGENT = "yield-curve-kalman/0.1 (research project; +https://github.com/bdcdc20/yield-curve-kalman)"

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session


def get(url: str, params: dict | None = None, retries: int = 3, timeout: int = 60) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session().get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # noqa: PERF203
            last = e
            time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last


def fetch_to_cache(url: str, raw_dir: Path, name: str | None = None, params: dict | None = None,
                   refresh: bool = False, timeout: int = 60) -> Path:
    """Download url into raw_dir and return the path. Reuses the cached copy unless refresh=True."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if name is None:
        tag = hashlib.sha1((url + repr(sorted((params or {}).items()))).encode()).hexdigest()[:10]
        name = f"{Path(url.split('?')[0]).name or 'download'}.{tag}"
    path = raw_dir / name
    if path.exists() and not refresh:
        return path
    path.write_bytes(get(url, params=params, timeout=timeout).content)
    return path
