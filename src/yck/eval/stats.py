"""Tests of equal predictive accuracy: Diebold-Mariano (non-nested) and Clark-West (nested).

Both reduce to a t-test on the mean of a loss-differential series d_t. Two
complications matter for yield forecasts:

- **Overlap.** h-step forecasts made monthly overlap for h > 1, so their errors
  are autocorrelated up to lag h-1 even if each forecast is optimal. The
  variance of mean(d) uses the autocovariances up to lag h-1 with **equal
  (rectangular) weights**, as in Diebold & Mariano (1995): this is exactly
  right for that MA(h-1) structure. Newey-West's Bartlett weights shrink
  those autocovariances and, in our simulations (300 observations, h = 12),
  made a nominal 5% test reject 8-11% of the time. The rectangular
  estimator with the HLN correction rejects 5.5-6.5%. If the rectangular
  estimate is not positive (possible, never seen in simulation), Bartlett
  is used instead.
- **Small samples.** With ~300 overlapping observations and h = 12, the
  plain DM statistic rejects too often. The Harvey-Leybourne-Newbold (1997)
  correction rescales it and uses Student-t critical values.

Conventions: positive statistics mean model 2 is *more* accurate than
model 1 (model 1 is the benchmark).

Pairwise tests against one benchmark invite data snooping once many variants
are tried (roadmap step 7b). The **model confidence set** (Hansen, Lunde &
Nason 2011) answers the joint question instead: which models cannot be
told apart from the best one, given every model compared (`mcs`, `mcs_table`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def newey_west_var_of_mean(d, lags: int, kernel: str = "bartlett") -> float:
    """Var(mean(d)) from autocovariances up to `lags`.

    kernel "bartlett": weights 1 - j/(lags+1) (Newey-West, always >= 0).
    kernel "rectangular": weights 1 (Diebold-Mariano; exact for MA(lags) but
    can be negative).
    """
    d = np.asarray(d, float)
    T = d.size
    u = d - d.mean()
    s = u @ u / T
    for j in range(1, lags + 1):
        w = 1.0 if kernel == "rectangular" else 1.0 - j / (lags + 1)
        s += 2.0 * w * (u[j:] @ u[:-j]) / T
    return s / T


def long_run_var_of_mean(d, h: int) -> float:
    """Variance of mean(d) for h-step forecast losses: rectangular, Bartlett if not positive."""
    lags = max(h - 1, 0)
    v = newey_west_var_of_mean(d, lags, "rectangular")
    return v if v > 0 else newey_west_var_of_mean(d, lags, "bartlett")


def diebold_mariano(e1, e2, h: int, hln: bool = True) -> dict:
    """DM test of equal mean squared error between forecast errors e1 (benchmark) and e2 (model).

    d_t = e1_t^2 - e2_t^2, so a positive statistic means the model has
    the lower MSE. Two-sided p-value. With `hln`, the statistic gets the
    Harvey-Leybourne-Newbold small-sample factor and is compared with
    t_{T-1}. Pairs where either error is missing are dropped.
    """
    e1, e2 = pd.Series(e1, dtype=float), pd.Series(e2, dtype=float)
    ok = e1.notna() & e2.notna()
    d = (e1[ok] ** 2 - e2[ok] ** 2).to_numpy()
    T = d.size
    var = long_run_var_of_mean(d, h) if T else np.nan
    if T < 10 or not var > 0:
        return {"stat": np.nan, "pvalue": np.nan, "mean_d": d.mean() if T else np.nan, "n": T}
    stat = d.mean() / np.sqrt(var)
    if hln:
        stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        p = 2 * sps.t.sf(abs(stat), df=T - 1)
    else:
        p = 2 * sps.norm.sf(abs(stat))
    return {"stat": float(stat), "pvalue": float(p), "mean_d": float(d.mean()), "n": T}


def clark_west(e_small, e_large, f_small, f_large, h: int) -> dict:
    """Clark-West (2007) test that a larger model nesting a smaller one forecasts better.

    adj_t = e_small^2 - [e_large^2 - (f_small - f_large)^2]. The subtracted
    term removes the noise the larger model's extra parameters add under the
    null (where they are truly zero), which would otherwise bias a plain MSE
    comparison against it. One-sided: reject equal accuracy in favour of the
    large model if the t-statistic exceeds 1.282 (10%) or 1.645 (5%).
    """
    s = pd.DataFrame({"es": e_small, "el": e_large, "fs": f_small, "fl": f_large}, dtype=float).dropna()
    adj = (s.es**2 - (s.el**2 - (s.fs - s.fl) ** 2)).to_numpy()
    T = adj.size
    var = long_run_var_of_mean(adj, h) if T else np.nan
    if T < 10 or not var > 0:
        return {"stat": np.nan, "pvalue": np.nan, "n": T}
    stat = adj.mean() / np.sqrt(var)
    return {"stat": float(stat), "pvalue": float(sps.norm.sf(stat)), "n": T}


def _stars(p: float) -> str:
    return "" if not np.isfinite(p) else "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def dm_table(forecasts: dict[str, pd.DataFrame], benchmark: str, actual: pd.DataFrame, h: int,
             columns=None) -> pd.DataFrame:
    """RMSE ratio vs `benchmark` and DM statistic for every model and maturity, on a common sample.

    Returns one row per maturity and, for each model, columns `ratio`, `dm`
    and `p`, plus a display column `cell` such as "0.88**", where the stars
    mark a significant difference in either direction (two-sided) at
    10/5/1%. Ratio < 1 means the model beats the benchmark.
    """
    columns = list(columns or actual.columns)
    idx = actual.index
    for f in forecasts.values():
        idx = idx.intersection(f.index)
    a = actual.loc[idx, columns]
    mask = a.notna()
    for f in forecasts.values():
        mask &= f.loc[idx, columns].notna()
    errs = {n: (f.loc[idx, columns] - a).where(mask) for n, f in forecasts.items()}
    rows = {}
    for col in columns:
        e_b = errs[benchmark][col]
        row = {}
        for n in forecasts:
            if n == benchmark:
                continue
            e_m = errs[n][col]
            ratio = np.sqrt((e_m**2).mean() / (e_b**2).mean())
            dm = diebold_mariano(e_b, e_m, h)
            row[(n, "ratio")], row[(n, "dm")], row[(n, "p")] = ratio, dm["stat"], dm["pvalue"]
            row[(n, "cell")] = f"{ratio:.2f}{_stars(dm['pvalue'])}"
        rows[col] = row
    out = pd.DataFrame(rows).T
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def cw_table(forecasts: dict[str, pd.DataFrame], small: str, actual: pd.DataFrame, h: int,
             columns=None) -> pd.DataFrame:
    """RMSE ratio vs the nested `small` model and Clark-West test for every larger model and maturity.

    Same layout as `dm_table` (columns `ratio`, `cw`, `p`, `cell`), on a common
    sample. Stars mark a one-sided rejection in favour of the larger model at
    10/5/1%. Every other model in `forecasts` must nest `small`.
    """
    columns = list(columns or actual.columns)
    idx = actual.index
    for f in forecasts.values():
        idx = idx.intersection(f.index)
    a = actual.loc[idx, columns]
    mask = a.notna()
    for f in forecasts.values():
        mask &= f.loc[idx, columns].notna()
    fc = {n: f.loc[idx, columns].where(mask) for n, f in forecasts.items()}
    rows = {}
    for col in columns:
        fs = fc[small][col]
        es = fs - a[col]
        row = {}
        for n in forecasts:
            if n == small:
                continue
            fl = fc[n][col]
            el = fl - a[col]
            ratio = np.sqrt((el**2).mean() / (es**2).mean())
            cw = clark_west(es, el, fs, fl, h)
            row[(n, "ratio")], row[(n, "cw")], row[(n, "p")] = ratio, cw["stat"], cw["pvalue"]
            row[(n, "cell")] = f"{ratio:.2f}{_stars(cw['pvalue'])}"
        rows[col] = row
    out = pd.DataFrame(rows).T
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


# --------------------------------------------------------------------------- model confidence set
def _block_bootstrap_means(L: np.ndarray, block: int, n_boot: int, rng) -> np.ndarray:
    """Moving-block bootstrap of the column means of L (T x m): returns (n_boot, m).

    Each resample joins ceil(T / block) blocks of `block` consecutive rows with
    uniform random starts, truncated to T rows. Block sums come from prefix sums,
    so a resample costs O(blocks x m) instead of O(T x m).
    """
    T, m = L.shape
    block = int(min(max(block, 1), T))
    nb = -(-T // block)
    cs = np.vstack([np.zeros((1, m)), np.cumsum(L, axis=0)])
    starts = rng.integers(0, T - block + 1, size=(n_boot, nb))
    sums = (cs[starts + block] - cs[starts]).sum(axis=1)  # (n_boot, m): nb full blocks
    extra = nb * block - T  # trim the surplus rows off the end of the last block
    if extra:
        last = starts[:, -1]
        sums -= cs[last + block] - cs[last + block - extra]
    return sums / T


def mcs(losses: pd.DataFrame, block: int = 3, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Hansen-Lunde-Nason model confidence set with the T_max statistic.

    `losses` is T x m (one column per model, no missing values). At each step,
    model i's relative loss is d_i = mean_t(L_it - mean_j L_jt) over the
    surviving models; t_i = d_i / sd(d_i), with sd from a moving-block
    bootstrap (blocks of `block` months, to respect the overlap of h-step
    errors). The test of "all surviving models are equally accurate" uses
    T_max = max_i t_i against its bootstrap distribution; if rejected, the
    model with the largest t_i goes. The MCS p-value of a model is the largest
    test p-value up to its elimination, so the set {p >= alpha} contains the
    best model with probability >= 1 - alpha asymptotically. The last model
    standing has p = 1.

    Returns one row per model: mean loss, MCS p-value and elimination rank
    (1 = first eliminated; the survivor gets m).
    """
    names = list(losses.columns)
    L = losses.to_numpy(float)
    if not np.isfinite(L).all():
        raise ValueError("mcs needs a common sample without missing losses")
    rng = np.random.default_rng(seed)
    Lbar = L.mean(axis=0)
    Lstar = _block_bootstrap_means(L, block, n_boot, rng)  # (n_boot, m)
    alive = list(range(len(names)))
    pvals, ranks, p_run = {}, {}, 0.0
    for rank in range(1, len(names)):
        a = np.array(alive)
        d = Lbar[a] - Lbar[a].mean()
        dstar = Lstar[:, a] - Lstar[:, a].mean(axis=1, keepdims=True)
        sd = np.sqrt(((dstar - d) ** 2).mean(axis=0))
        sd = np.where(sd > 0, sd, np.inf)  # identical models: never the worst
        t = d / sd
        tmax_star = ((dstar - d) / sd).max(axis=1)
        p_run = max(p_run, float(np.mean(tmax_star >= t.max())))
        worst = alive[int(np.argmax(t))]
        pvals[worst], ranks[worst] = p_run, rank
        alive.remove(worst)
    pvals[alive[0]], ranks[alive[0]] = 1.0, len(names)
    return pd.DataFrame({"mean_loss": Lbar, "p_mcs": [pvals[i] for i in range(len(names))],
                         "rank": [ranks[i] for i in range(len(names))]}, index=pd.Index(names, name="model"))


def mcs_table(forecasts: dict[str, pd.DataFrame], actual: pd.DataFrame, h: int, columns=None,
              block: int | None = None, n_boot: int = 2000, seed: int = 0, pooled: bool = True) -> pd.DataFrame:
    """MCS p-values (squared-error loss) for every model, one column per maturity.

    Each maturity uses the targets where the realised yield and every model's
    forecast exist; a model with no forecasts at all for a maturity (e.g.
    GBP forwards at h = 1, which need the 1m yield) is left out there and
    gets NaN. With `pooled`, an extra column 'all' uses the loss averaged over
    maturities, on the dates where every maturity is available.
    The block length defaults to max(h, 3) months. A model is in the 90% MCS
    when its p-value is >= 0.10.
    """
    columns = list(columns or actual.columns)
    block = block or max(h, 3)
    idx = actual.index
    for f in forecasts.values():
        idx = idx.intersection(f.index)
    a = actual.loc[idx, columns]
    sq = {n: (f.loc[idx, columns] - a) ** 2 for n, f in forecasts.items()}
    out = {}

    def run(L: pd.DataFrame) -> pd.Series:
        return mcs(L.dropna(axis=1, how="all").dropna(), block, n_boot, seed)["p_mcs"]

    for col in columns:
        out[col] = run(pd.DataFrame({n: e[col] for n, e in sq.items()}))
    if pooled:
        out["all"] = run(pd.DataFrame({n: e.mean(axis=1, skipna=False) for n, e in sq.items()}))
    return pd.DataFrame(out).reindex(list(forecasts))
