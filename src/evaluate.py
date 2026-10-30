"""Run every sleeve and score it honestly: costs, Newey-West t-stats, block-bootstrap
Sharpe intervals, the Deflated Sharpe Ratio for the number of variants tried,
sub-periods, and alpha against a long-only equity benchmark."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, FX, IDX
import strategy as S

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PERIODS = [("1990-2002", "1990", "2002"), ("2003-2014", "2003", "2014"), ("2015-2026", "2015", "2026")]


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)


def nw_tstat(r, lags=10):
    x = r.values; m = x.mean(); e = x - m; n = len(x)
    v = e @ e / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return m / np.sqrt(v / n)


def boot_sharpe(r, block=21, n=2000, seed=0):
    """95% CI of the annualised Sharpe from a stationary-ish moving-block bootstrap."""
    rng = np.random.default_rng(seed); x = r.values; T = len(x)
    k = int(np.ceil(T / block)); out = np.empty(n)
    for i in range(n):
        idx = (rng.integers(0, T - block, k)[:, None] + np.arange(block)).ravel()[:T]
        s = x[idx]; out[i] = s.mean() / s.std() * np.sqrt(252)
    return np.percentile(out, [2.5, 97.5])


def deflated_sharpe(r, n_trials, sr_trials_sd):
    """Bailey & Lopez de Prado (2014): probability the true Sharpe > the best Sharpe
    you'd expect from n_trials useless strategies.  Daily units throughout."""
    sr = r.mean() / r.std(); T = len(r)
    g = 0.5772156649
    e_max = sr_trials_sd * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials)
                            + g * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    sk, ku = stats.skew(r), stats.kurtosis(r, fisher=False)
    z = (sr - e_max) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return stats.norm.cdf(z)


def summary(name, bt, bench=None):
    r = bt["net"]; r = r[r.ne(0).idxmax():]          # start at first trade
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    m = (1 + r).resample("ME").prod() - 1
    row = dict(sleeve=name, start=r.index[0].year, ann_ret=r.mean() * 252, ann_vol=r.std() * np.sqrt(252),
               sharpe=sharpe(r), nw_t=nw_tstat(r), max_dd=dd, monthly_skew=stats.skew(m),
               cost_bp=bt["cost"].loc[r.index].mean() * 252 * 1e4, turnover=bt["turnover"].loc[r.index].mean())
    lo, hi = boot_sharpe(r); row.update(sr_lo=lo, sr_hi=hi)
    for lbl, a, b in PERIODS:
        x = r.loc[a:b]; row["SR " + lbl] = sharpe(x) if len(x) > 252 else np.nan
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        beta = np.cov(r, b)[0, 1] / b.var()
        res = r - beta * b
        row.update(beta_eq=beta, alpha_sr=sharpe(res), alpha_t=nw_tstat(res))
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    P, R, C = load()
    kinds = {c: ("fx" if c in FX else "idx") for c in P.columns}
    fx, ix = list(FX), list(IDX)
    T = S.trend_forecast(P, R); K = S.carry_forecast(C, R)
    hist = P.notna().cumsum() > S.MIN_HIST
    T, K = T.where(hist), K.where(hist)

    def run(F, cols):
        return S.backtest(S.positions(F[cols], R[cols]), R[cols], kinds)

    long_eq = run(pd.DataFrame(S.FCAST_AVG, index=P.index, columns=ix).where(hist[ix]), ix)
    bench = long_eq["net"]
    combo_fx = S.scale(0.5 * T[fx] + 0.5 * K[fx])
    sleeves = {
        "FX trend": run(T, fx),
        "FX carry": run(K, fx),
        "FX trend+carry": run(combo_fx.reindex(columns=P.columns), fx),
        "Index trend": run(T, ix),
        "All: FX trend+carry, index trend": run(pd.concat([combo_fx, T[ix]], axis=1), fx + ix),
        "Benchmark: long-only indices": long_eq,
    }
    # cross-sectional (relative) versions: subtract the cross-sectional mean forecast, so the
    # book is roughly dollar-neutral (FX) or market-neutral (indices)
    xs = lambda F: S.scale(F.sub(F.mean(axis=1), axis=0))
    sleeves["FX carry, cross-sectional"] = run(xs(K[fx]), fx)
    sleeves["FX trend, cross-sectional"] = run(xs(T[fx]), fx)
    sleeves["Index trend, cross-sectional"] = run(xs(T[ix]), ix)
    for f, s in S.SPEEDS:                             # robustness: each speed alone
        lp = np.log(P); psd = S.ewm_vol(R) / np.sqrt(252)
        F1 = S.scale((lp.ewm(span=f).mean() - lp.ewm(span=s).mean()) / psd).where(hist)
        sleeves[f"All trend, EWMAC {f}/{s} only"] = run(F1, fx + ix)

    rows = [summary(k, v, None if k.startswith("Benchmark") else bench) for k, v in sleeves.items()]
    tab = pd.DataFrame(rows).set_index("sleeve")
    # Deflated Sharpe: every non-benchmark row counts as a trial
    trials = [k for k in sleeves if not k.startswith("Benchmark")]
    srs = [sleeves[k]["net"].pipe(lambda r: r[r.ne(0).idxmax():]) for k in trials]
    sd = np.std([x.mean() / x.std() for x in srs])
    tab["deflated_SR_prob"] = [deflated_sharpe(sleeves[k]["net"].pipe(lambda r: r[r.ne(0).idxmax():]),
                                                len(trials), sd) if k in trials else np.nan for k in tab.index]
    tab.to_csv(os.path.join(OUT, "summary.csv"))
    pd.DataFrame({k: v["net"] for k, v in sleeves.items()}).to_csv(os.path.join(OUT, "daily_net.csv"))
    pd.set_option("display.width", 250)
    cols = ["start", "ann_ret", "ann_vol", "sharpe", "sr_lo", "sr_hi", "nw_t", "max_dd", "monthly_skew",
            "cost_bp", "SR 1990-2002", "SR 2003-2014", "SR 2015-2026", "beta_eq", "alpha_t", "deflated_SR_prob"]
    print(tab[cols].round(2).to_string())
    yr = pd.DataFrame({k: v["net"] for k, v in sleeves.items() if not "only" in k}).resample("YE").sum()
    yr.index = yr.index.year
    yr.loc[2000:].round(3).to_csv(os.path.join(OUT, "by_year.csv"))
    print("\nBy year (sum of daily net returns):\n", yr.loc[2000:].round(3).to_string())


if __name__ == "__main__":
    main()
