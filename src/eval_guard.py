"""Project 3: does a CBF drawdown guard beat plain vol targeting and Grossman-Zhou?

Books (from project 1): long-only equal-risk indices, the trend+carry book, and a
70/30 blend.  Each guard's single parameter is chosen on 1990-2007 only (largest
return with max drawdown <= the limit); 2008-2026 is the untouched test.  A block
bootstrap of each book's test-period returns estimates the probability of
breaching the limit."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, FX, IDX
import strategy as S
from guard import run_guard, vol_forecast

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
L = 0.15                                   # drawdown limit, from a 1-year high
TRAIN, TEST = ("1990", "2007"), ("2008", "2026")
C_GRID = np.round(np.geomspace(0.002, 2.0, 31), 4)      # gamma/k for the CBF
M_GRID = np.round(np.geomspace(0.25, 250, 31), 3)        # Grossman-Zhou multiplier


def books():
    P, R, C = load()
    kinds = {c: ("fx" if c in FX else "idx") for c in P.columns}
    fx, ix = list(FX), list(IDX)
    hist = P.notna().cumsum() > S.MIN_HIST
    T = S.trend_forecast(P, R).where(hist); K = S.carry_forecast(C, R).where(hist)
    lo = S.positions(pd.DataFrame(S.FCAST_AVG, index=P.index, columns=ix).where(hist[ix]), R[ix])
    tr = S.positions(pd.concat([S.scale(0.5 * T[fx] + 0.5 * K[fx]), T[ix]], axis=1), R[fx + ix])
    W = {"Long-only indices": lo.reindex(columns=P.columns),
         "Trend+carry book": tr.reindex(columns=P.columns)}
    W["70/30 long-only/trend"] = 0.7 * W["Long-only indices"].fillna(0) + 0.3 * W["Trend+carry book"].fillna(0)
    bp = pd.Series({c: S.COST_BP[kinds[c]] / 1e4 for c in P.columns})
    out = {}
    for k, w in W.items():
        w = S.buffered(w.fillna(0))
        r = (w.shift(1) * R.fillna(0)).sum(axis=1) - (w.diff().abs() * bp).sum(axis=1)
        out[k] = (r.loc["1990":], (w.abs() * bp).sum(axis=1).loc["1990":])
    return out


def stats(g):
    r = g["r"]; eq = (1 + r).cumprod()
    return dict(ann_ret=r.mean() * 252, sharpe=r.mean() / r.std() * np.sqrt(252),
                max_dd_1y=g["dd"].max(), days_over=(g["dd"] > L).mean(),
                worst_alltime_dd=(1 - eq / eq.cummax()).max(), avg_exposure=g["s"].mean())


def pick(r, vol, tc, rule, grid):
    """Largest training-period return whose 1-year drawdown stays within L."""
    best, val = None, -np.inf
    for p in grid:
        kw = dict(c=p) if rule == "cbf" else dict(m=p)
        g = run_guard(r.loc[TRAIN[0]:TRAIN[1]], L, rule, vol=vol, tc=tc, **kw)
        if g["dd"].max() <= L and g["r"].mean() > val:
            best, val = p, g["r"].mean()
    return best if best is not None else grid[0]


def guards(r, tc, params, prior=None):
    va, vs = vol_forecast(r, True, prior), vol_forecast(r, False, prior)
    return {"No guard (vol target only)": run_guard(r, L, "none", tc=tc),
            "Grossman-Zhou": run_guard(r, L, "gz", m=params["gz"], tc=tc),
            "CBF, slow vol": run_guard(r, L, "cbf", c=params["cbf_slow"], vol=vs, tc=tc),
            "CBF, vol-aware": run_guard(r, L, "cbf", c=params["cbf_aware"], vol=va, tc=tc)}


def bootstrap(r, tc, params, prior, n=300, block=63, seed=1):
    """P(1-year drawdown ever > L) over resampled test-period histories."""
    rng = np.random.default_rng(seed); x = r.values; c = tc.values; T = len(x)
    hits = {}
    for i in range(n):
        idx = (rng.integers(0, T - block, T // block + 1)[:, None] + np.arange(block)).ravel()[:T]
        rr = pd.Series(x[idx], index=r.index); cc = pd.Series(c[idx], index=r.index)
        for k, g in guards(rr, cc, params, prior).items():
            hits.setdefault(k, []).append(g["dd"].max() > L)
    return {k: np.mean(v) for k, v in hits.items()}


def main():
    rows = []
    for book, (r, tc) in books().items():
        va, vs = vol_forecast(r, True), vol_forecast(r, False)
        params = {"gz": pick(r, None, tc, "gz", M_GRID),
                  "cbf_slow": pick(r, vs, tc, "cbf", C_GRID),
                  "cbf_aware": pick(r, va, tc, "cbf", C_GRID)}
        test = r.loc[TEST[0]:]
        boot = bootstrap(test, tc.loc[TEST[0]:], params, r.loc[:TRAIN[1]].std())
        for name, g in guards(r, tc, params).items():
            for per, (a, b) in (("train 1990-2007", TRAIN), ("test 2008-2026", TEST)):
                row = dict(book=book, guard=name, period=per, **stats(g.loc[a:b]))
                if per.startswith("test"):
                    row["P_breach_bootstrap"] = boot[name]
                rows.append(row)
        print(book, params, flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "guard_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(tab[tab.period.str.startswith("test")].drop(columns="period").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
