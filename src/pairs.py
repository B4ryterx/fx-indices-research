"""Pairs trading on daily closes: Engle-Granger cointegration (Vidyamurthy; Chan,
Algorithmic Trading ch. 2-3) and the Gatev-Goetzmann-Rouwenhorst (2006) distance method.

Walk-forward: every TRADE days, look back FORM days to choose pairs and fix their
hedge ratios, then trade only the following TRADE days with those numbers frozen.
Nothing about a trading window is known when its pairs are chosen."""
import itertools
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import coint

FORM, TRADE = 504, 126          # 2-year formation, 6-month trading window
N_PAIRS = 5
ENTRY, EXIT, STOP = 2.0, 0.5, 4.0
HL_RANGE = (5, 126)             # half-life (days) the spread must revert within
# Markets whose daily closes are at (nearly) the same time.  Pairing across sessions
# (e.g. ASX closes ~10 h before SPX) creates fake mean reversion from stale prices.
SESSION = {"SPX": "us", "NDX": "us", "DJI": "us", "RUT": "us",
           "DAX": "eu", "FTSE": "eu", "CAC": "eu", "SX5E": "eu",
           "NKY": "asia", "HSI": "asia", "ASX": "asia"}       # FX: all one 24h market, "fx"
PAIR_VOL = 0.10                 # annualised vol target for the whole pairs book


def half_life(e):
    """OU half-life from the AR(1) fit  de_t = phi * e_{t-1} + noise."""
    de, lag = np.diff(e), e[:-1] - e[:-1].mean()
    phi = (lag @ de) / (lag @ lag)
    return -np.log(2) / np.log(1 + phi) if -1 < phi < 0 else np.inf


def fit_pair(ya, xb):
    """OLS hedge in log prices: ya = alpha + beta * xb + e."""
    X = np.column_stack([np.ones_like(xb), xb])
    (alpha, beta), *_ = np.linalg.lstsq(X, ya, rcond=None)
    e = ya - alpha - beta * xb
    return alpha, beta, e


def select(lp, method, rng=None, same_session=False):
    """Choose up to N_PAIRS pairs from a formation window of log prices.
    Returns list of (a, b, alpha, beta, sd_e, sd_de)."""
    cols = [c for c in lp.columns if lp[c].notna().all()]
    cands = []
    for a, b in itertools.combinations(cols, 2):
        if same_session and SESSION.get(a, "fx") != SESSION.get(b, "fx"):
            continue
        ya, xb = lp[a].values, lp[b].values
        alpha, beta, e = fit_pair(ya, xb)
        if method == "coint":
            if beta <= 0:
                continue
            p = coint(ya, xb, maxlag=1, autolag=None)[1]
            hl = half_life(e)
            if p < 0.05 and HL_RANGE[0] <= hl <= HL_RANGE[1]:
                cands.append((p, a, b, alpha, beta, e))
        elif method == "distance":           # normalised prices, beta = 1, rank by squared distance
            na, nb = ya - ya[0], xb - xb[0]
            e = na - nb
            cands.append((float(e @ e), a, b, ya[0] - xb[0] + e.mean(), 1.0, e - e.mean()))
        elif method == "random":
            cands.append((rng.random(), a, b, alpha, max(beta, 0.0) or 1.0, ya - alpha - (max(beta, 0.0) or 1.0) * xb))
    cands.sort(key=lambda c: c[0])
    return [(a, b, al, be, e.std(), np.diff(e).std()) for _, a, b, al, be, e in cands[:N_PAIRS]]


def trade_window(lp, pairs):
    """Positions (notional per unit capital) for each instrument over one trading window.
    Each open pair holds +k of a and -k*beta of b (long spread) or the reverse, with k set
    so the pair's spread runs at PAIR_VOL / sqrt(N_PAIRS)."""
    W = pd.DataFrame(0.0, index=lp.index, columns=lp.columns)
    trades = []
    for a, b, alpha, beta, sd_e, sd_de in pairs:
        z = ((lp[a] - alpha - beta * lp[b]) / sd_e).values
        k = (PAIR_VOL / np.sqrt(N_PAIRS) / np.sqrt(252)) / sd_de
        pos, stopped, opened = 0, False, None
        side = np.zeros(len(z))
        for t, zt in enumerate(z):
            if not np.isfinite(zt):
                pass
            elif pos == 0 and not stopped:
                if zt > ENTRY: pos, opened = -1, t
                elif zt < -ENTRY: pos, opened = 1, t
            elif pos != 0:
                if abs(zt) > STOP:
                    trades.append((a, b, opened, t, "stop")); pos, stopped = 0, True
                elif (pos == 1 and zt > -EXIT) or (pos == -1 and zt < EXIT):
                    trades.append((a, b, opened, t, "converged")); pos = 0
            side[t] = pos
        if pos != 0:
            trades.append((a, b, opened, len(z) - 1, "window end"))
        side[-1] = 0                                   # flat at the end of every window
        W[a] += side * k
        W[b] -= side * k * beta
    return W, trades


def walk_forward(P, cols, method, seed=0, same_session=False):
    """Positions over the full history for one selection method and instrument set."""
    lp = np.log(P[cols])
    rng = np.random.default_rng(seed)
    Ws, trades = [], []
    for start in range(FORM, len(lp) - 1, TRADE):
        form = lp.iloc[start - FORM:start]
        pairs = select(form, method, rng, same_session)
        win = lp.iloc[start:start + TRADE]
        W, tr = trade_window(win, pairs)
        Ws.append(W); trades += [(win.index[0],) + t for t in tr]
    return pd.concat(Ws), trades


def pnl(W, R, bp):
    """Signal at close t, position held over day t+1; costs on every change of notional."""
    W = W.reindex(R.index).fillna(0)
    gross = (W.shift(1) * R[W.columns].fillna(0)).sum(axis=1)
    cost = (W.diff().abs() * bp[W.columns]).sum(axis=1)
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                         "turnover": W.diff().abs().sum(axis=1) * 252})
