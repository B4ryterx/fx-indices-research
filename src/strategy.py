"""Trend + carry, volatility-targeted, in the style of Carver (Systematic Trading)
and Moskowitz-Ooi-Pedersen (Time Series Momentum, 2012).

All parameters are textbook defaults fixed before looking at results; nothing
here is fitted to the data.  Every rolling quantity uses only past data and the
position decided at the close of day t earns the return of day t+1.
"""
import numpy as np, pandas as pd

SPEEDS = [(8, 32), (16, 64), (32, 128), (64, 256)]   # EWMAC fast/slow spans in days
VOL_SPAN = 35                # EWMA span for the volatility estimate
FCAST_AVG = 10.0             # forecasts are scaled so their average |value| is 10 ...
FCAST_CAP = 20.0             # ... and capped at +-20 (Carver's convention)
TARGET_VOL = 0.10            # 10% annualised risk target for each sleeve
BUFFER = 0.10                # don't trade unless the position is >10% (of its average size) away
COST_BP = {"fx": 1.0, "idx": 1.5}   # cost per unit of notional traded, in basis points
MIN_HIST = 256               # an instrument trades only after a year of data


def ewm_vol(R):
    """Annualised EWMA vol of daily returns, floored at the 5th percentile of its own past."""
    v = R.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std() * np.sqrt(252)
    return np.maximum(v, v.expanding(252).quantile(0.05))


def scale(F):
    """Scale raw forecasts so the pooled expanding mean |F| is FCAST_AVG, then cap.
    Uses only data up to each day, pooled across instruments."""
    pooled = F.abs().stack().groupby(level=0).mean()
    s = FCAST_AVG / pooled.expanding(MIN_HIST).mean()
    return (F.mul(s, axis=0)).clip(-FCAST_CAP, FCAST_CAP)


def trend_forecast(P, R):
    """Average of 4 EWMAC speeds; each is (fast EMA - slow EMA) / price vol."""
    lp = np.log(P)
    psd = ewm_vol(R) / np.sqrt(252)                   # daily return vol ~ daily log-price vol
    fs = [scale((lp.ewm(span=f).mean() - lp.ewm(span=s).mean()) / psd) for f, s in SPEEDS]
    return scale(sum(fs) / len(fs))                   # rescale: averaging shrinks |F|


def carry_forecast(C, R):
    """Interest differential / vol = carry as a Sharpe ratio; smoothed over a month."""
    return scale((C / ewm_vol(R)).ewm(span=20).mean())


def positions(F, R):
    """Vol-targeted notional per unit of capital, with equal risk across live instruments.
    The diversification multiplier is the realised ratio target / portfolio vol, measured
    on the previous year only."""
    vol = ewm_vol(R)
    live = F.notna() & vol.notna()
    n = live.sum(axis=1).replace(0, np.nan)
    raw = (F / FCAST_AVG) * (TARGET_VOL / vol)
    raw = raw.div(np.sqrt(n), axis=0)                 # sqrt(n) = the multiplier if uncorrelated
    pnl = (raw.shift(1) * R).sum(axis=1, min_count=1)
    realised = pnl.rolling(252, min_periods=126).std() * np.sqrt(252)
    lev = (TARGET_VOL / realised).clip(0.5, 2.0).shift(1).fillna(1.0)
    return raw.mul(lev, axis=0).where(live)


def buffered(W):
    """Trade only when the target leaves a band of BUFFER * average |position| around
    the current holding; then trade to the band edge (Carver ch. 11)."""
    band = BUFFER * W.abs().expanding(20).mean()
    held = np.zeros(W.shape[1]); out = np.full(W.shape, np.nan)
    for i, (w, b) in enumerate(zip(W.values, band.values)):
        w = np.nan_to_num(w); b = np.nan_to_num(b)
        held = np.where(w > held + b, w - b, np.where(w < held - b, w + b, held))
        out[i] = held
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def backtest(W, R, kinds):
    """Daily net returns of holding W (decided at close t) over day t+1."""
    W = buffered(W)
    gross = (W.shift(1) * R.fillna(0)).sum(axis=1)
    cost = pd.Series({c: COST_BP[kinds[c]] / 1e4 for c in W.columns})
    tc = (W.diff().abs() * cost).sum(axis=1)
    return pd.DataFrame({"gross": gross, "cost": tc, "net": gross - tc,
                         "turnover": W.diff().abs().sum(axis=1) * 252})
