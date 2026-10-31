"""Drawdown guards that scale a whole book's exposure by s_t in [0, 1].

Safe set: h_t = L - DD_t >= 0, where DD_t is the drawdown from the running peak
(or from the peak of the last `window` days) and L is the drawdown limit.

Discrete-time control barrier function (CBF) condition, imposed on a k-sigma
one-day loss:      h_{t+1} >= (1 - gamma) * h_t
With exposure s and loss l = s * k * sigma, DD_{t+1} = DD_t + (1 - DD_t) * l, so
the largest s that satisfies it (the 1-D min-intervention QP) is

      s_t = min(1, gamma * h_t / (k * sigma_t * (1 - DD_t)))

which is Grossman-Zhou (1993) drawdown control, s = m * h / (1 - DD), with the
constant multiplier m replaced by gamma / (k * sigma_t): it de-risks harder when
volatility is high.  Barrier violations can then only come from a loss bigger
than k forecast sigmas.
"""
from collections import deque
import numpy as np, pandas as pd


def ewm_sd(r, span):
    """Past-only EWMA sd of daily returns (value at t uses returns up to t)."""
    return r.ewm(span=span, min_periods=span).std()


def run_guard(r, L, rule, c=None, m=None, window=252, vol=None, tc=None):
    """Apply a guard to nominal daily book returns r (exposure 1).

    rule   : "none", "gz" (constant multiplier m) or "cbf" (c = gamma / k, needs vol)
    window : drawdown is measured from the highest equity of the last `window` days
             (None = all-time peak, which can lock the book out for years)
    vol    : Series, forecast of next-day sd of r known at close t
    tc     : Series, cost of changing exposure by 1.0 (sum of |notional| x cost per unit)
    Returns DataFrame with the guarded net return, exposure s and drawdown."""
    rv = r.values; n = len(rv)
    sig = vol.values if vol is not None else np.full(n, np.nan)
    cost = tc.values if tc is not None else np.zeros(n)
    eq = 1.0; peak = 1.0; s = 1.0
    hist = np.empty(n); dq = deque()                # monotonic deque -> rolling max in O(1)
    out = np.zeros((n, 3))
    for t in range(n):
        g = s * rv[t]
        eq *= 1 + g
        hist[t] = eq
        if window:
            while dq and hist[dq[-1]] <= eq:
                dq.pop()
            dq.append(t)
            if dq[0] <= t - window:
                dq.popleft()
            peak = hist[dq[0]]
        else:
            peak = max(peak, eq)
        dd = 1 - eq / peak
        h = L - dd
        if rule == "gz":
            new = min(max(m * h / (1 - dd), 0.0), 1.0)
        elif rule == "cbf" and np.isfinite(sig[t]) and sig[t] > 0:
            new = min(max(c * h / (sig[t] * (1 - dd)), 0.0), 1.0)
        else:
            new = 1.0
        fee = abs(new - s) * cost[t]
        eq *= 1 - fee
        out[t] = (g - fee, s, dd)
        s = new
    return pd.DataFrame(out, index=r.index, columns=["r", "s", "dd"])


def vol_forecast(r, aware=True, prior=None):
    """Book-level vol forecast.  aware=True takes the max of a fast and a slow EWMA so
    the guard reacts to a vol spike within days (like the market maker's vol-aware barrier).
    prior: sd to use until the EWMAs have enough data (must be known before r starts)."""
    slow = ewm_sd(r, 60)
    v = np.maximum(slow, ewm_sd(r, 10)) if aware else slow
    return v.fillna(prior) if prior is not None else v
