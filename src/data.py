"""Turn the raw Yahoo/FRED downloads into one aligned daily panel.

Everything is quoted as "USD per unit of the asset", so a long position in
JPY means long yen, short dollar.  FX daily returns include carry: the
interest you earn on the currency you hold minus the dollar rate you pay.
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")

# name: (yahoo file, invert?, rate country)  -- inverted pairs are quoted USDXXX on Yahoo
FX = {"EUR": ("EURUSD_X", False, "EZ"), "GBP": ("GBPUSD_X", False, "GB"),
      "AUD": ("AUDUSD_X", False, "AU"), "NZD": ("NZDUSD_X", False, "NZ"),
      "JPY": ("JPY_X", True, "JP"), "CAD": ("CAD_X", True, "CA"), "CHF": ("CHF_X", True, "CH")}
IDX = {"SPX": "GSPC", "NDX": "NDX", "DJI": "DJI", "RUT": "RUT", "DAX": "GDAXI", "FTSE": "FTSE",
       "CAC": "FCHI", "SX5E": "STOXX50E", "NKY": "N225", "HSI": "HSI", "ASX": "AXJO"}
SPIKE = 0.03     # Yahoo bad ticks: a >3% move that is >70% undone the very next day


def yahoo(name):
    r = json.load(open(os.path.join(RAW, "yahoo", name + ".json")))["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
    close = r["indicators"]["quote"][0]["close"]
    s = pd.Series(close, index=ts, dtype=float).dropna()
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")]


def rate(country):
    """Monthly 3m rate in % -> daily decimal, lagged one month (FRED publishes after month end)."""
    df = pd.read_csv(os.path.join(RAW, "fred", country + ".csv"), index_col=0, parse_dates=True)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna() / 100
    s.index = s.index + pd.offsets.MonthBegin(1)        # value for month m usable from m+1
    return s


def clean_fx(px):
    """Drop single-day spikes (out and straight back).  Real shocks such as the
    2015 SNB de-peg or Brexit do not reverse the next day, so they survive."""
    for _ in range(3):
        lr = np.log(px).diff()
        bad = (lr.abs() > SPIKE) & ((lr + lr.shift(-1)).abs() < 0.3 * lr.abs())
        if not bad.any():
            break
        px = px[~bad]
    return px


def load():
    """Returns (prices, returns, carry) as DataFrames on a common weekday calendar.

    prices  : USD per unit (FX) or index level, forward-filled across holidays
    returns : daily excess returns (FX includes the interest differential)
    carry   : annualised carry in decimal (FX only; NaN for indices)
    """
    days = pd.bdate_range("1985-01-01", pd.Timestamp.today().normalize())
    us = rate("US").reindex(days, method="ffill")
    prices, rets, carry = {}, {}, {}
    for k, (f, inv, ctry) in FX.items():
        px = clean_fx(yahoo(f))
        px = 1 / px if inv else px
        px = px.reindex(days).ffill(limit=5)
        diff = rate(ctry).reindex(days, method="ffill") - us
        prices[k] = px
        carry[k] = diff.where(px.notna())
        rets[k] = px.pct_change(fill_method=None) + diff.shift(1).fillna(0) / 252 * px.notna()
    for k, f in IDX.items():
        px = yahoo(f).reindex(days).ffill(limit=5)
        prices[k] = px
        carry[k] = pd.Series(np.nan, index=days)
        rets[k] = px.pct_change(fill_method=None)       # price return; see README on dividends
    P, R, C = (pd.DataFrame(d) for d in (prices, rets, carry))
    R = R.where(P.notna())
    return P, R, C


if __name__ == "__main__":
    P, R, C = load()
    first = P.apply(pd.Series.first_valid_index)
    out = pd.DataFrame({"first": first.dt.date, "ann_vol": R.std() * np.sqrt(252),
                        "max_abs_day": R.abs().max(), "carry_now": C.ffill().iloc[-1]})
    print(out.round(4).to_string())
