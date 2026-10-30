# Trend and carry on FX majors and equity indices

Project 1 of the forex/indices track. A textbook (Carver / Moskowitz-Ooi-Pedersen)
trend-and-carry system, built with **no fitted parameters**, scored honestly.

**Universe**: 7 FX majors vs USD (EUR, GBP, AUD, NZD, JPY, CAD, CHF; 1996/2003→) and
11 equity indices (SPX, NDX, DJI, RUT, DAX, FTSE, CAC, SX5E, NKY, HSI, ASX; 1985→).
**Data**: Yahoo daily closes; FRED 3-month rates for FX carry (lagged one month).

## Method
- Forecasts: EWMAC trend at 4 speeds (8/32 … 64/256), FX carry = rate differential / vol.
  Scaled to average |10|, capped at 20, using expanding (past-only) statistics.
- Positions: 10% vol target, equal risk per instrument, leverage from last year's realised vol.
- Trading: decided at close t, earns day t+1; 10% no-trade buffer; costs 1 bp (FX) / 1.5 bp (indices) per unit traded.
- Evaluation: Newey-West t-stats, 21-day block-bootstrap Sharpe CI, Deflated Sharpe Ratio over all
  12 variants tried, sub-periods, and beta/alpha vs a long-only equal-risk index benchmark.

## Results (net of costs; full table in `results/summary.csv`)

| Sleeve | Sharpe | 95% CI | Equity beta | Alpha t | Deflated SR prob |
|---|---|---|---|---|---|
| FX trend | 0.04 | −0.30, 0.39 | −0.02 | 0.28 | 0.06 |
| FX carry | 0.04 | −0.36, 0.44 | 0.11 | −0.20 | 0.08 |
| FX carry, cross-sectional | 0.15 | −0.26, 0.56 | 0.23 | −0.05 | 0.20 |
| Index trend | 0.38 | 0.05, 0.71 | 0.46 | 0.54 | 0.63 |
| All combined | 0.38 | 0.07, 0.72 | 0.42 | 0.70 | 0.65 |
| Long-only indices (benchmark) | 0.74 | 0.36, 1.10 | — | — | — |

**Verdict: no sleeve has significant alpha.** Index trend's Sharpe is mostly equity beta (trend is
long most of the time because indices drift up); FX trend and carry are flat since 2003, consistent with the
post-2008 literature. No variant passes the Deflated Sharpe test at 95%. Slow speeds did best, but choosing them
now would be data-mining.

**What does hold up: trend as a crash cushion.** In the benchmark's worst 10% of months the
combined book lost 0.7% on average vs 7.0% for the benchmark. Mixing 30% trend with 70% long-only (1990–2026)
keeps Sharpe 0.61 vs 0.62 and cuts the maximum drawdown from 43% to 28% (50/50: 22%).

## Limitations
- Index returns are price returns (Yahoo has no dividends) with no financing charge; the two roughly offset
  for a futures-style position, but not exactly.
- Yahoo FX has bad ticks; `data.py` removes one-day out-and-back spikes and keeps real shocks (SNB 2011/2015, Brexit).
- Daily closes in different time zones; each instrument is traded on its own close.
- Dukascopy intraday bid/ask data was blocked by the network's TLS inspection; needed for project 2.

## Run
```
python3 src/fetch.py && python3 src/fetch_fred.py   # downloads ~16 MB
python3 src/evaluate.py                             # ~5 s
```
