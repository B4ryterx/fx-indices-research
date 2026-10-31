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

---

# Project 3: a control-barrier-function drawdown guard

**Question**: on the project-1 books, does a CBF guard keep drawdowns inside a limit more cheaply than
plain vol targeting or the classic Grossman-Zhou (1993) drawdown rule?

**Guard** (`src/guard.py`): safe set h = L − DD ≥ 0 with L = 15% drawdown from the 1-year high. The discrete CBF
condition h(t+1) ≥ (1−γ)·h(t) under a k-sigma one-day loss gives the exposure

  s = min(1, (γ/k) · h / (σ̂ · (1 − DD)))

i.e. **Grossman-Zhou with its constant multiplier replaced by γ/(k·σ̂)**: it de-risks harder when volatility is high.
"Vol-aware" uses σ̂ = max(10-day, 60-day EWMA), as in the market maker's vol-aware barrier.

**Protocol** (`src/eval_guard.py`, `src/frontier.py`): one parameter per guard, chosen on 1990–2007 (most return with
drawdown ≤ L); 2008–2026 untouched. Then a return-vs-breach-risk frontier over 200 block-bootstrap histories
(63-day blocks), the same histories for every guard.

## Results

Out of sample, 2008–2026, settings chosen on 1990–2007 (`results/guard_summary.csv`):

| Book | Guard | Ann. return | Sharpe | Max DD (1-yr) | Days over limit |
|---|---|---|---|---|---|
| Long-only indices | none (vol target) | 5.8% | 0.47 | 31.9% | 15.8% |
| | Grossman-Zhou | 5.0% | 0.48 | 15.1% | 0.2% |
| | **CBF, vol-aware** | 5.4% | **0.52** | 15.0% | **0** |
| Trend+carry book | none | −0.6% | −0.05 | 19.3% | 8.5% |
| | Grossman-Zhou | −2.2% | −0.20 | 15.0% | 1.1% |
| | **CBF, vol-aware** | −1.4% | −0.15 | 13.9% | **0** |
| 70/30 blend | none | 4.1% | 0.38 | 24.1% | 10.7% |
| | Grossman-Zhou | 3.6% | 0.36 | 14.9% | 0 |
| | **CBF, vol-aware** | 4.4% | **0.46** | 14.6% | **0** |

Frontier (best return with bootstrap P(breach) ≤ 5%): vol-aware CBF beats Grossman-Zhou by **+38 bp/yr** (long-only),
**+33 bp/yr** (blend) and +11 bp/yr (trend book), ahead on 74–82% of paired histories.

**Verdict**: the vol-aware barrier is a strict improvement on Grossman-Zhou here: same or better safety, more return.
On the blend it even beats doing nothing on Sharpe (0.46 vs 0.38) while halving the worst drawdown. It does not
create alpha: on the trend book (no edge) every guard loses a little to insurance cost.

**Caveats**
- The limit is from the 1-year high, so losses can still add up beyond L over longer spans (all-time max DD ~18–20%).
- Frontier t-stats (10–12) overstate certainty: the paths are resampled from one 19-year history.
- Settings chosen "at the edge" of the training limit are fragile; bootstrap breach odds for those single points
  swing from 0% to 74%, which is why the frontier is the main comparison.
- Daily data: an intraday gap larger than k·σ̂ can still breach the limit; no guard can prevent that.

---

# Project 2: pairs trading (cointegration and distance), daily

**Question**: do cointegrated or distance-matched pairs among the 7 FX majors and 11 indices mean-revert
profitably out of sample?

**Method** (`src/pairs.py`, `src/eval_pairs.py`): walk-forward from 2004. Every 6 months, look back 2 years,
pick the 5 best pairs (Engle-Granger p < 0.05 with a 5–126-day half-life, or the 5 closest normalised price paths
for the distance method), freeze their hedge ratios, then trade only the next 6 months: enter at |z| > 2, exit at
|z| < 0.5, stop at |z| > 4, flat at the window end. Same costs as project 1. Placebo: 20 runs of random pairs under
the same rules.

## Results (2006–2026, net)

| Variant | Sharpe | NW t | Beats random pairs |
|---|---|---|---|
| Cointegration, any pair | 0.28 | 1.64 | 100% |
| Cointegration, indices, any session | 0.26 | 1.72 | 100% |
| **Cointegration, same session only** | **−0.09** | −0.43 | 80% |
| Cointegration, indices, same session | −0.17 | −0.82 | 50% |
| Cointegration, FX | 0.07 | 0.34 | 95% |
| Distance, same session | −0.44 | −2.18 | 10% |

**The apparent edge was a timing artifact.** Without a session filter the most-traded pairs were FTSE/ASX, SPX/ASX,
RUT/ASX and DJI/ASX. The ASX closes about 10 hours before London and New York, so a "spread" between their daily
closes partly measures news that arrived after Sydney shut, and that gap "reverts" the next day when the ASX catches up.
That's stale prices, not tradable mean reversion. Restricting pairs to markets that close together removes all of it.

**Verdict**: no tradable pairs edge on daily closes in this universe. FX crosses are not reliably cointegrated
(40% of trades converge, 28% hit the stop), and same-session index pairs (FTSE/CAC, CAC/SX5E) revert too rarely to
beat costs. The distance method loses money: its pairs (SPX/DJI, CAC/SX5E, EUR/CHF) drift apart slowly rather than revert.

**Next**: the hourly Dukascopy data allows a fair intraday test using simultaneous snapshots across sessions.
