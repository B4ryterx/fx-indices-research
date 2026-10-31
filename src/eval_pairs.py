"""Project 2: do cointegrated / distance pairs on FX and equity indices make money
out of sample, and does the selection beat random pairs traded with the same rules?"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, FX, IDX
import strategy as S
import pairs as PR
from evaluate import summary, deflated_sharpe, OUT

N_RANDOM = 20


def main():
    P, R, C = load()
    P, R = P.loc["2004":], R.loc["2004":]          # all 18 instruments live from 2004 (SX5E 2007)
    bp = pd.Series({c: S.COST_BP["fx" if c in FX else "idx"] / 1e4 for c in P.columns})
    fx, ix, allc = list(FX), list(IDX), list(FX) + list(IDX)
    runs, trades = {}, {}
    specs = [("Cointegration, any pair", allc, "coint", False), ("Cointegration, FX", fx, "coint", True),
             ("Cointegration, indices any session", ix, "coint", False),
             ("Cointegration, same session", allc, "coint", True),
             ("Cointegration, indices same session", ix, "coint", True),
             ("Distance, same session", allc, "distance", True)]
    for name, cols, method, same in specs:
        W, tr = PR.walk_forward(P, cols, method, same_session=same)
        runs[name] = PR.pnl(W, R, bp); trades[name] = tr
        print(name, "trades:", len(tr), flush=True)
    rand = [PR.pnl(PR.walk_forward(P, allc, "random", seed=s, same_session=True)[0], R, bp)["net"] for s in range(N_RANDOM)]
    rand_sr = np.array([x[x.ne(0).idxmax():].pipe(lambda r: r.mean() / r.std() * np.sqrt(252)) for x in rand])

    rows = [summary(k, v) for k, v in runs.items()]
    tab = pd.DataFrame(rows).set_index("sleeve")
    srs = [v["net"][v["net"].ne(0).idxmax():] for v in runs.values()]
    sd = np.std([x.mean() / x.std() for x in srs])
    tab["deflated_SR_prob"] = [deflated_sharpe(x, len(srs), sd) for x in srs]
    tab["pct_random_beaten"] = [(rand_sr < s).mean() for s in tab["sharpe"]]
    for k, tr in trades.items():
        t = pd.DataFrame(tr, columns=["window", "a", "b", "open", "close", "how"])
        tab.loc[k, "n_trades"] = len(t)
        tab.loc[k, "converged_pct"] = (t.how == "converged").mean()
        tab.loc[k, "stopped_pct"] = (t.how == "stop").mean()
        tab.loc[k, "avg_days"] = (t.close - t.open).mean()
        top = t.groupby(["a", "b"]).size().sort_values(ascending=False).head(4)
        tab.loc[k, "most_traded"] = ", ".join(f"{a}/{b}({n})" for (a, b), n in top.items())
    tab.to_csv(os.path.join(OUT, "pairs_summary.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(tab[["ann_ret", "ann_vol", "sharpe", "sr_lo", "sr_hi", "nw_t", "max_dd", "cost_bp",
               "SR 2003-2014", "SR 2015-2026", "deflated_SR_prob", "pct_random_beaten"]].round(2).to_string())
    print(tab[["n_trades", "converged_pct", "stopped_pct", "avg_days", "most_traded"]].round(2).to_string())
    print(f"\nRandom same-session pairs, same rules ({N_RANDOM} seeds): Sharpe mean {rand_sr.mean():.2f}, "
          f"5-95% {np.percentile(rand_sr, 5):.2f} to {np.percentile(rand_sr, 95):.2f}")
    pd.DataFrame({k: v["net"] for k, v in runs.items()}).to_csv(os.path.join(OUT, "pairs_daily_net.csv"))


if __name__ == "__main__":
    main()
