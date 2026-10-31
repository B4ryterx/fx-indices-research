"""Fairer comparison than one tuned point: for every parameter value, measure the
test-period (2008-2026) return and the bootstrap probability of breaching the
limit, using the SAME resampled histories for every guard.  A guard is better if
its curve gives more return for the same breach risk."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guard import run_guard, vol_forecast
from eval_guard import books, L, OUT, TEST, TRAIN

C = np.geomspace(0.01, 1.0, 13)
M = np.geomspace(1.0, 100, 13)
N_PATHS, BLOCK = 200, 63


def paths(r, tc, seed=7):
    rng = np.random.default_rng(seed); T = len(r)
    for _ in range(N_PATHS):
        idx = (rng.integers(0, T - BLOCK, T // BLOCK + 1)[:, None] + np.arange(BLOCK)).ravel()[:T]
        yield pd.Series(r.values[idx], index=r.index), pd.Series(tc.values[idx], index=r.index)


def main():
    rows = []; per_path = {}
    for book, (r, tc) in books().items():
        prior = r.loc[:TRAIN[1]].std()                 # known before the test starts
        r, tc = r.loc[TEST[0]:], tc.loc[TEST[0]:]
        P = list(paths(r, tc))
        vols = [(vol_forecast(x, True, prior), vol_forecast(x, False, prior)) for x, _ in P]
        specs = [("Grossman-Zhou", "gz", m, None) for m in M] + \
                [("CBF, slow vol", "cbf", c, 1) for c in C] + [("CBF, vol-aware", "cbf", c, 0) for c in C]
        for name, rule, p, vi in specs:
            kw = dict(m=p) if rule == "gz" else dict(c=p)
            res = [run_guard(x, L, rule, tc=cc, vol=None if vi is None else vols[i][vi], **kw)
                   for i, (x, cc) in enumerate(P)]
            per_path[(book, name, p)] = np.array([g["r"].mean() * 252 for g in res])
            rows.append(dict(book=book, guard=name, param=p,
                             ann_ret=np.mean([g["r"].mean() * 252 for g in res]),
                             p_breach=np.mean([g["dd"].max() > L for g in res]),
                             avg_exposure=np.mean([g["s"].mean() for g in res])))
        print(book, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "guard_frontier.csv"), index=False)
    # return each guard can earn while keeping breach risk at or under 5% / 10%
    for cap in (0.05, 0.10):
        best = df[df.p_breach <= cap].groupby(["book", "guard"]).ann_ret.max().unstack()
        print(f"\nBest mean annual return with P(breach) <= {cap:.0%}:\n", best.round(4).to_string())
        # paired test on the same resampled histories: vol-aware CBF vs Grossman-Zhou
        ok = df[df.p_breach <= cap]
        for book in df.book.unique():
            pick = lambda gname: ok[(ok.book == book) & (ok.guard == gname)].sort_values("ann_ret").iloc[-1].param
            d = per_path[(book, "CBF, vol-aware", pick("CBF, vol-aware"))] - per_path[(book, "Grossman-Zhou", pick("Grossman-Zhou"))]
            print(f"  {book}: CBF vol-aware minus GZ = {d.mean()*1e4:.0f} bp/yr, "
                  f"t = {d.mean() / d.std() * np.sqrt(len(d)):.1f}, CBF ahead on {np.mean(d > 0):.0%} of paths")


if __name__ == "__main__":
    main()
