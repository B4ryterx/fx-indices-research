"""Download monthly 3-month interest rates from FRED into data/raw/fred."""
import os, subprocess, time
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "fred")
IDS = {"US": "TB3MS", **{c: f"IR3TIB01{c}M156N" for c in ["EZ", "GB", "JP", "AU", "CA", "CH", "NZ"]}}
os.makedirs(RAW, exist_ok=True)
for c, sid in IDS.items():
    out = os.path.join(RAW, c + ".csv")
    subprocess.run(["/usr/bin/curl", "-sf", "--http1.1", "--retry", "3", "-o", out,
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"], check=True)
    lines = open(out).read().splitlines()
    print(c, sid, len(lines) - 1, lines[1], lines[-1])
    time.sleep(1)
