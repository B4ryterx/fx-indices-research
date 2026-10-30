"""Download daily prices (Yahoo) and short rates (FRED) into data/raw. Data only, no code."""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
YAHOO = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "JPY=X", "CAD=X", "CHF=X",
         "^GSPC", "^NDX", "^DJI", "^RUT", "^GDAXI", "^FTSE", "^FCHI", "^STOXX50E", "^N225", "^HSI", "^AXJO"]
FRED = {c: f"IR3TIB01{c}M156N" for c in ["US", "EZ", "GB", "JP", "AU", "CA", "CH", "NZ"]}

def get(url, out):
    subprocess.run(["/usr/bin/curl", "-sf", "-A", "Mozilla/5.0", "-o", out, url], check=True)

os.makedirs(os.path.join(RAW, "yahoo"), exist_ok=True)
os.makedirs(os.path.join(RAW, "fred"), exist_ok=True)
for s in YAHOO:
    out = os.path.join(RAW, "yahoo", s.replace("^", "").replace("=", "_") + ".json")
    get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s.replace('^', '%5E')}"
        f"?period1=0&period2={int(time.time())}&interval=1d", out)
    r = json.load(open(out))["chart"]["result"][0]
    print(s, len(r["timestamp"]), time.strftime("%Y-%m-%d", time.gmtime(r["timestamp"][0])))
    time.sleep(1)
for c, sid in FRED.items():
    out = os.path.join(RAW, "fred", c + ".csv")
    get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", out)
    lines = open(out).read().splitlines()
    print(c, len(lines) - 1, lines[1][:10], lines[-1])
