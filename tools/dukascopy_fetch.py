"""Download Dukascopy bid/ask candles for FX majors and index CFDs.

Run this on a network that does NOT inspect TLS (home Wi-Fi or a phone hotspot;
the MBZUAI network blocks it).  Standard library only, safe to stop and re-run:
finished months/days are skipped.

    python3 tools/dukascopy_fetch.py            # hourly bars 2015->now  (~3,000 requests, ~10 min)
    python3 tools/dukascopy_fetch.py --minute   # 1-minute bars 2015->now (~70,000 requests, a few hours)

Output: data/raw/dukascopy/<tf>/<SYMBOL>_<BID|ASK>.csv with columns
time_utc,open,high,low,close,volume.  Prices are divided by POINT below; I check
the scale against Yahoo before using the data.
"""
import argparse, csv, datetime as dt, lzma, os, struct, time, urllib.request

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "USA500IDXUSD", "USATECHIDXUSD", "DEUIDXEUR", "GBRIDXGBP"]
POINT = {"USDJPY": 1e3}                     # FX default 1e5; indices default 1e3
URL = "https://datafeed.dukascopy.com/datafeed"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "dukascopy")


def point(sym):
    return POINT.get(sym, 1e3 if "IDX" in sym else 1e5)


PAUSE = 1.0     # seconds between requests; Dukascopy answers 429 (too many requests) if you go faster


def fetch(url, tries=8):
    """GET with a polite pause; on 429/5xx wait 30 s, 60 s, ... up to ~10 min before giving up."""
    err = None
    for k in range(tries):
        time.sleep(PAUSE)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            err = e
        except Exception as e:
            err = e
        wait = min(30 * 2 ** k, 600)
        print(f"  {err!r}; waiting {wait} s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"failed: {url} ({err!r}) -- if this is an SSL error you are on a network that blocks Dukascopy")


def candles(blob, t0, sym):
    """Each record: uint32 seconds-from-t0, open, close, low, high (uint32 points), float32 volume."""
    if not blob:
        return []
    b = lzma.decompress(blob); p = point(sym); rows = []
    for i in range(0, len(b) - 23, 24):
        s, o, c, lo, hi, v = struct.unpack(">5If", b[i:i + 24])
        if v > 0:                                   # zero-volume candles are filler
            rows.append(((t0 + dt.timedelta(seconds=s)).strftime("%Y-%m-%d %H:%M"), o / p, hi / p, lo / p, c / p, round(v, 4)))
    return rows


def units(minute, start):
    """(url suffix, start time) for every month (hourly) or day (minute) up to yesterday."""
    d = start; end = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=1)
    while d < end:
        if minute:
            if d.weekday() != 5:                    # no trading on Saturdays
                yield f"{d.year}/{d.month - 1:02d}/{d.day:02d}/%s_candles_min_1.bi5", d
            d += dt.timedelta(days=1)
        else:
            yield f"{d.year}/{d.month - 1:02d}/%s_candles_hour_1.bi5", d
            d = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minute", action="store_true")
    ap.add_argument("--start", default="2015-01-01")
    a = ap.parse_args()
    tf = "m1" if a.minute else "h1"
    os.makedirs(os.path.join(OUT, tf), exist_ok=True)
    start = dt.datetime.fromisoformat(a.start)
    for sym in SYMBOLS:
        for side in ("BID", "ASK"):
            path = os.path.join(OUT, tf, f"{sym}_{side}.csv")
            done = set()
            if os.path.exists(path):
                with open(path) as f:
                    done = {r[0][:10 if a.minute else 7] for r in csv.reader(f) if r and r[0][0].isdigit()}
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time_utc", "open", "high", "low", "close", "volume"])
                n = 0
                for suffix, t0 in units(a.minute, start):
                    key = t0.strftime("%Y-%m-%d" if a.minute else "%Y-%m")
                    if key in done:
                        continue
                    w.writerows(candles(fetch(f"{URL}/{sym}/{suffix % side}"), t0, sym)); n += 1
                    if n % 50 == 0:
                        f.flush(); print(f"  {sym} {side} {key}", flush=True)
            print(f"{sym} {side} done -> {path}", flush=True)


if __name__ == "__main__":
    main()
