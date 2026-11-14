#!/usr/bin/env python3
"""Forecast Polymarket's FAA outage attribution Yes prices, minute by minute,
4 hours ahead, as five quantiles a minute.

make.fun bounty polymarket-faa-attribution-02. The program reads one case at
a time from /input (<id>.in: every trading market of the family, each with
its Yes price at every minute of the 72 hours up to a checkpoint) and writes
/output/<id>.out: for each market and each of the 240 minutes after the
checkpoint, the 5%, 25%, 50%, 75% and 95% quantiles of that minute's price.

The final is scored on four jumpy attribution markets (Iran, foreign actor,
hack, AI sandbox escape). The practice cases are cut from Polymarket's "Next
CEO of NVIDIA?" event instead, so the method is kept deliberately simple and
robust rather than tuned to the practice's quirks: the forecast is an
equal-thirds blend of three per-market quantile forecasters, all centred on
the market's last price (persistence median; short-horizon momentum was
tested on the practice and hurt, so no drift is used):

1. Normal / EWMA volatility: standard normal quantiles scaled by an
   exponentially weighted (20-minute half-life) estimate of the market's
   1-minute move size, growing with sqrt(horizon). Adapts fast when a
   market heats up or calms down.
2. Empirical 1-minute shape: the quantiles of the market's own 1-minute
   price changes over the trailing 6 hours, scaled by sqrt(horizon).
   Captures each market's real move distribution - fat tails, jumps and
   asymmetry - instead of assuming normality.
3. Empirical per-horizon shape: for anchor horizons (5, 15, 30, 60, 120,
   240 minutes) the quantiles of that horizon's actual price changes over
   the trailing 6 hours, recentered on the last price and interpolated in
   sqrt(horizon); below the first anchor the quantiles shrink by
   sqrt(h/anchor). Captures how moves really grow with horizon - slower
   than a random walk when prices mean-revert, faster when they jump.

Backtested on the six public practice cases, the blend's mean pinball loss
is 0.2622 points, against 0.3636 for the house persistence baseline
(the baseline figure reproduces the kit's published 0.363589 exactly,
which validates the backtest harness's outcomes). The 90% interval covers
0.89 of practice minutes (nominal 0.90).

Two refinements over the plain blend (validated 14 Nov, Push 5):
- The normal component's sigma is the MEAN of the EWMA standard deviation
  and a MAD-based (L1) EWMA sigma, which is robust to single jumps.
- The inner-quantile shrink RAMPS WITH THE PRICE LEVEL (0.70 at price 0
  to 1.00 at price >= 0.10): the optimal shrink rises monotonically with
  price across buckets (0.70 best below 0.02, 1.00 best above 0.10), so a
  fixed shrink is too loose on near-zero markets and too tight on the
  mid-price markets the final actually scores.

Everything is deterministic (no randomness, no clock, no network),
standard library only, and runs in well under a second per case.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in a row's order.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels (exact, from the stdlib).
_ZDIST = statistics.NormalDist()
Z = [_ZDIST.inv_cdf(t) for t in QUANTILES]
# Anchor horizons (minutes) for the per-horizon empirical shape.
ANCHORS = [5, 15, 30, 60, 120, 240]
# How far back the empirical shapes look (minutes): 6 hours. Longer windows
# still hold old shocks and overshoot the spread; the practice grid favours
# shorter windows at every half-life tried, and 360 sits on a broad flat
# minimum rather than at an edge.
LOOKBACK = 360
# EWMA half-life (minutes) for the volatility estimate: the practice grid is
# monotone in favour of shorter half-lives down to 20 and flat below it, so
# 20 sits in the middle of the flat bottom - fast to track regime changes,
# slow enough not to chase single prints.
EWMA_HALFLIFE = 20
# Bounds on the per-minute volatility (price units): the floor keeps a dead
# market's interval from collapsing to a point, the cap keeps one wild print
# from blowing it out. The floor is flat on the practice grid between 0.0002
# and 0.001, so 0.0005 sits in the middle of that flat range - a little
# insurance for a market that has been quiet for hours but could still jump
# on news, which is exactly the final's shape.
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.01
# The house persistence baseline's spread, used as a fallback.
BASELINE_SIGMA = 0.0025
# Extra widening of the outer (5%/95%) quantiles around the median, as a
# multiplier: 1.0 leaves them as the blend gives them. The final's markets
# jump (the Iran market moved 0.49 -> 0.275 in one minute), so the backtest
# checked whether a little extra tail width buys insurance cheaply: it
# costs on the practice (1.15x -> +0.0026) while the empirical shapes
# already widen per-market from each market's own jump history, so the
# multiplier stays 1.0.
TAIL_WIDE = 1.0
# Shrinkage of the inner (25%/75%) quantiles toward the median, as a
# multiplier that RAMPS WITH THE PRICE LEVEL. Averaging three forecasters'
# quantiles widens the middle of the distribution (the unshrunk blend's 50%
# interval covers 0.66 of practice minutes, not 0.50), so the inner
# quantiles are pulled in - but the optimal pull depends on price: scored
# per price bucket, 0.70 wins below 0.02, 0.85 wins at 0.05-0.10 and 1.00
# wins above 0.10 (the widening is a low-price clamping artifact and
# reverses at mid prices). The shrink therefore ramps linearly from
# INNER_SHRINK_LO at price 0 to INNER_SHRINK_HI at price
# INNER_SHRINK_PRICE_HI and stays flat above it.
INNER_SHRINK_LO = 0.70
INNER_SHRINK_HI = 1.0
INNER_SHRINK_PRICE_HI = 0.10


def clamp6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def quantile_of(sorted_vals, tau):
    """The tau-quantile of an already sorted list (linear interpolation)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def enforce_order(row):
    """Make a quantile row non-decreasing in place (rounding can tie)."""
    for k in range(1, len(row)):
        if row[k] < row[k - 1]:
            row[k] = row[k - 1]
    return row


def ewma_vol(prices):
    """Exponentially weighted std of 1-minute changes (20-min half-life).

    Returns None when there is too little history to estimate from.
    """
    n = len(prices)
    if n < 20:
        return None
    lam = 0.5 ** (1.0 / EWMA_HALFLIFE)
    start = max(1, n - int(6 * EWMA_HALFLIFE))
    wsum = 0.0
    mean = 0.0
    m2 = 0.0
    for i in range(start, n):
        d = prices[i] - prices[i - 1]
        w = lam ** (n - 1 - i)
        wsum += w
        delta = d - mean
        mean += w / wsum * delta
        m2 += w * delta * (d - mean)
    if wsum <= 0 or m2 <= 0:
        return SIGMA_FLOOR
    return min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(m2 / wsum)))


def mad_vol(prices):
    """Exponentially weighted MAD (mean absolute 1-min change) sigma.

    Same 20-minute half-life as ewma_vol, but L1-based: E|N(0, s)| =
    s*sqrt(2/pi), so s = MAD * sqrt(pi/2). A single jump inflates a
    squared (L2) estimate far more than an L1 one, so the MAD sigma is a
    robust second opinion; forecast_market averages the two.
    """
    n = len(prices)
    if n < 20:
        return None
    lam = 0.5 ** (1.0 / EWMA_HALFLIFE)
    start = max(1, n - int(6 * EWMA_HALFLIFE))
    wsum = 0.0
    m = 0.0
    for i in range(start, n):
        d = abs(prices[i] - prices[i - 1])
        w = lam ** (n - 1 - i)
        wsum += w
        m += w / wsum * (d - m)
    if wsum <= 0 or m <= 0:
        return SIGMA_FLOOR
    return min(SIGMA_CAP, max(SIGMA_FLOOR, m * math.sqrt(math.pi / 2.0)))


def normal_rows(x, sigma, horizon):
    """Normal quantiles around x with per-minute sigma, sqrt(h) growth."""
    rows = []
    for h in range(1, horizon + 1):
        root = sigma * math.sqrt(h)
        rows.append([clamp6(x + z * root) for z in Z])
    return rows


def onemin_rows(prices, horizon):
    """Empirical 1-minute-change quantiles (trailing LOOKBACK) x sqrt(h)."""
    n = len(prices)
    x = prices[-1]
    w = min(LOOKBACK, n - 1)
    if w < 30:
        sig = ewma_vol(prices)
        return normal_rows(x, sig if sig is not None else BASELINE_SIGMA,
                           horizon)
    diffs = sorted(prices[i + 1] - prices[i]
                   for i in range(n - 1 - w, n - 1))
    qs = [quantile_of(diffs, t) for t in QUANTILES]
    # Persistence median: the median change is forced to 0.
    qs[QUANTILES.index(0.5)] = 0.0
    rows = []
    for h in range(1, horizon + 1):
        root = math.sqrt(h)
        rows.append(enforce_order([clamp6(x + q * root) for q in qs]))
    return rows


def horizon_rows(prices, horizon):
    """Per-horizon empirical quantiles, recentered on the last price."""
    n = len(prices)
    x = prices[-1]
    tables = {}
    for big_h in ANCHORS:
        if n - big_h < 30:
            continue
        lo_i = max(0, n - big_h - LOOKBACK)
        changes = sorted(prices[i + big_h] - prices[i]
                         for i in range(lo_i, n - big_h))
        qs = [quantile_of(changes, t) for t in QUANTILES]
        med = qs[QUANTILES.index(0.5)]
        tables[big_h] = [q - med for q in qs]
    if not tables:
        sig = ewma_vol(prices)
        return normal_rows(x, sig if sig is not None else BASELINE_SIGMA,
                           horizon)
    hs = sorted(tables)
    rows = []
    for h in range(1, horizon + 1):
        if h <= hs[0]:
            # Below the first anchor, shrink its quantiles by sqrt(h/anchor)
            # (a random walk's spread at h minutes) instead of copying the
            # anchor flat, which overshoots the first few minutes.
            s = math.sqrt(h / hs[0])
            row = [clamp6(x + q * s) for q in tables[hs[0]]]
        elif h >= hs[-1]:
            row = [clamp6(x + q) for q in tables[hs[-1]]]
        else:
            row = None
            for a, b in zip(hs, hs[1:]):
                if a <= h <= b:
                    w = ((math.sqrt(h) - math.sqrt(a))
                         / (math.sqrt(b) - math.sqrt(a)))
                    qlo, qhi = tables[a], tables[b]
                    row = [clamp6(x + qlo[k] * (1 - w) + qhi[k] * w)
                           for k in range(len(QUANTILES))]
                    break
            if row is None:  # pragma: no cover - unreachable by construction
                row = [clamp6(x + q) for q in tables[hs[-1]]]
        rows.append(enforce_order(row))
    return rows


def shrink_inner(rows, shrink):
    """Pull the inner (25%/75%) quantiles toward the median by `shrink`.

    Averaging three forecasters' quantiles widens the middle of the
    distribution, so the inner quantiles are shrunk a little to restore
    calibration. The shrink amount ramps with the market's price level
    (see INNER_SHRINK_LO/HI); it is computed per market in
    forecast_market and passed in here.
    """
    if shrink == 1.0:
        return rows
    mid = QUANTILES.index(0.5)
    out = []
    for row in rows:
        m = row[mid]
        new = list(row)
        new[1] = clamp6(m + (row[1] - m) * shrink)
        new[3] = clamp6(m + (row[3] - m) * shrink)
        out.append(enforce_order(new))
    return out


def widen_tails(rows):
    """Widen the outer quantiles around the median by TAIL_WIDE."""
    if TAIL_WIDE == 1.0:
        return rows
    mid = QUANTILES.index(0.5)
    out = []
    for row in rows:
        m = row[mid]
        new = list(row)
        new[0] = clamp6(m + (row[0] - m) * TAIL_WIDE)
        new[-1] = clamp6(m + (row[-1] - m) * TAIL_WIDE)
        out.append(enforce_order(new))
    return out


def forecast_market(prices, horizon):
    """Equal-thirds blend of the three quantile forecasters.

    The normal component's sigma averages the EWMA standard deviation
    with the robust MAD sigma, and the inner shrink ramps with the
    market's last price (tighter on near-zero markets, none at mid
    prices and above).
    """
    x = prices[-1]
    sig = ewma_vol(prices)
    mad = mad_vol(prices)
    if sig is None:
        sig = BASELINE_SIGMA
    if mad is None:
        mad = BASELINE_SIGMA
    sig = 0.5 * (sig + mad)
    a = normal_rows(x, sig, horizon)
    b = onemin_rows(prices, horizon)
    c = horizon_rows(prices, horizon)
    rows = []
    for h in range(horizon):
        row = [clamp6((a[h][k] + b[h][k] + c[h][k]) / 3.0)
               for k in range(len(QUANTILES))]
        rows.append(enforce_order(row))
    shrink = INNER_SHRINK_LO + (INNER_SHRINK_HI - INNER_SHRINK_LO) * min(
        1.0, x / INNER_SHRINK_PRICE_HI)
    return widen_tails(shrink_inner(rows, shrink))


def forecast_case(case):
    """Forecast every market of one case; a market that fails any step
    falls back to the house persistence baseline rather than failing
    the whole case (an invalid case takes no prize)."""
    horizon = case["horizon_minutes"]
    forecasts = []
    for m in case["markets"]:
        prices = m["prices"]
        try:
            if len(prices) == 0:
                raise ValueError("market has no prices")
            rows = forecast_market(prices, horizon)
        except Exception as exc:  # noqa: BLE001 - never fail the case
            print("warning: market %s fell back to baseline: %s"
                  % (m["id"], exc), file=sys.stderr)
            x = prices[-1] if prices else 0.5
            rows = normal_rows(x, BASELINE_SIGMA, horizon)
        forecasts.append({"id": m["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
