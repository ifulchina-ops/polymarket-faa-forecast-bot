# Polymarket FAA outage attribution forecast bot

Entry for the make.fun bounty
**polymarket-faa-attribution-02**: forecast Polymarket's FAA outage
attribution Yes prices minute by minute, 4 hours ahead, as five quantiles
(5%, 25%, 50%, 75%, 95%) a minute.

## The task

`main.py` reads one case at a time from `/input` (`<id>.in`: every trading
market of the family, each with its Yes price at every minute of the 72
hours up to a checkpoint) and writes `/output/<id>.out`: for each market
and each of the 240 minutes after the checkpoint, five quantiles of that
minute's price. It runs offline in the sandbox (no network, standard
library only), deterministically, in well under a second per case.

## Method

The final is scored on four jumpy attribution markets (Iran, foreign
actor, hack, AI sandbox escape). The practice cases are cut from
Polymarket's "Next CEO of NVIDIA?" event instead, so the method is kept
deliberately simple and robust rather than tuned to the practice's quirks:
an equal-thirds blend of three per-market quantile forecasters, all
centred on the market's last price (persistence median; short-horizon
momentum was tested on the practice and hurt, so no drift is used):

1. **Normal / EWMA volatility**: standard normal quantiles scaled by an
   exponentially weighted (20-minute half-life) estimate of the market's
   1-minute move size, growing with sqrt(horizon). Adapts fast when a
   market heats up or calms down.
2. **Empirical 1-minute shape**: the quantiles of the market's own
   1-minute price changes over the trailing 6 hours, scaled by
   sqrt(horizon). Captures each market's real move distribution - fat
   tails, jumps and asymmetry - instead of assuming normality.
3. **Empirical per-horizon shape**: for anchor horizons (5, 15, 30, 60,
   120, 240 minutes) the quantiles of that horizon's actual price changes
   over the trailing 6 hours, recentered on the last price and
   interpolated in sqrt(horizon); below the first anchor the quantiles
   shrink by sqrt(h/anchor). Captures how moves really grow with horizon.

Two small calibration steps on the blend: the inner (25%/75%) quantiles
are pulled 15% toward the median (averaging three forecasters widens the
middle of the distribution), and the outer quantiles are left as the blend
gives them (extra tail width was tested and cost more than it bought -
the empirical shapes already widen per-market from each market's own jump
history). A market that fails any step falls back to the house persistence
baseline rather than failing the whole case.

## Backtest

Scored on the six public practice cases against outcomes derived from the
next case file (p00-p04) and Polymarket's CLOB price history (p05):

| forecaster | mean pinball loss |
|---|---|
| House baseline (sigma 0.0025) | 0.3636 |
| This entry (blend) | **0.2638** |

The baseline figure reproduces the kit's published 0.363589 exactly,
which validates the backtest's outcomes. The blend beats the baseline on
every practice case (0.473 / 0.300 / 0.330 / 0.240 / 0.197 / 0.106), and
its 90% interval covers 0.89 of practice minutes (nominal 0.90).

Sensitivity, all on the same six cases: EWMA half-life 60 instead of 20
gives 0.2740; a 24-hour lookback instead of 6 hours gives 0.2710; tail
widening 1.15x gives 0.2678; the volatility floor is flat between 0.0002
and 0.001, so 0.0005 sits in the middle of that range. The chosen
parameters sit on broad flat minima, not knife edges.

A live sanity check on the four real FAA markets (built from CLOB
histories) passes: every median equals the last live price, intervals
are market-specific (the hack market, which jumped most, gets the widest
intervals) and widen with horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT. See LICENSE.
