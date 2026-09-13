# The lab that publishes red ink

Retail quant sells winners. This repo audits losers — with methodology
intact. Every strategy here runs through monkeys, sealed holdouts,
costs from day one, and a trial budget; every failure gets a death
record in FINDINGS.md. Two frozen models trade on paper monthly and get
graded in public, green or red.

Live scoreboard (see `paper.py grade`, `data/paper/log.csv):
- `wide-seed0-v1` (ML rank): 30 picks, 37% hit, -2.7% excess vs IWM.
- `mom12-1-v1` (textbook momentum): logging, backtest says +6–11% halves.

Origin: independent rebuild of a community-documented AI stock predictor,
starting from 29 published video transcripts (`sources-channel-predictor/`
and `sources-channel-strategy/`). Every design choice maps to a failure
the source project documented across a year of public iteration — then
we kept score honestly where it stopped. Source channel credited in full;
all code, tests, and findings here are original work.

## What changed vs his V1

| His failure | This rebuild |
|---|---|
| 1-day direction, MSE learns the mean | 20-day forward **excess vs SPY**, cross-sectional top-20% label |
| LSTM on a flat table | Tree baseline first (`model_lgbm.py`, LightGBM if present else sklearn) |
| Stock-split + overlapping labels leak | Walk-forward + purge + embargo (`model_lgbm.walk_forward`) |
| Forced daily buy | Cash default, max 10 names, proba gate (`backtest.py`) |
| No costs, impossible fills | `COST_BPS` on every change, monthly closes |
| Single backtest luck | Monkey baseline + mean over seeds (`run_monkeys`) |
| Survivors-only list | **Not fixed here** — free Stooq has the same bias. See below. |

## The bias warning (read this)

His finale (`noK0IwZAnyE`): alphabetical survivors-only list made the
model look like a genius. This scaffold uses free Stooq data with the
**same flaw**. Any CAGR here is a plumbing check. For real numbers you
need point-in-time membership + delisted prices (Massive/CRSP/Quiver).

## Run

```
pip install -r requirements.txt
python3 run.py --tickers AAPL,MSFT,JNJ,PG,XOM,CVX,KO,MRK,WMT,IBM --start 20150101
```

## Files

- `predictor/data.py` — Yahoo downloader with survivorship warning
- `predictor/features.py` — trailing-only features, 20d excess labels
- `predictor/model_lgbm.py` — walk-forward folds, purge/embargo, calibration
- `predictor/backtest.py` — monthly rebalance, costs, monkey baseline,
  risk configs (base / wide / v2)
- `predictor/evaluate.py` — AUC + CAGR/Sharpe/maxDD/turnover/cost sweep
- `predictor/insider.py` — Form 4 features via OpenInsider (rejected twice)
- `predictor/universe.py` — SP100 + S&P 600 sample
- `run.py` — backtests, `--save-proba` for sweeps
- `sweep.py` — cost sweeps with tune/holdout splits
- `paper.py` — FROZEN forward paper trading (wide-seed0-v1)

## Paper trading (the only test that matters now)

```
# monthly, after month-end close:
python3 paper.py predict --universe smallcap --market IWM --refresh
# anytime (scores picks 20 trading days old vs IWM):
python3 paper.py grade
```

Picks append to `data/paper/log.csv` with model version. No tuning on
results — the market grades, we read.

## Insider verdict (buried)

Raw Form 4 counts hurt twice (large-cap 0.530 to 0.524, small-cap 0.550
to 0.533). Conviction-filtered (C-suite, >$50k, first buy in 12mo, 458
conviction-months) scored 0.5554 vs 0.5587 price-only on the clean
tune-half — pre-committed bar was +0.01. Failed. Insider is out.

## Mid-cap generalization (untouched universe, reported once)
S&P 400 sample, MDY benchmark, frozen wide config, 25bps, 26 folds:
AUC 0.559, CAGR +4.2%, maxDD -0.44, monkeys 1.04. Rank signal replicates
on data that could not have been tuned to; P&L does not replicate at
small-cap magnitude. Weak signal, unharvestable risk — same verdict.

## Research wave: factors + textbook momentum

French 2015-2026: RMW (profitability) +2.4%/yr and MOM +2.0%/yr alive;
SMB/HML/CMA all negative. Our small-cap pain = SMB headwind; value
failure = HML headwind. 4chan sentiment dead both directions (his newest
video agrees). FINRA/Nasdaq short data reachable but heavy.
Textbook 12-1 momentum, small caps, 25bps, 188 months, no ML, no tuning:
tune +9.5% (Sharpe 0.41, DD -0.31), holdout +16.3% (Sharpe 0.20, DD
-0.56). Beats every ML config in both halves. Mid-cap replication:
tune +3.3% (Sharpe 0.15), holdout +15.1% (Sharpe 0.35) — positive both
halves, both universes. The model never beat the textbook. Momentum
sleeve now logs alongside ML in paper.py (mom12-1-v1) with per-model
grading.

## Value lane (in progress)

`value/`: Graham IV (his formula, hand-validated like his ABBV video:
`python3 -m value.test_value`), monthly buy-at-discount/sell-at-IV/3yr
backtest with his fair-benchmark fix, tune/holdout from minute one.
Regime context: French HML 2015-2026 ran -1.1%/yr — headwind quantified.
Blocked on real quarterly history: set FMP_KEY (free tier), the loader
is a one-line plug. SEC bulk EDGAR is WAF-blocked from here.
