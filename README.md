# The lab that publishes red ink

Retail quant sells winners. This repo audits losers — with methodology
intact. Every strategy here runs through monkeys, sealed holdouts,
costs from day one, and a trial budget; every failure gets a death
record in FINDINGS.md. Two frozen models enter a provenance-locked monthly
paper series and get graded in public, green or red.

**[Current results and evidence status](RESULTS.md)** — the short,
maintained view. `FINDINGS.md` remains the full chronological record.

**[Working paper](PAPER.md)** — manuscript draft centered on the controlled
finding that near-identical aggregate AUC can conceal unstable portfolio
selection. Open statistical and matched-universe work is declared in the draft.

The 30 rows in `data/paper/log.csv` are retained as historical records, not
forward evidence. The clean `prospective-v2` scoreboard starts at the next
month end with `wide-seed0-v1` and `mom12-1-v1`.

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
| Survivors-only list | Local runs retain the bias; the QuantConnect replication adds a point-in-time check. |

## The bias warning (read this)

His finale (`noK0IwZAnyE`): alphabetical survivors-only list made the
model look like a genius. This scaffold uses free Stooq data with the
**same flaw**. The guarded QuantConnect 252/21 replication remained strongly
positive with point-in-time membership and delisting-aware data (+3,455.56%
total return, 53.70% max drawdown, 0.695 Sharpe), which supports the direction
of the finding. Its dynamic top-200 liquid universe is different from these
small- and mid-cap lists, so local CAGR magnitudes remain provisional.

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
- `run.py` — backtests; new probability caches include an input/code manifest
- `cache_compare.py` — row-level probability, rank, and selected-name comparison
- `selection_stability.py` — monthly overlap and distance-to-cutoff figure
- `paper_provenance.py` — verify every paper table, figure, input, and environment lock
- `purge_ablation.py` — controlled legacy-vs-corrected purge refit
- `sweep.py` — cost sweeps with tune/holdout splits
- `paper.py` — provenance-locked prospective ML and momentum ledger

Verify the paper's displayed results and their locked inputs with
`python3 paper_provenance.py verify`.

## Prospective paper trading (the only test that matters now)

```
# monthly, after month-end close:
python3 paper.py predict --universe smallcap --market IWM --refresh
# anytime (scores picks 20 trading days old vs IWM):
python3 paper.py grade --refresh
```

The clean `prospective-v2` series begins with the next month-end run. Each run
records its UTC creation time, code commit, input hashes, model backend,
train/validation dates, parameters, and signals in a verified manifest. Picks
enter at the next session's open and are graded after 20 sessions with actual
close-based stop exits and costs. Off-cycle runs and modified pipeline code are
rejected. The old `data/paper/log.csv` remains historical and separate. No
tuning on results — the market grades, we read.

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
Textbook 12-1 momentum, small caps, 25bps, 188 months, no ML, no tuning,
under the corrected common accounting convention: tune +17.8% (Sharpe
0.84, DD -0.26), holdout +21.7% (Sharpe 0.78, DD -0.37). Mid-cap
replication: tune +9.8% (Sharpe 0.54), holdout +24.7% (Sharpe 0.81).
These Yahoo universes contain today's survivors, so the magnitude remains
provisional. A point-in-time QuantConnect run confirms that momentum survives
on a different liquid-equity universe. A fixed-universe cloud run then produced
a 0.933 correlation with the local monthly return path; on common month-end
sampling its Sharpe/DD are 0.93/-39.3% versus 0.83/-37.1% locally. The model
never beat the textbook. Momentum logs alongside ML in
paper.py (`mom12-1-v1`) with per-model grading.

## Value lane (in progress)

`value/`: Graham IV (his formula, hand-validated like his ABBV video:
`python3 -m value.test_value`), monthly buy-at-discount/sell-at-IV/3yr
backtest with his fair-benchmark fix, tune/holdout from minute one.
Regime context: French HML 2015-2026 ran -1.1%/yr — headwind quantified.
Blocked on real quarterly history: set FMP_KEY (free tier), the loader
is a one-line plug. SEC bulk EDGAR is WAF-blocked from here.
