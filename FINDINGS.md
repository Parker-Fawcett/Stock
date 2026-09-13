# Findings

Living record of everything tested in this project. Updated after every
experiment. Numbers are net of stated costs; methods in README.
Rule: tuned windows never count. Only clean, holdout, or live results
move a verdict.

## Method rules (frozen)

- Walk-forward + purge/embargo, no shuffling val/test.
- Monkey/random baseline on every backtest. Mean over seeds, never best.
- Costs from day one. Survivorship bias disclosed, never hidden.
- Tune/holdout split before any selection. One report per question.
- Paper log is the final judge. Nothing here has earned real dollars.

## Data status

| source | status |
|---|---|
| Yahoo prices (all universes) | working, cached |
| SEC EDGAR / bulk statements | WAF-blocked from here |
| OpenInsider Form 4 (scrape) | working, 15k txns cached |
| FMP free key (shared demo) | partial: mega-cap annuals only, 250 calls/day, ~1 call/sec |
| French factor library | working, through Jul 2026 |
| FINRA / Nasdaq short data | reachable, heavy; not yet pulled |

## Experiment log

1. **Large-cap LSTM-style baseline → trees first.** Trees beat a year of
   LSTM work on identical data (his video + our replication). LSTM buried.
2. **Large-cap price-only ML (104 names, SPY-relative, 20d excess).**
   AUC 0.507–0.530, mostly cash. Null.
3. **+ raw Form 4 features.** AUC 0.530 → 0.524. Hurt. Buried.
4. **Small-cap price-only (100 names, IWM-relative).** AUC 0.550.
   Backtest beats monkeys 5/8 folds, fold CAGRs -75% to +123%.
   Signal with disqualifying variance.
5. **+ raw Form 4 on small caps.** AUC 0.550 → 0.533. Hurt. Buried.
6. **+ conviction Form 4 (C-suite, >$50k, first buy 12mo).**
   Tune-half 0.5554 vs 0.5587 price-only. Bar was +0.01. Buried.
7. **Risk overlays (breadth gate, stops, vol target, DD brake).**
   v1 concentrated into blowups; v2 into cash (2.6% CAGR). Hand-tuning
   risk knobs = overfitting. Stopped, switched to tune/holdout.
8. **Tune/holdout, 3 configs, small caps.** Tune (2018–22): everything
   loses to monkeys except cash. Holdout (2022–26): wide +20.8% — but
   holdout was eyeballed during construction. Contaminated, not claimed.
9. **Long-short (top-10/bottom-10, 5% borrow).** Tune Sharpe -0.03.
   Bar failed, no holdout. Hedge cuts DD (-0.26), short book earns nothing.
10. **Mid-cap ML generalization (fresh universe).** AUC 0.559 —
    replicates where overfitting cannot reach. P&L +4.2% / -0.44 DD.
    Signal real, harvest weak.
11. **Textbook 12-1 momentum, no ML.** Small: tune +9.5% (Sharpe 0.41),
    holdout +16.3% (Sharpe 0.20). Mid: tune +3.3%, holdout +15.1%.
    Positive every half, both universes. Best result in the project.
12. **Value lane, real data (12 mega caps, FMP annuals).** Median
    price/IV 2.77 — mega caps never near Graham value. d0.5: zero trades.
    d0.0: +2.7% vs +8.1% fair benchmark, avg 0.4 names. Free FMP caps
    history at 5 years and covers mega caps only: no crash in window, no
    small caps, growth from 5 points. Weak test of value generally;
    decisive for "Graham on mega caps 2021-26": cash, and trails.

## Live paper log (frozen models)

- `wide-seed0-v1`: 30 picks, 37% hit, -5.2% mean, -2.7% excess vs IWM.
- `mom12-1-v1`: logging starts next month-end. Backtest says +9–16%.

## Factor regimes 2015–2026 (French)

Alive: market +11.3%, profitability +2.4%, momentum +2.0%.
Dead: size -2.3%, value -1.1%, investment -1.1%.
Explains our small-cap pain (size headwind) and value failure.

## Buried

LSTM for tabular data, 1-day targets, raw insider counts, conviction
insider (failed bar), hand-tuned risk knobs, long-short small caps,
WSB/4chan sentiment (both directions), famous macro (VIX/rates/QE),
FFT features, MACD/candlesticks.

## Open

- Value backtest on FMP mega-caps (quota-gated).
- Quality/profitability tilt (live factor, working endpoints).
- Paper log accumulation (months, no peeking).

## Bottom line (current)

Public-data cross-sectional rank predicts weakly (AUC ~0.56) and profits
~never after risk. The only strategy positive in every half tested is
textbook momentum with no ML. Same verdict as LosingLoonies, tighter
bounds, three universes.
