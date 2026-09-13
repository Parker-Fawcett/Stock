# Repository review — September 13, 2026

## Assessment

This is a research lab built around replicating and extending LosingLoonies' stock experiments. It contains substantial working scaffolding and an unusually explicit record of rejected ideas. However, the implemented evaluation does not yet support the confidence expressed in FINDINGS.md. Neither “ML has a proven weak edge” nor “the index definitively wins” follows securely from the current accounting and validation.

This review examined the project notes, Python implementation, experiment registry, paper log, cached price/fold data, and git history. Source transcripts supply background; their external research claims were not independently fact-checked. No strategies were retuned, no paper picks appended, and no historical results overwritten.

## What is here

- `sources-channel-*`: 29 video transcripts: 11 predictor, 14 strategy, four recent.
- `predictor/`: adjusted Yahoo price downloads, hard-coded stock universes, price/volume features, optional insider features, tree models, validation, backtests and metrics.
- `run.py`, `sweep.py`: model fitting and cached-probability experiments. AUC spans seeds; portfolio results use seed zero only.
- `mom_run.py`, `multi_run.py`, `ens_test.py`: stock momentum, ETF trend, and ML/momentum blending.
- `improve.py`: a human-authored proposal registry with automated scoring and promotion. It does not independently generate or rewrite strategies.
- `value/`, `value_run.py`: Graham valuation and a separate portfolio simulator.
- `paper.py`: manually invoked pick logging/grading. No scheduler is implemented in this repository.
- `qc_momentum.py`: an unvalidated QuantConnect migration draft, not evidence of a completed independent replication.
- `data/`: about 102 MB of raw prices, 49 MB of cached model folds, insider/fundamental data, trial registry and paper log. Prices and model caches are ignored by git; rebuilding them later need not reproduce the original data snapshot.

## What the record actually says

The persisted registry is the clearest source for trial status: **12 of 20 trials spent; eight remain**, with four promotions. Recorded holdout CAGR/maximum drawdown:

| Proposal | CAGR | Maximum drawdown |
|---|---:|---:|
| P5 lower probability gate | 12.68% | -35.0% |
| P8 inverse-volatility weights | 6.14% | -39.0% |
| P9 wider portfolio | 4.31% | -40.1% |
| P12 widest portfolio | 3.54% | -40.8% |

These are stored historical outputs, not validated performance estimates. FINDINGS.md also contains obsolete three-promotion/ten-trials-left summaries. README retains superseded momentum CAGRs. The paper CSV contains 30 ML picks on just two dates, December 2, 2025 and August 12, 2026, and **zero momentum picks**. Its first git record is September 13, 2026; no contemporaneous creation timestamp establishes those picks as forward-live observations.

## Verified problems, in priority order

### 1. P1's spectacular return has a direct implementation explanation

`improve.py:122` passes `stop_frac=2.0` to mean “stops off.” In `predictor/backtest.py`, any future price below twice entry triggers a credited return of `log(2)`. A constant $100 price series therefore generates a 100% gross gain in each tested month with future observations. This falsifies the notes' explanation that the output can be attributed to genuine biotech/crypto doubles. The exact contribution to the archived result requires a corrected replay.

### 2. Walk-forward validation leaks future outcomes

Labels cover 20 trading observations, but `features.py:101` assigns a calendar-day label end and `model_lgbm.py:53` purges only 20 calendar days. Validation rows are not purged before the test period, although validation labels train the calibrator or select boosting iterations. `embargo_days` is unused.

Cached-data check: one fold's last validation date is September 10, 2013, its test starts September 11, and that validation label requires prices through October 8. The final training label also extends past validation start. Thus the claimed clean separation is not implemented. Cached probabilities need regeneration after fixing this.

### 3. Paper inference uses an old date, and momentum receives the wrong timestamp

`build_panel` removes rows without known forward returns. `paper.py` then predicts on this labeled panel's latest date. Verified: IWM prices end September 10, 2026, but the panel ends August 12. The momentum path uses the newest raw data and prices, then logs them under that older ML date. Grading those records would mix incompatible dates.

Grading also uses next-day close through the twentieth subsequent close, rather than the documented next-open entry; assumes exact stop fills; omits costs; and maps model identity by date/ticker, so matching picks across sleeves overwrite attribution. The current scoreboard is not a reliable forward portfolio track record.

### 4. Portfolio accounting needs a common replacement

- Several backtests average stock log returns then exponentiate. An equal-weight portfolio requires weighted simple returns: a +100% stock and a -50% stock produce +25%, while this calculation produces zero.
- Fixed 20/21-day windows starting at month-end can overlap or leave gaps. Fold-end windows truncate; boundary months can produce multiple rows that are annualized as separate months.
- Stops credit exactly the threshold even when observed prices gap below it. The long-short runner also assigns a positive return when a short stop is hit.
- “Vol targeting” and the drawdown brake reduce the number of names, but remaining names still receive the entire invested portfolio. That changes concentration rather than proportionally reducing exposure.
- Turnover uses ticker membership, missing weight changes and some stop exit/re-entry costs. Monkeys use a different construction and cost approximation.
- Drawdown omits starting equity, potentially missing an initial loss.

These issues affect both favorable and unfavorable conclusions; their combined impact cannot be inferred without rerunning.

### 5. The value engine does not implement its stated diversification

The buy loop gives the first qualifying stock all available cash. A controlled two-stock example with identical prices and valuations holds just one name. Results depend on ticker iteration order. `value_run.py` also evaluates holdout slices without rebasing strategy or benchmark equity, retaining the class of CAGR bug already corrected elsewhere.

### 6. Momentum is not one frozen specification across implementations

The standalone backtest uses positions -252 and -21; paper and QC use -273 and -22. Paper grading adds stops absent from the standalone momentum backtest. The QC universe changes to 200 liquid equities rather than the local small-cap sample. Its coarse filter returns coarse objects rather than selecting their symbols, which needs runtime/API validation. This is not yet a like-for-like replication.

### 7. Research governance is stronger in prose than in enforcement

The same holdout is consulted across proposal rounds; later variants cannot each establish a wholly untouched research test. The registry has no data/code fingerprint or immutable preregistration timestamp. An evaluation failure after some folds can still score the completed subset. “Loop frozen” is a note, not an enforced execution state. P3 remains eligible for promotion despite prose calling it dead. The vintage guard checks relative file endpoints, not absolute freshness, and is not wired into every runner as claimed.

## Recommended sequence

1. Preserve existing results as historical, unvalidated outputs and reconcile documentation against the registry.
2. Fix actual label endpoints and training/validation/test separation; separate inference features from labeled training rows.
3. Establish one portfolio ledger with explicit entry/exit dates, executable fills, cash, simple returns, weight-based costs and complete holding periods.
4. Make local momentum and the migration use the same specification and test their agreement before changing the universe.
5. Rebuild probabilities and replay existing strategies as a correctness exercise, not new strategy selection. Previously inspected history is development evidence now.
6. Start genuinely prospective paper logging with creation timestamps and locked model/data versions, then accumulate observations.

The next useful milestone is a trustworthy measurement system. More signals or paid data alone will not repair the current evidence.
