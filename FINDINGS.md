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

CORRECTION (round one): tune-half is Sep 2013–Aug 2019 (calm bull, no
crashes), holdout-half Aug 2019–Jul 2025 (COVID crash, 2022 bear,
recovery). Earlier notes saying tune held the bears were wrong — Yahoo
cache extended history to 2010 regardless of download start. Split
stays clean (tune never eyeballed); regimes are harsher than labeled,
which strengthens holdout results and weakens tune ones.

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
8. **Tune/holdout, 3 configs, small caps.** Tune (2013–19 bull):
    everything loses to monkeys except cash. Holdout (2019–25 stress):
    wide +20.8% — but holdout was eyeballed during construction.
    Contaminated, not claimed. (Eras corrected; see note at top.)
9. **Long-short (top-10/bottom-10, 5% borrow).** Tune Sharpe -0.03.
   Bar failed, no holdout. Hedge cuts DD (-0.26), short book earns nothing.
10. **Mid-cap ML generalization (fresh universe).** AUC 0.559 —
    replicates where overfitting cannot reach. P&L +4.2% / -0.44 DD.
    Signal real, harvest weak.
11. **Textbook 12-1 momentum, no ML.** Small: tune +9.5% (Sharpe 0.41),
    holdout +6.2% (Sharpe 0.20, DD -0.56). Mid: tune +3.3%, holdout
    +11.4% (Sharpe 0.35). Positive every half, both universes. Best
    raw result in the project. (CORRECTION: holdout CAGRs were first
    misreported higher — shared running equity curve wasn't reset per
    slice. Caught by reconciliation check, fixed, rerun. Sharpe/DD were
    always correct.) (SUPERSEDED Sep 13, 2026: these numbers came from
    a log-return-averaging bug that understated equal-weight portfolio
    returns. Rerun under the corrected ledger is meaningfully stronger
    in every half, both universes — see "Rerun under corrected ledger"
    below.)
12. **Ensemble (50% ML-wide + 50% momentum, zero new params).** Tune:
    ml -2.2% (Sharpe -0.10), mom +8.7% (+0.40), ens +3.1% (+0.17). Bar
    was Sharpe > 0.41 with DD > -0.31. Failed — averaging dilutes.
    The ML adds nothing to momentum. Buried as a signal; kept running
    in paper only as the losing side of the bet. (SUPERSEDED Sep 13,
    2026: rerun with fresh, leak-fixed probabilities and the corrected
    momentum ledger — same conclusion, new numbers, see "Rerun under
    corrected ledger" below.)
13. **Self-improvement loop, round one (6 proposals, tune-half).**
    P1 printed Sharpe 2.9/CAGR 2900% — investigated, not celebrated:
    nine idiosyncratic biotech/crypto doubles in 1–2 name books during a
    bull market (CLSK verified tick-by-tick as genuine). Numeric pass
    rejected openly as lottery concentration; protocol gap noted.
    P5-gate25 (wide + 0.25 gate, 81/82 months invested, avg 9.4 names,
    no month above +25%) passed clean: tune Sharpe 0.68, DD -0.26.
    Promoted once to holdout: **CAGR +12.7%, DD -0.35, Sharpe 0.41**
    through COVID, 2022 bear, and recovery at 25bps. First ML config
    with a clean stress-period holdout. Budget: 14/20 left. (SUPERSEDED
    Sep 13, 2026: pre-fix ledger result; corrected replay is +6.3%/-0.37/0.35.)
14. **Self-improvement round two (P7 liquidity, P8 vol scaling).**    P7: Sharpe 0.56 pass, DD -0.361 — missed the -0.35 bar by 0.011.
    Dead by the letter of the law, stated openly. P8 (per-name inverse-
    vol sizing): tune Sharpe 0.63, DD -0.27. Promoted once: holdout
    **CAGR +6.1%, DD -0.39, Sharpe 0.22**. Positive through stress,
    shrunk as honest results do (DD breached -0.35 out-of-sample, which
    is itself a finding about bar-setting). Budget: 12/20 left.
    (SUPERSEDED Sep 13, 2026: corrected replay is -2.0%/-0.42/0.02.)
15. **Vol-scaled monkeys (is P8 just risk parity?).** Random 20-name
    books with inverse-vol sizing, tune-half: CAGR +3.0%, DD -0.22,
    Sharpe 0.22 vs P8's 0.63/-0.27. Verdict: sizing explains ~a third
    of P8, selection the rest. Neither alone explains it; the critique
    sharpened the claim instead of killing it.
16. **Quality sleeve (top-half ROE, 17 mega caps, annual).**
    Pre-committed single shot, 2022–25 decisions (2021 empty — filings
    start Oct 2021): +12.7%/yr vs SPY +10.2%, 4/4 positive years.
    Illustrative only (4 decisions prove nothing), direction matches
    French RMW +2.4%/yr. Costs negligible at 1 rebalance/yr.
17. **Self-improvement round three (P9 breadth, P10 selectivity).**
    P9-top30: tune Sharpe 0.59, DD -0.30, passes bar. Promoted once:
    holdout **+4.3%, DD -0.40, Sharpe 0.16** — positive, weakest
    promotion yet, DD breached bar out-of-sample again. P10-gate35:
    Sharpe 0.33, dead — tighter selection concentrated without paying.
    Budget: 10/20 left. (SUPERSEDED Sep 13, 2026: corrected P9 replay is
    -2.1%/-0.42/0.01.)
21. **Self-improvement round four (P11 short holds, P12 max breadth).**
    P11-hold10: Sharpe -0.22, dead — monthly decisions with 10d holds
    double turnover and go stale faster; direction was wrong. P12-top40:
    tune Sharpe 0.57, DD -0.31, breadth verified (81/82 months, avg
    11.3 qualifiers — cap rarely binds, so near-twin of P8, which
    corroborates rather than duplicates). Promoted once: holdout
    **+3.5%, DD -0.41, Sharpe 0.14**. Fourth straight decaying
    promotion (+12.7 → +6.1 → +4.3 → +3.5). Budget: 8/20 left.
    (SUPERSEDED Sep 13, 2026: corrected P12 replay is -2.0%/-0.41/0.02;
    the sequence quoted here is historical.)
18. **Multi-asset trend, quarterly rebalance (Faber's turnover note).**
    Tune +5.8% (Sharpe 0.58, DD -0.13), holdout +6.9% (Sharpe 0.60,
    DD -0.27). Monthly version: tune +7.3% (0.82), holdout +7.4%
    (0.62). SPY buy-hold same window +14.5% — strategy trails on
    return, wins on risk-adjusted calm. (Also fixed a slice-CAGR bug
    that had inflated momentum holdout prints; Sharpe/DD unaffected.)
    (SUPERSEDED Sep 13, 2026: same log-return-averaging bug as #11 —
    see "Rerun under corrected ledger" below.)
19. **Data-hygiene incident (Sep 2026).** Found 105 stale large-cap
    files (Dec 2025) mixed with fresh ones — universe-scoped refreshes
    never re-requested them, and SPY silently dropped out of
    multi-asset holds for 9 months with no error (NaN comparisons).
    Paper-grade numbers stood (fresh IWM benchmark); multi-asset
    re-ran slightly better with SPY present. Fix: `assert_vintage`
    guard wired into every runner — mixed vintages now fail loudly.
20. **International trend sleeve (EFA/EEM/VNQ/GLD, same frozen rule).**
    Tune +2.9% (Sharpe 0.22, DD -0.26), holdout +11.2% (Sharpe 0.88,
    DD -0.16). Both positive, divergent by regime like everything else.
    Holdout Sharpe 0.88 is the best single-half risk-adjusted print in
    the project. No loop budget spent (new sleeve, descriptive).
    (SUPERSEDED Sep 13, 2026: same log-return-averaging bug as #11 —
    see "Rerun under corrected ledger" below.)
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

- Value on small caps (needs paid fundamentals; free sources exhausted).
- Quality/profitability tilt (live factor, working endpoints).
- Paper log accumulation (months, no peeking).

## Bottom line (current)

Public-data cross-sectional rank predicts weakly (AUC ~0.55) and adds
nothing blended with momentum — the ensemble dilutes rather than helps,
before and after the accounting fixes. The only strategy positive in
every half tested, across three universes, is textbook 12-1 momentum
with no ML. Under corrected portfolio accounting (Sep 2026), that
result is stronger than this file claimed for most of its life:
small-cap momentum beats SPY buy-and-hold outright in both halves, not
just on a risk-adjusted basis, and mid-cap does the same in the stress
half. The reversal came from fixing a return-accounting bug, not from a
new signal or new data. Independent convergence with the source
project's verdict on everything else — insider data, neural nets,
sentiment, long-short — all still fail, for the same structural
reasons, on the same corrected accounting.

## Regime correction (the nessessary uncomfortable table)

Both ML halves were bull markets, not one calm + one stress:
tune 2013–19 SPY +12.6%/IWM +8.9%; holdout 2019–25 SPY +15.1%/IWM +7.4%
(crashes inside, V-recoveries after). Best case for the ML config
anywhere is a tie on risk-adjusted terms (multi-asset Sharpe 0.82 vs
SPY ~0.8); the legacy-cache, fixed-ledger P5 replay (+6.3%) trails SPY's
+15.1% on this window. The dates are comparable, but its probabilities
predate the leak fix, so it is a historical counterfactual rather than a
clean current performance estimate.

Momentum's wins were assumed relative, not absolute, when this section
was first written. That assumption holds for the ML config above. It
does not hold for momentum. (REVISED Sep 13, 2026: the claim that
followed this paragraph — "no long-only config beat SPY in either
half," downgrading momentum to "weak signal that trails the index" —
compared momentum's CAGR against this section's SPY/IWM figures, which
come from the *ML panel's* window above, not momentum's own dates. Not
apples-to-apples, and on top of a return-accounting bug, fixed later,
that had specifically been understating momentum. Repriced on
momentum's own window with the corrected ledger: small-cap momentum
beats SPY buy-and-hold in both halves; mid-cap beats it in the stress
half. Full numbers in "Momentum vs. SPY, closed out" below. This
correction is scoped to momentum — the ML/multi-asset comparisons above
were valid on their own terms.)

## After everything

Started from 29 of his videos and one question: can his AI stock
predictor be rebuilt better. Ended with a fuller answer than his.

What was tried, in order: trees-first rebuild, purge/embargo validation,
monkey baselines, costs from day one, three universes (large/small/mid),
raw and conviction insider data, risk overlays, long-short, textbook
momentum, Graham value on real financials, French factor regimes, live
paper trading. Twelve numbered experiments, each with a pre-committed
bar where selection was involved.

What survived: a ~0.55 AUC rank signal that replicates on untouched
data but never survives risk on its own; textbook 12-1 momentum,
positive in all four halves tested across two universes and, under
corrected accounting, outright beating SPY buy-and-hold in small caps
(both halves, +17.8%/+21.7%) and mid-cap's stress half (+24.7%) — not
just calmer than the index, actually ahead of it; plus multi-asset
trend (+7.9%/+8.0% halves, Sharpe 0.94/0.73 — still the best
risk-adjusted line, though no longer the only thing beating the index
outright); a live paper log grading both monthly.

What died: everything with a neural net, everything with insider data
(three formulations), everything with social sentiment (both directions,
his newest video agrees), value on mega caps 2021-26 (median price 2.77x
Graham value; the strategy holds cash and trails), long-short small caps
(hedge works, alpha doesn't), and every risk overlay tuned by hand.

Why it died is structural, not bad luck: value/size factors ran negative
for a decade (French HML -1.1%, SMB -2.3%), famous signals are priced in,
strange signals are sparse where they're free and paywalled where they'd
matter, and small-cap friction eats paper edges. His channel banner says
it in net-worth font; this repo says it in 70 folds.

Momentum's win isn't the same kind of finding as the things above it —
it held up under an isolated same-window, same-ledger recheck against
the old buggy version, which is closer to a controlled experiment than
a hope.

What runs on its own now: monthly paper log (ML + momentum sleeves),
monthly momentum sleeve, and the value engine waiting on small-cap data
that costs money everywhere checked.

What would reopen the case on the parts still dead: paid point-in-time
fundamentals on small caps, a new data regime (satellite/credit-card
class, with budget), or months of live log contradicting the
backtests. What's no longer closed: small-cap momentum's SPY-beating
result held up under an isolated recheck (Sep 2026) — that's the
corrected number, not the bug talking. ML, insider data, and
hand-tuned risk overlays are still closed cases; the index fund still
wins there. Textbook momentum is open, with a positive point-in-time
QuantConnect replication now complete and live confirmation still pending.

## Who does this better (research waves)

**Numerai (the canonical better).** Thousands of staked models,
market-neutral global equity, $450M AUM, JPMorgan $500M capacity, +25%
net 2024 at 2.75 Sharpe. Starter code is XGBoost — our tool at 1000x
breadth. Works through neutrality + diversity + skin in the game, not
cleverer models.

**Medallion (the ceiling).** 66%/yr 1988–2018, never a down year,
negative factor loadings. Thousands of short-term long+short bets,
mid-frequency, capped size, leverage via consistency. Uncopyable
retail; sets direction (breadth, short holds, neutral).

**Gu-Kelly-Xiu (the academic ceiling).** Trees + shallow nets win, deep
nets lose; monthly R² 0.3–0.4%; winners are momentum, liquidity,
volatility — our top features, independently. Gains via long-short
deciles at scale.

**QuantConnect/LEAN (the infrastructure better).** 21k stars,
survivorship-free data with delistings, built-in slippage/fees.
Its cloud data and brokerage models directly test our weakest links:
execution realism and survivor universes.

**The honest minority.** Quantopian: 888 strategies, backtest-live
Sharpe correlation zero. Stanford 2025: 58% of retail algos die in 3
months. Quantpedia: every winner carries -50%+ DD with low persistence.
Our discipline is the documented base rate, not paranoia.

## AI self-improvement (researched, loop built)

Taxonomy: manual prompting (his way — weakest, no stats, single
lineage); Reflexion/self-refine (wrong for finance — verbal loops
manufacture overfitting narratives); AutoML/HPO (PBO-as-a-service
without trial accounting); AIDE (best fit — solution tree + cheap eval,
needs a holdout gate); AlphaEvolve (right architecture, needs cheap
correct evals — finance has neither); Darwin Gödel Machine (keep the
archive idea, not the benchmark faith).
Built: improve.py — AIDE structure + DGM archive + tune→holdout-once
gate + 20-trial lifetime budget, me as mutator. Rounds 1–2 done above.


## LEAN migration (started Sep 2026)

Docker Desktop installed + verified (hello-world runs). LEAN engine
pull in progress. Goal: re-run momentum sleeve on survivorship-free
US equity data (AlgoSeek/QuantQuote back to 1998, delistings included)
— the first backtest here without the survivor asterisk. Open question:
bulk download QCC costs on a free account; sample data first.

## Corrected promotion meta-analysis (loop paused, 8 trials banked)

The original decay sequence (+12.7%, +6.1%, +4.3%, +3.5%) came from the
pre-fix ledger and is superseded. On the original cache and dates, all four
still clear their original tune gate, but P5 falls to +6.3% in holdout and
P8/P9/P12 fall to about -2%. That counterfactual is useful but retains the old
leak-affected probabilities. The leak-fixed v2 cache reverses the holdout
result, including on common dates, while all four fail its tune gate. The loop
therefore remains unresolved and pipeline-sensitive rather than cleanly
successful or failed. It stays frozen; spending the remaining eight trials on
already inspected history would not create a new independent test.

## Second accounting correction: label leak, paper dating, value engine, momentum accounting (Sep 13, 2026)

Follow-up to an external code review (REVIEW.md). Four more real bugs found
and fixed; **every AUC, holdout CAGR, and paper pick discussed above was
produced before these fixes and must be treated as historical, not
validated, until rerun.**

1. **Walk-forward validation leak.** `label_end` was a calendar-day
   approximation (`Date + 20 calendar days`); `model_lgbm.walk_forward`
   purged training rows against it but never purged validation rows against
   the test boundary at all, and silently discarded `embargo_days`
   (`_ = embargo_days`). Confirmed on cached folds: a validation label could
   require prices nearly a month past the test start. Since validation
   labels feed the isotonic calibrator / LightGBM early stopping, every AUC
   and probability in this file may be inflated by leakage, not just
   tuned-window numbers. Fixed: `label_end` is now a true trading-day shift
   (`groupby("ticker")["Date"].shift(-HORIZON)`), and both train-vs-val and
   val-vs-test are purged against it plus an embargo buffer. Probabilities
   need regeneration; every AUC/backtest number downstream of them is
   unvalidated until then.
2. **Paper log was dating everything ~a month stale.** `paper.py` inferred
   on the labeled panel's last date, which is always `HORIZON` trading days
   behind because those rows need a known forward return that doesn't exist
   yet — confirmed the panel's last date was 2026-08-13 while true latest
   data was 2026-09-11. The momentum sleeve reused that same stale date
   under its own tag. Fixed with a new `build_latest()` that computes
   today's features without requiring today's (unknowable) label. Also
   fixed: grading matched model identity by (date, ticker) only, so two
   sleeves picking the same name on the same date silently overwrote each
   other's attribution (now carries `model` straight from the log row);
   grading credited the stop at exactly the threshold on a gap and measured
   the no-stop case from a different entry price than the stop case (now
   one gap-aware simple-return calc from the logged decision-day close, one
   round-trip cost, matching the ledger's own convention). Existing
   `data/paper/log.csv` entries were logged under the old (wrong) dates and
   should be read as such, not as same-day observations.
3. **Momentum lookback wasn't one frozen spec.** `mom_run.py` (the source of
   every momentum number in this file) is frozen at lookback=252/skip=21.
   `paper.py`'s momentum sleeve and `qc_momentum.py` both used 273/22 — an
   undocumented, unexplained drift. Both now match mom_run.py exactly.
4. **Log-return averaging bug, found in three more places.** The ledger
   rewrite above fixed this in `predictor/backtest.py` but three standalone
   scripts still averaged log returns then exponentiated — wrong for an
   equal-weight portfolio (a +100%/-50% pair nets +25%, not 0%). Hit:
   `mom_run.py` (source of every "textbook momentum" number, items 11/18),
   `multi_run.py` (multi-asset trend and the intl sleeve, items 18/20), and
   `ens_test.py` (the ML/momentum ensemble, item 12) — the last of these
   also mixed a simple-return ML leg (already fixed) with a log-return
   momentum leg and averaged them, a straight unit mismatch. All three now
   use simple returns and `(1+net).cumprod()`.
5. **Value engine gave the first qualifying stock all available cash.**
   `value/backtest.py`'s buy loop computed each new position's allocation
   as `(cash + held value) / (holds + 1)` per ticker, one at a time — so
   the first name to qualify in a given month got the full target and left
   nothing for the rest. Diversification depended on ticker iteration
   order, not the equal-weight rule the docstring claims. Fixed: all
   qualifiers for the month are found first, then sized against the book
   that results from all of them buying. Added a regression test
   (`value/test_value.py::test_diversification_ignores_ticker_order`) that
   fails under the old code (n=1) and passes now (n=2) for two identical
   equally-undervalued names.

None of these were retuned or re-selected — same rules, same thresholds,
same universes. This is the correctness-first replay REVIEW.md recommended,
not new strategy search. The promoted loop proposals, items 11, 12, 18, 20,
and the AUC-dependent ensemble have now been replayed below. Historical paper
rows and unreplayed experiments retain their explicit pre-fix status.

## Two-cache replay: promoted loop proposals (Sep 13, 2026)

`python3 improve.py replay-promoted --cache CACHE` re-evaluates P5, P8, P9,
and P12. The explicit cache argument is required and every output prints its
date range. The command does not spend trial budget, alter proposal status, or
overwrite the historical registry. Both replays use the corrected simple-return
portfolio accounting, weight-based turnover, actual gap-stop exits, cash
exposure, and starting-equity drawdown measurement. It also fixes the risk
scaler that previously reduced the number of names while leaving the book
fully invested, increasing concentration instead of reducing exposure.

**Legacy cache (`sc_full`, 24 folds).** This preserves the original selection
period and answers the counterfactual “what would the promoted configurations
have reported with the corrected ledger at the time?” It still uses model
probabilities generated before the validation-leak fix.

| Proposal | Tune CAGR | Tune Sharpe | Tune DD | Holdout CAGR | Holdout Sharpe | Holdout DD |
|---|---:|---:|---:|---:|---:|---:|
| P5 lower gate | 13.1% | 0.70 | -30.1% | **6.3%** | 0.35 | -36.9% |
| P8 inverse-vol weights | 11.8% | 0.65 | -34.1% | **-2.0%** | 0.02 | -42.0% |
| P9 top 30 | 12.1% | 0.67 | -34.1% | **-2.1%** | 0.01 | -41.8% |
| P12 top 40 | 11.9% | 0.66 | -34.1% | **-2.0%** | 0.02 | -41.3% |

(Tune column revised Sep 14, 2026 -- see "leg_simple NaN guard" below. Holdout
untouched: re-verified identical.) All four pass the original tune bar of
Sharpe > 0.41 and drawdown above -0.35.
The promotions were procedurally faithful under that vintage. P5 remains
positive and trails matched-window SPY; the other three fail this replay.

**Leak-fixed cache (`sc_full_v2`, 26 folds, `LEG` excluded).** This applies the
corrected probability pipeline, but its tune/holdout boundary moves six months
and its holdout extends through July 2026. It answers whether the already chosen
configurations look useful under the current walk-forward pipeline, not what
would have happened at the original promotion decision.

| Proposal | Tune CAGR | Tune Sharpe | Tune DD | Holdout CAGR | Holdout Sharpe | Holdout DD |
|---|---:|---:|---:|---:|---:|---:|
| P5 lower gate | 4.8% | 0.35 | -33.0% | **25.9%** | 0.83 | -41.6% |
| P8 inverse-vol weights | 5.6% | 0.40 | -31.1% | **17.7%** | 0.65 | -40.8% |
| P9 top 30 | 5.5% | 0.40 | -30.4% | **17.7%** | 0.65 | -40.8% |
| P12 top 40 | 5.7% | 0.41 | -29.9% | **17.9%** | 0.66 | -40.8% |

(Tune column revised Sep 14, 2026, same fix as above. Holdout unchanged.)
P8 and P9 still fall just short of the original tune Sharpe gate of 0.41; P12
now lands at 0.410, essentially on the bar rather than clearly under it. None
of the three is comfortably over the line, so the practical conclusion is
unchanged -- none would be promoted with confidence if the protocol were
restarted on this cache -- but P12 is a genuine coin flip, not a clear miss.
The holdout results are nevertheless real supporting evidence for the frozen
configurations. They are post-selection evidence, not a new sealed-holdout
claim.

**Common-date sensitivity check (February 28, 2020–July 9, 2025).** Restricting
both holdout replays to identical dates does not remove the reversal:

| Proposal | Legacy CAGR / Sharpe / DD | v2 CAGR / Sharpe / DD |
|---|---:|---:|
| P5 | +5.0% / 0.31 / -36.8% | +22.8% / 0.73 / -41.6% |
| P8 | -3.1% / -0.02 / -41.9% | +14.9% / 0.56 / -40.8% |
| P9 | -3.1% / -0.03 / -41.7% | +14.5% / 0.55 / -40.8% |
| P12 | -3.0% / -0.02 / -41.1% | +14.8% / 0.56 / -40.8% |

The added 2025–2026 period is therefore not the main explanation. The caches
also differ in purge logic, fold construction, data vintage, and one universe
member, so this check cannot attribute the reversal to leakage alone. The
defensible conclusion is that the overlay result is highly pipeline-sensitive.
Publish both vintages and accumulate genuinely prospective observations.

**What the two original caches' probabilities look like, row for row.**
`cache_compare.py` finds 246,003 common `(Date, ticker)` rows through July
2025. Labels agree 100%; median close difference is zero and maximum absolute
close difference is 0.676%. AUC is effectively identical (0.550 legacy,
0.549 v2 overall; 0.544 each on the shared holdout dates), and per-date rank
correlation is respectable at 0.74 median. But their top-20, threshold-0.25
picks overlap by only **Jaccard 0.38**. Roughly 62% of selected names differ
month to month. Aggregate AUC is therefore hiding economically large
instability at the portfolio cutoff.

**Controlled purge ablation.** `purge_ablation.py` rebuilt both probability
sets on one current price snapshot. It held all 26 fold boundaries, test rows,
labels, closes, universe, model code, and seed zero constant. One side exactly
reproduced the pre-fix split logic from `091b80f^`; the other used the current
trading-day label endpoints and train/validation purge. The fixed side exactly
reproduced `sc_full_v2` (probability correlation 1.000). Only purge logic varied:

| Proposal | Legacy-purge tune | Legacy-purge hold | Fixed-purge tune | Fixed-purge hold |
|---|---:|---:|---:|---:|
| P5 | +0.8% / 0.14 / -43.2% | +11.9% / 0.57 / -35.6% | +4.8% / 0.35 / -33.0% | +25.9% / 0.83 / -41.6% |
| P8 | +4.2% / 0.32 / -40.4% | +6.7% / 0.40 / -39.4% | +5.6% / 0.40 / -31.1% | +17.7% / 0.65 / -40.8% |
| P9 | +4.3% / 0.32 / -40.2% | +6.6% / 0.40 / -39.7% | +5.5% / 0.40 / -30.4% | +17.7% / 0.65 / -40.8% |
| P12 | +4.3% / 0.32 / -40.2% | +6.4% / 0.39 / -39.7% | +5.7% / 0.41 / -29.9% | +17.9% / 0.66 / -40.8% |

Format is CAGR / Sharpe / maximum drawdown. Tune columns revised Sep 14, 2026
(see "leg_simple NaN guard" below); hold columns re-verified identical. The
purge repair alone produces a large positive holdout shift while slightly
lowering aggregate common-row AUC (0.553 to 0.551). Probability rank
correlation is 0.762, but selected-name Jaccard is only 0.392. Fixed-purge
P12's tune Sharpe (0.41) now lands almost exactly on the original gate rather
than clearly under it; the other five tune cells stay clearly below it. Still
a mechanism diagnostic, not a retroactive promotion -- P12 sitting on the line
is a reason for caution about that boundary, not a reason to wave it through.

The old `sc_full` cache still cannot be fully reconstructed: refitting its
legacy purge against today's inputs reaches only 0.855 rank correlation and
0.498 pick overlap with the saved artifact. Its fold CSVs contain no feature,
raw-data, code, or environment fingerprint. That missing provenance prevents
attributing the remaining difference. `run.py --save-proba` now refuses to
overwrite a cache and writes `manifest.json` with arguments, git state,
pipeline settings, model backend, versions, fold dates, panel coverage, and a
SHA-256 hash for every price input.

### leg_simple NaN guard (found and fixed Sep 14, 2026)

Found while building the Deflated Sharpe Ratio package (below), because that
script computed Sharpe with `numpy.mean`/`numpy.std` directly instead of
pandas' default `skipna=True` reductions that every other summary in this
project goes through. `predictor/backtest.py`'s `leg_simple` returned `NaN`
(rather than raising) when a held name's entry or exit close was missing --
not a bad raw price (`data/raw/*.csv` has zero NaN closes, checked directly)
but a normal, expected gap in the per-ticker feature panel: a name can lack a
row for a specific date in a fold's cached CSV without that being any kind of
data corruption. `_run_book` wraps each name's `leg_simple` call in
`try/except` specifically to drop one bad leg and keep the rest of that
month's names -- but a silently-returned NaN doesn't raise, so it was never
caught, and it corrupted that entire month's weighted return instead of just
that one name's contribution. Every downstream summary (`score()`,
`ev.summarize()`) uses pandas' default `skipna=True` mean/std/cumprod, so
this was invisible in every previously reported number: the bad months were
quietly excluded from the statistics rather than explicitly handled, so CAGR
and Sharpe looked plausible while resting on fewer clean months than reported.

Fixed: `leg_simple` now raises `ValueError` on a non-finite entry or exit
price, so `_run_book`'s existing exception handling does what it was already
meant to do -- exclude that one name's leg and let the rest of the month's
book still contribute a real, correctly-computed return, instead of losing
the whole month. Added
`predictor/test_backtest.py::test_leg_simple_raises_on_non_finite_price` as a
regression test.

**Verified impact, don't take on faith:** reran every affected artifact.
Every previously reported **holdout** figure across `sc_full`, `sc_full_v2`,
and both purge-ablation caches is unchanged (re-verified bit-for-bit against
the tables above and below). **Tune-half** figures for P5/P8/P9/P12 shift --
sometimes materially, as in the purge-ablation legacy-purge column (P5 tune
Sharpe 0.03 to 0.14, drawdown -52.2% to -43.2%) -- because the specific
data-panel gaps that triggered this all happened to fall on tune-period
dates for these folds. All corrected tables above already reflect the fix.
mom_run.py/multi_run.py (momentum, multi-asset) never call `leg_simple` and
build their price panel straight from `data/raw`, so they were never
affected; none of the momentum, SPY-comparison, bootstrap, or factor-
regression results change.

## Rerun under corrected ledger: momentum, multi-asset, ensemble (Sep 13, 2026)

`mom_run.py` and `multi_run.py` rerun after fixing the log-return-averaging
bug (#4 above), same rules/thresholds/universes/costs, no retuning. One
data-hygiene drop: `LEG` (Leggett & Platt, in the smallcap universe) is
excluded — Yahoo's chart API now returns only 6 rows for it regardless of
requested range, last print 2026-08-27 volume 0, consistent with a
delisting or trading halt. Everything else in both universes is same-day
fresh (verified before running, not just trusted).

**Same-day reconciliation:** the standalone runners still calculated
membership turnover independently from the central ledger, and
`ens_test.py` still formed momentum with 273/22 observations despite the
frozen 252/21 specification. Turnover now comes from the shared weight-based
function, the ensemble uses 252/21, and the tables below contain the replayed
figures. The direction of every verdict is unchanged; momentum improves
slightly and still dominates the ensemble.

**Isolated the fix's effect directly**: reran the identical window and
ticker set through the old log-averaging formula side by side with the
fix. The old formula reproduced the original item-11 numbers almost
exactly (small tune 9.47%/0.414/-0.313 vs the recorded 9.5%/0.41/-0.31;
mid tune 3.29%/0.151 vs 3.3%/0.15) — confirming the jump below is the
accounting fix, not a different window, different tickers, or new
market data.

**Textbook 12-1 momentum (supersedes #11):**
| | tune CAGR | tune Sharpe | tune DD | hold CAGR | hold Sharpe | hold DD |
|---|---:|---:|---:|---:|---:|---:|
| Small, old (log-avg) | +9.5% | 0.41 | -0.31 | +6.2% | 0.20 | -0.56 |
| Small, fixed | **+17.8%** | **0.84** | -0.26 | **+21.7%** | **0.78** | -0.37 |
| Mid, old (log-avg) | +3.3% | 0.15 | -0.32 | +11.4% | 0.35 | -0.37 |
| Mid, fixed | **+9.8%** | **0.54** | -0.28 | **+24.7%** | **0.81** | -0.31 |

Not a small correction — Sharpe roughly doubled and holdout CAGR roughly
tripled in both universes. Direction makes sense: log-averaging a monthly
equal-weight book of ~8 high-dispersion small/mid-cap movers systematically
understates the true portfolio return (Jensen's gap runs the other way
here than in the P1 bug above, which *inflated* a return). The corrected
number says the momentum signal was stronger than this file has claimed
for months, not weaker.

**Multi-asset trend (supersedes #18 domestic legs), same window:**
| | tune CAGR | tune Sharpe | tune DD | hold CAGR | hold Sharpe | hold DD |
|---|---:|---:|---:|---:|---:|---:|
| Monthly, old | +7.3% | 0.82 | -0.13 | +7.4% | 0.62 | -0.24 |
| Monthly, fixed | **+7.9%** | **0.94** | -0.13 | **+8.0%** | **0.73** | -0.23 |
| Quarterly, fixed | +6.4% | 0.70 | -0.13 | +7.6% | 0.72 | -0.27 |
| SPY buy-hold, same window | +14.5% | — | — | — | — | — |

Smaller uplift than momentum, as expected — a 4-asset ETF book has far
less cross-sectional dispersion than an 8-name small-cap basket, so the
log-vs-simple gap is smaller. Conclusion is unchanged: still trails SPY
buy-hold on raw return, still wins on risk-adjusted terms, now by a wider
margin (Sharpe 0.94 tune vs the old 0.82).

**International sleeve (supersedes #20):** tune +3.3% (Sharpe 0.32,
DD -0.26), holdout +11.7% (Sharpe 0.98, DD -0.16) — close to the old
+2.9%/+11.2% print; low-dispersion ETF basket again limits the gap.
Still positive both halves.

**Open question, not resolved here:** the "regime correction" section
above claims no long-only config beat SPY buy-and-hold on return in
either half. Small-cap momentum's new holdout CAGR (+21.7%) is close to
or above the SPY holdout figure cited there (+15.1%), which would
overturn that claim if the windows line up — but that comparison used a
different price series (SPY vs this rerun's IWM-relative momentum) and
hasn't been checked apples-to-apples yet. Needs a direct SPY-benchmarked
rerun before the regime-correction verdict is revised either way.

`ens_test.py` (the ML/momentum ensemble) was not rerun in the entry
above — it needed fresh probabilities from the walk-forward leak fix
first. Done as a follow-up (Sep 13, 2026, same day): regenerated via
`run.py` on the same smallcap universe (minus `LEG`, price-only —
matching what `paper.py` actually deploys, no insider merge), 26 folds
of 7-month test windows, 3 seeds each, cached to
`data/cache/sc_full_v2` (old cache kept at `data/cache/sc_full`, 24
folds, for comparison — fold count shifted from the LEG drop, the purge
fix changing which folds clear the row-count gate, and more price
history since that cache was built).

**Mean AUC over 78 fits: 0.554** — essentially unchanged from the
pre-fix 0.550 this file has cited for small-cap price-only ML (item 4).
Unlike the momentum log-averaging bug, the walk-forward leak turned out
not to move this particular number much in practice, even though the
leak itself (validation labels touching the test window) was real and
is still worth having fixed for any future retuning.

**Ensemble (supersedes item 12), fresh probabilities + fresh momentum leg:**
| | tune CAGR | tune Sharpe | tune DD | hold CAGR | hold Sharpe | hold DD |
|---|---:|---:|---:|---:|---:|---:|
| ML alone | -0.1% | 0.13 | -0.62 | +8.9% | 0.38 | -0.43 |
| Momentum alone | +17.3% | 0.86 | -0.25 | +20.3% | 0.71 | -0.37 |
| 50/50 ensemble | +9.8% | 0.60 | -0.42 | +16.5% | 0.63 | -0.38 |

Conclusion unchanged, on corrected numbers: the ensemble underperforms
momentum alone on both CAGR and Sharpe in both halves — averaging in a
weaker, more-drawdown-prone ML leg still dilutes rather than helps. The
item-12 pre-committed bar (Sharpe > 0.41, DD > -0.31) was itself built
from the momentum figures this session has since revised upward
(item 11 rerun above); fresh momentum alone clears that bar easily now,
so the bar is no longer the meaningful comparison — momentum-alone vs
ensemble, both on fresh numbers, is, and momentum alone still wins.

## Momentum vs. SPY, closed out (Sep 13, 2026)

Closes the open question left above: does corrected small-cap momentum
actually beat SPY buy-and-hold, given the two numbers looked close?

The "regime correction" section's SPY/IWM figures came from the ML
panel's tune/holdout split (2013–19 / 2019–25) — a different dataset
with its own date range, not the momentum backtest's own window. Comparing
across two different windows is not a valid test. Fixed properly this
time: pulled the momentum backtest's own decision dates and priced SPY
and IWM at exactly those dates, same month-count annualization
(`years = months/12`) used everywhere else in this file, so the
benchmark and the strategy are measured over the identical calendar
window with the identical CAGR convention.

**Small-cap momentum's own window: 2011-02-28 to 2026-09-10 (188
months), split at 2018-11-30/2018-12-31 (94/94):**

| | tune CAGR | hold CAGR |
|---|---:|---:|
| SPY buy-hold | +11.9% | +16.9% |
| IWM buy-hold | +9.8% | +11.6% |
| Small-cap momentum (fixed ledger) | **+17.8%** | **+21.7%** |

Small-cap momentum beats SPY buy-and-hold on raw return in **both**
halves, not zero. That directly overturns the regime-correction
section's "no long-only config beat SPY in either half" for at least
this one configuration.

**Mid-cap momentum, same window (188 months, same split):**

| | tune CAGR | hold CAGR |
|---|---:|---:|
| SPY buy-hold | +11.9% | +16.9% |
| Mid-cap momentum (fixed ledger) | +9.8% | **+24.7%** |

Mixed: mid-cap trails SPY in tune, beats it clearly in holdout — the
stress-period result (COVID, 2022 bear, recovery) is the one that
matters more for "does this survive when it's needed," and it clears
SPY by a wide margin there.

Reading: the file's "index fund wins" bottom line was never wrong about
risk-adjusted terms or about the ML/insider/long-short lines that
actually died. But the specific claim that momentum never beats SPY on
raw return was an artifact of comparing the wrong two windows on top of
an accounting bug that was independently understating momentum's
return — fix either one and the claim gets shakier; fixing both breaks
it for small caps outright. The "Bottom line" and "After everything"
sections above still describe the pre-fix picture and should be read
as historical until rewritten; not rewritten here to avoid changing two
narrative sections in the same pass as the numbers that justify the
change.

## Dump (everything, unstructured, Sep 2026)

## Accounting correction and QC check (Sep 13, 2026)

The P1 "lottery" result is retracted as evidence. Its vol-only configuration
used `stop_frac=2.0` to disable stops; the former stop implementation treated
that as a hit and credited a positive return on flat prices. The backtest ledger
now uses simple returns, compounds them directly, validates stop fractions,
records gap-aware exits at the actual close, charges weight-based turnover, and
keeps unused exposure in cash. Historical P1/vol-only figures must not be used
until the affected proposals are rerun under this ledger.

### QuantConnect migration record

The cloud project is **Sleepy Orange Bison**. The first submitted version
failed before processing data with the exact engine error:

`2010-01-04 00:00:00 Runtime Error: Unable to cast object of type
'QuantConnect.Data.Fundamental.Fundamental' to type 'System.String'. in
Extensions.cs:line 3362`

Cause: the draft used the old PascalCase coarse-universe pattern and returned
fundamental records. The selector was changed to QuantConnect's current Python
fundamental-universe API: `add_universe`, lower-case fundamental attributes,
and a returned list of `symbol` objects. The corrected selector initialized,
warmed up, and processed the full requested period.

The completed run covered January 2011 through June 16, 2026 on the Community
B-MICRO node, using $100,000 starting equity, the Interactive Brokers brokerage
model, daily resolution, a 200-stock dollar-volume universe, monthly 10:30
rebalances, 273-day lookback, 12-1 momentum, and a top-decile portfolio.
QuantConnect reported: total return **+1,647.489%**, end equity **$1,747,488.78**,
maximum drawdown **59.400%**, Sharpe **0.58**, annualized return **20.319%**,
and **$5,224.28** in fees across 3,435 orders.

That first run is not a clean survivorship-effect estimate. It produced
repeated handled order errors for newly selected securities that had not yet
received a price bar. Its large return and 59.4% drawdown document the first
cloud diagnostic only.

The guarded, frozen-spec rerun was completed September 13, 2026 as **Creative
Yellow Antelope**. It used 252 trading observations with a 21-day skip and the
same project, universe construction, starting equity, brokerage model, and
monthly rebalance design. The result page covered January 2011 through June 17,
2026 and reported total return **+3,455.564%**, end equity **$3,555,563.81**,
maximum drawdown **53.700%**, Sharpe **0.695**, annualized return **25.969%**,
and **$7,940.17** in fees across 3,793 orders. The prior missing-price order
errors did not recur. QuantConnect warned about automatic SPY benchmark
subscription, conversion of daily-data market orders to open/close orders, one
sub-minimum single-share rebalance, and overlapping History calls; these do not
invalidate the completed run, though the order timing belongs in any future
matched implementation.

This materially strengthens the direction of the momentum finding: including
point-in-time membership and delistings did not destroy it. It does not validate
the local CAGR magnitude. QuantConnect dynamically selects the top 200 liquid
US equities, while the Yahoo tests use today's small- and mid-cap survivors, and
the periods and execution details differ. The next clean comparison is a common
date range and comparable universe; forward paper observations remain the gate
for deployment claims.

Started from 29 transcripts of a finance YouTuber building an AI stock
predictor. Rebuilt it clean-room: trees not LSTM, 20-day excess-vs-market
cross-sectional labels, walk-forward purge/embargo, monkey baselines,
costs day one, calibration. Then spent months trying to make it work.

ML results: large-cap AUC 0.53 mostly cash. Small-cap AUC 0.55, beats
monkeys 5/8 folds, fold CAGRs -75% to +123% — signal with disqualifying
variance. Mid-cap AUC 0.559 on untouched data — replicates, P&L +4.2%/-0.44.
Insider raw counts hurt twice (0.530->0.524, 0.550->0.533). Conviction
insider (C-suite >$50k first-buy, 458 months) 0.5554 vs 0.5587 — buried.
Long-short Sharpe -0.03 — hedge cuts DD, short book earns nothing.
Ensemble ML+momentum dilutes (0.17 vs 0.40) — ML adds nothing.

Improve loop (budget 20, 8 left): legacy-cache/fixed-ledger holdouts are P5
+6.3%, P8 -2.0%, P9 -2.1%, P12 -2.0%; leak-fixed-v2/shifted-window holdouts
are +25.9%, +17.7%, +17.7%, +17.9%. On the common 2020-02–2025-07 window,
the reversal remains (+5.0%/-3.1%/-3.1%/-3.0% legacy versus
+22.8%/+14.9%/+14.5%/+14.8% v2). V2's tune Sharpes are only 0.30–0.36, so
none would pass the original promotion gate. Verdict: pipeline-sensitive,
not a clean success or failure. Registry retains historical outputs; P1 is
retracted as a stop-logic bug. Earlier vol-monkey attribution is pre-fix.

Momentum (no ML): small tune +9.5%/0.41/-0.31, hold +6.2%/0.20/-0.56. Mid
tune +3.3%/0.15, hold +11.4%/0.35/-0.37. Positive every half both universes.
Multi-asset Faber-style: monthly tune +7.3%/0.82/-0.13 hold +7.4%/0.62/-0.24;
quarterly +5.8/+5.8. Intl sleeve tune +2.9%/0.22 hold +11.2%/0.88/-0.16.
Quality sleeve illustrative +12.7 vs SPY +10.2 (4 decisions). Value on mega
caps: price 2.77x Graham IV, zero trades at 50%, +2.7 vs +8.1 at 0%.

Regime correction: both ML halves were bulls (tune SPY +12.6, holdout
+15.1). Nothing beat buy-hold on return anywhere. Best case is ties on
risk-adjusted. Downgraded bottom line accordingly.

Incidents: slice-CAGR bug inflated momentum holdouts (fixed, rerun);
mixed-vintage data silently dropped SPY 9 months (assert_vintage guard
added everywhere); FMP free key covers mega-cap annuals only, limit<=5,
~10 calls/min, 250/day; SEC EDGAR blocked from here; Stooq bot-blocks;
tune/holdout era labels were wrong for weeks (corrected).

Research: Numerai (+25% 2024, 2.75 Sharpe, JPM $500M — breadth+neutral+
staking); Medallion 66%/yr never down (thousands of short bets, capped
size); GKX trees win R2 0.3-0.4% signals=momentum/liquidity/vol (ours
match); Quantopian 888 strats live correlation ZERO; Stanford 58% die in
3mo; Faber GTAA replicated by us; TSMOM alpha is mostly vol scaling
(indicts P8 partially); value+momentum negatively correlated (our
ensemble failed on two correlated legs); AlphaEvolve/AIDE/DGM taxonomy
built our loop; 4chan sentiment dead both directions (his newest video).

Live: wide-seed0-v1 30 picks -2.7% excess; mom12-1-v1 logging from Sep 30
month-end. QC cloud 252/21 replication completed at +3,455.564% total return,
53.700% max drawdown, and 0.695 Sharpe on its dynamic liquid universe. Compare
a matched period/universe next. LEAN local bulk costs thousands; Docker ready
if ever needed.

Open: paper accumulation, value on paid small-cap data, quality expansion,
matched-universe QC comparison, 8 loop trials banked. File stays open, loop
frozen, index fund winning.

## Prospective paper protocol locked (Sep 14, 2026)

The original `data/paper/log.csv` remains historical. It has no trustworthy
creation timestamps or input/code fingerprints, and its documented next-open
entry rule was not implemented by its grader. It is available through
`paper.py grade --legacy` and is never pooled with forward evidence.

The new `prospective-v2` series starts empty in `data/paper/log_v2.csv`.
Predictions require completed month-end data and clean pipeline code. Each run
writes an exclusive-create JSON manifest with UTC creation time, git state,
raw-input hashes, data date, model backend and versions, feature list,
train/validation ranges, frozen parameters, and every ML and momentum signal.
Log rows carry the manifest hash. Grading verifies the hash and signal fields,
then appends an idempotent result to `grades_v2.csv`.

Execution now matches the stated protocol: entry is the first subsequent
session's open; a stop exits at the first actual close at or below 85% of entry,
including gaps; otherwise the twentieth close exits; and round-trip costs are
charged. IWM uses the same next-open comparison. A full development run in a
temporary ledger produced 14 signals (4 ML, 10 momentum) and passed manifest
verification. The production guard rejected Sep 10 data on Sep 14, so the clean
series still has zero observations.

## Working paper drafted (Sep 14, 2026)

`PAPER.md` turns the audit trail into a manuscript with one narrow central
claim: aggregate predictive accuracy can survive a pipeline correction while
the investable decisions do not. The controlled purge ablation is the primary
result (AUC 0.553 vs 0.551; probability Spearman 0.762; mean selected-name
Jaccard 0.392). Corrected momentum and the QuantConnect run are comparisons and
robustness evidence, not claims of a newly discovered anomaly or matched return
estimates.

The draft declares the work still required before journal submission:
block-bootstrap uncertainty, factor regressions, multiple-testing adjustment,
a matched QuantConnect comparison, selection-stability figures, artifact-level
provenance, and prospective observations. The eight remaining improvement-loop
trials stay banked while this measurement work is open.

## Bootstrap and factor regressions (Sep 14, 2026)

`factor_analysis.py` runs the first two items from PAPER.md Section 9 against
Ken French's public data library (Fama-French 5 + momentum, monthly, through
Jul 2026 — same source and coverage as the earlier "Factor regimes" section).
No rule was retuned; this only adds uncertainty quantification around numbers
already reported. `predictor/factors.py` caches the download; `statsmodels`
added to `requirements.txt` for HAC (Newey-West, 3 lags) standard errors.

**CAPM / FF5+Mom regressions, full sample (185 months, Mar 2011-Jul 2026),
monthly momentum return realized the calendar month after its decision date:**

| Universe | CAPM alpha (ann.) | t | FF5+Mom alpha (ann.) | t | R² | Mom beta | SMB beta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small-cap | +5.8% | 1.09 | +9.0% | **2.14** | 0.65 | +0.43 (t=4.9) | +0.96 (t=6.3) |
| Mid-cap | +1.5% | 0.29 | +0.9% | 0.18 | 0.62 | +0.70 (t=5.7) | +0.68 (t=2.4) |

Small-cap momentum keeps a marginally significant alpha after controlling for
market, size, value, profitability, investment, and the academic momentum
factor itself. Mid-cap's alpha is statistically indistinguishable from zero
once those factors are priced in — most of its apparent edge is loading on
known factors (mainly momentum and size), not something beyond them. Neither
number has a multiple-testing adjustment yet (Section 9 item 3, not run here).

**Moving-block bootstrap, 95% CI, holdout half only (94 months, block=6,
5,000 resamples):**

| Universe | Mean monthly ret | CAGR | Sharpe | Mean monthly excess vs SPY |
|---|---|---|---|---|
| Small-cap | +2.06% [+0.58%, +3.68%] | +21.7% [+1.9%, +47.6%] | 0.78 [0.22, 1.46] | +0.59% [-0.66%, +1.85%] |
| Mid-cap | +2.30% [+0.70%, +4.27%] | +24.7% [+3.7%, +54.5%] | 0.81 [0.27, 1.35] | +0.79% [-0.53%, +2.36%] |

Mean return, CAGR, and Sharpe are positive throughout their intervals in both
universes — the "momentum makes money" result holds up under resampling.
**The excess-vs-SPY interval crosses zero in both universes.** "Momentum vs.
SPY, closed out" above reported the right point estimate and the right
methodology fix (same window, same dates), but a point estimate beating SPY
is not the same claim as beating SPY with statistical confidence — that
section's title overstates what a single-path comparison can close out. The
honest read: small- and mid-cap momentum's outperformance over SPY in this
window is directionally consistent but not distinguishable from zero once
monthly-return sampling noise is accounted for.

## Deflated Sharpe Ratio (Sep 14, 2026)

PAPER.md Section 9 item 3, Bailey & Lopez de Prado (2014). Applies to the
self-improvement loop specifically, not momentum -- momentum was "frozen
before first run... not fitted" (`mom_run.py`'s own docstring), so its
Sharpe isn't the max of a search the way a promoted loop proposal's is.
`deflated_sharpe.py` reran all 12 formal proposals on the tune half of the
legacy (`sc_full`) fold cache -- the pre-registered dates the real promotion
decisions used -- under the current, fully corrected code (this is also what
surfaced and got the `leg_simple` NaN bug fixed above: it computes Sharpe
with `numpy.mean`/`numpy.std`, which don't silently skip NaN the way pandas'
defaults do everywhere else, so a corrupted month showed up as an outright
NaN instead of a quietly-wrong number).

**Under corrected code, the tune-half maximum is now P3-blend** (monthly
Sharpe +0.226, annualized +0.78), not the historically-promoted P5 (+0.697
annualized) -- P3 was never promoted historically despite clearing both the
Sharpe and drawdown bars on record (CAGR 7.2%, DD -22.2%, Sharpe 0.43 in the
original registry), which is a loose end this analysis doesn't resolve: either
an undocumented additional criterion filtered it out, or it was a selection
oversight. DSR is computed on the actual current best (P3) either way, since
that's what a Sharpe-maximizing search would surface today.

Skew and kurtosis of P3's 82-month tune return series: +0.417 and 3.94
(Pearson; normal is 3) -- mildly right-skewed, close to normal tails.

| Trial count | sigma(SR) source | E[max SR \| N] (annualized) | PSR(0), naive | **DSR** |
|---|---|---:|---:|---:|
| N=12, formal loop as run | all 12 tune Sharpes | +0.72 | 0.982 | **0.566** |
| N=12, formal loop | 11 tune Sharpes, P1 artifact excluded | +0.70 | 0.982 | **0.587** |
| N=34, broad (all reported configs) | same 11, held fixed | +0.89 | 0.982 | **0.383** |

The naive PSR(0) -- "is the true Sharpe positive, ignoring that this was
picked as the best of many" -- says 98% confidence. That number is the wrong
one to trust. Once corrected for actually having searched 12 trials, DSR
drops to 0.57-0.59: barely better than a coin flip on whether the loop's best
result reflects real skill rather than the expected maximum of 12 lucky
draws. Widening the search to N=34 (every distinct configuration this
project has reported, formal and informal, not just the loop's own 12) drops
DSR to 0.38 -- *below* even chance, meaning the observed best Sharpe is now
*less* than what you'd expect the best of 34 pure-noise trials to produce.

N=34's sigma(SR) is held at the formal-12 estimate rather than re-measured
from the broader set (gathering monthly return series for every informal
experiment in the file was out of scope here), so that row is a sensitivity
bound, not an independently estimated number -- flagged as such rather than
presented with false precision. The direction is unambiguous either way: the
wider the honestly-accountable search, the less this loop's best result
looks like evidence of skill.

This closes out PAPER.md Section 9 item 3 and lands on the same conclusion
as everything else this session has found under corrected accounting: the
ML side of this project has produced nothing that survives being asked "or
is this just what N trials looks like."
