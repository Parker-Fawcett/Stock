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
    always correct.)
12. **Ensemble (50% ML-wide + 50% momentum, zero new params).** Tune:
    ml -2.2% (Sharpe -0.10), mom +8.7% (+0.40), ens +3.1% (+0.17). Bar
    was Sharpe > 0.41 with DD > -0.31. Failed — averaging dilutes.
    The ML adds nothing to momentum. Buried as a signal; kept running
    in paper only as the losing side of the bet.
13. **Self-improvement loop, round one (6 proposals, tune-half).**
    P1 printed Sharpe 2.9/CAGR 2900% — investigated, not celebrated:
    nine idiosyncratic biotech/crypto doubles in 1–2 name books during a
    bull market (CLSK verified tick-by-tick as genuine). Numeric pass
    rejected openly as lottery concentration; protocol gap noted.
    P5-gate25 (wide + 0.25 gate, 81/82 months invested, avg 9.4 names,
    no month above +25%) passed clean: tune Sharpe 0.68, DD -0.26.
    Promoted once to holdout: **CAGR +12.7%, DD -0.35, Sharpe 0.41**
    through COVID, 2022 bear, and recovery at 25bps. First ML config
    with a clean stress-period holdout. Budget: 14/20 left.
14. **Self-improvement round two (P7 liquidity, P8 vol scaling).**    P7: Sharpe 0.56 pass, DD -0.361 — missed the -0.35 bar by 0.011.
    Dead by the letter of the law, stated openly. P8 (per-name inverse-
    vol sizing): tune Sharpe 0.63, DD -0.27. Promoted once: holdout
    **CAGR +6.1%, DD -0.39, Sharpe 0.22**. Positive through stress,
    shrunk as honest results do (DD breached -0.35 out-of-sample, which
    is itself a finding about bar-setting). Budget: 12/20 left.
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
    Budget: 10/20 left.
21. **Self-improvement round four (P11 short holds, P12 max breadth).**
    P11-hold10: Sharpe -0.22, dead — monthly decisions with 10d holds
    double turnover and go stale faster; direction was wrong. P12-top40:
    tune Sharpe 0.57, DD -0.31, breadth verified (81/82 months, avg
    11.3 qualifiers — cap rarely binds, so near-twin of P8, which
    corroborates rather than duplicates). Promoted once: holdout
    **+3.5%, DD -0.41, Sharpe 0.14**. Fourth straight decaying
    promotion (+12.7 → +6.1 → +4.3 → +3.5). Budget: 8/20 left.
18. **Multi-asset trend, quarterly rebalance (Faber's turnover note).**
    Tune +5.8% (Sharpe 0.58, DD -0.13), holdout +6.9% (Sharpe 0.60,
    DD -0.27). Monthly version: tune +7.3% (0.82), holdout +7.4%
    (0.62). SPY buy-hold same window +14.5% — strategy trails on
    return, wins on risk-adjusted calm. (Also fixed a slice-CAGR bug
    that had inflated momentum holdout prints; Sharpe/DD unaffected.)
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

Public-data cross-sectional rank predicts weakly (AUC ~0.56) and profits
~never after risk. The only strategy positive in every half tested is
textbook momentum with no ML. Independent convergence with the source
project's verdict, tighter bounds, three universes.

## Regime correction (the nessessary uncomfortable table)

Both ML halves were bull markets, not one calm + one stress:
tune 2013–19 SPY +12.6%/IWM +8.9%; holdout 2019–25 SPY +15.1%/IWM +7.4%
(crashes inside, V-recoveries after). Consequence: NO long-only config
in this project beat buy-and-hold SPY on return in either half. P5's
+12.7% trails SPY's +15.1% on the same window. Best case anywhere is a
tie on risk-adjusted terms (multi-asset Sharpe 0.82 vs SPY ~0.8).
Momentum's wins are real but relative, not absolute. The file's claim
is hereby downgraded from 'weak signal, unharvestable risk' to 'weak
signal that trails the index, with calmer variants'.

## After everything

Started from 29 of his videos and one question: can his AI stock
predictor be rebuilt better. Ended with a fuller answer than his.

What was tried, in order: trees-first rebuild, purge/embargo validation,
monkey baselines, costs from day one, three universes (large/small/mid),
raw and conviction insider data, risk overlays, long-short, textbook
momentum, Graham value on real financials, French factor regimes, live
paper trading. Twelve numbered experiments, each with a pre-committed
bar where selection was involved.

What survived: a 0.56 AUC rank signal that replicates on untouched data
but never survives risk; textbook 12-1 momentum, positive in all four
halves tested across two universes (small hold +6.2%, mid hold +11.4%),
plus multi-asset trend (+6.3%/+7.3% halves, Sharpe 0.82/0.54, DD under
-0.24 — best risk-adjusted in the project); a live paper log grading
both monthly.

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

What runs on its own now: monthly paper log (ML + momentum sleeves),
monthly momentum sleeve, and the value engine waiting on small-cap data
that costs money everywhere checked.

What would reopen the case: paid point-in-time fundamentals on small
caps, a new data regime (satellite/credit-card class, with budget), or
months of live log contradicting the backtests. Until one arrives, the
file is closed and the index fund wins.

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
Our weakest links (hand execution realism, survivor universes) fixed
free — migration candidate.

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

## Decay meta-analysis (loop paused, 10 trials banked)

Promoted holdout CAGRs decline in promotion order: P5 +12.7%, P8 +6.1%,
P9 +4.3%. Tune Sharpes decline too (0.68/0.63/0.59). Every holdout DD
breached its tune-set bar. Reading: each promotion mines a thinner vein
and every round is another selection bite the budget only weakly prices.
The loop worked (three clean promotions, all documented) and is now
frozen until genuinely new data or a new game — spending the rest on
this universe would be the slow version of the sin it guards.

## Dump (everything, unstructured, Sep 2026)

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

Improve loop (budget 20, 10 left): P5-gate25 promoted (+12.7%/-0.35/0.41
holdout through COVID+2022). P8-volscale promoted (+6.1%/-0.39/0.22).
P9-top30 promoted (+4.3%/-0.40/0.16). P1 lottery artifact rejected openly
(Sharpe 2.9 from 9 biotech/crypto doubles in 1-2 name books — real prices,
verified CLSK tick by tick, meaningless stats). P7 died by 0.011 on the DD
bar. P10, P2-P4, P6 dead on tune. Vol-monkeys +3%/0.22: sizing is a third
of P8, selection the rest.

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
month-end. QC cloud algorithm written (qc_momentum.py, UNTESTED) for
survivor-free validation — paste into free QuantConnect account, compare
overlap first. LEAN local bulk costs thousands; Docker ready if ever needed.

Open: paper accumulation, value on paid small-cap data, quality expansion,
LEAN/QC migration results, 10 loop trials banked. File stays open, loop
frozen, index fund winning.
