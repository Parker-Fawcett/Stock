# Current results

This is the maintained product view of the research. `FINDINGS.md` keeps
the full history, including failed tests and superseded numbers.

**Last reconciled:** September 14, 2026  
**Research posture:** promising momentum result; not ready for real capital.

## Verdict board

| Line | Status | Current verdict |
|---|---|---|
| Textbook 12-1 momentum | **SURVIVED — provisional** | Positive in every tested half, resampling-robust (bootstrap CI stays positive). Beats matched-window SPY on point estimate in both halves (small-cap) / holdout (mid-cap), but that margin's 95% CI crosses zero — directional, not statistically confident. Yahoo survivor bias remains. |
| Price-based tree ML | **WEAK / NOT USEFUL** | Ranking survives at mean AUC 0.554, but the portfolio is unstable and weaker than momentum. |
| ML + momentum | **FAILED** | Adding ML reduces CAGR and Sharpe in both halves. |
| Self-improvement overlays | **MIXED / PIPELINE-SENSITIVE / NOT DISTINGUISHABLE FROM LUCK** | On original dates with the fixed ledger, only P5 stays positive. With leak-fixed probabilities and shifted folds, all four are strongly positive later—but none passes that cache's tune gate. Deflated Sharpe Ratio: the historical loop's best result is a coin flip against 12-trial luck (DSR 0.57–0.59) and below even chance against the project's historical 34-configuration search (DSR 0.38). A later four-trial exploratory round found P15 consensus, but it has no untouched holdout. |
| Insider signals | **FAILED** | Raw and conviction-filtered Form 4 features do not improve the model. |
| Multi-asset trend | **SURVIVED — defensive** | Trails SPY on raw return, with smoother drawdowns and better measured Sharpe. |
| Graham value, mega caps | **FAILED FOR THIS UNIVERSE** | Few or no qualifying trades; paid point-in-time small-cap fundamentals would be a different test. |
| Paper portfolios | **LOCKED / AWAITING FIRST RUN** | The provenance-locked `prospective-v2` series starts at the next month end. Existing rows remain historical and separate. |
| QuantConnect replication | **SURVIVED — DYNAMIC AND FIXED UNIVERSES** | Dynamic-universe run: +3,455.56% total return, 25.97% CAGR, 53.70% max DD, 0.695 Sharpe. Fixed 99-name matched-universe run: +2,557.48%, 23.62% CAGR, 48.10% max DD, 0.690 Sharpe versus local 21.65%/37.15%/0.85. |

## Current numbers

All local returns below use the common simple-return and turnover accounting.
The momentum, ensemble, and multi-asset runners call the same weight-turnover
implementation; the ensemble uses the same frozen 252/21 formation rule.

| Strategy | Tune CAGR | Tune Sharpe | Tune DD | Holdout CAGR | Holdout Sharpe | Holdout DD |
|---|---:|---:|---:|---:|---:|---:|
| Small-cap momentum | **17.8%** | 0.84 | -26.5% | **21.7%** | 0.78 | -37.1% |
| Mid-cap momentum | 9.8% | 0.54 | -27.8% | **24.7%** | 0.81 | -31.0% |
| ML alone, ensemble window | -0.1% | 0.13 | -61.5% | 8.9% | 0.38 | -43.3% |
| Momentum, ensemble window | 17.3% | 0.86 | -25.3% | 20.3% | 0.71 | -37.4% |
| 50/50 ML + momentum | 9.8% | 0.60 | -42.3% | 16.5% | 0.63 | -38.2% |
| Multi-asset trend | 8.0% | 0.95 | -13.2% | 8.0% | 0.74 | -23.3% |

Matched-window buy-and-hold SPY returned 11.9% and 16.9% in the two
momentum halves. The small-cap momentum result is ahead by 5.9 and 4.8
percentage points respectively. That comparison uses the same dates and
annualization convention.

A moving-block bootstrap (6-month blocks, 5,000 resamples) on the holdout
half's monthly excess return over SPY gives a 95% CI of [-0.7%, +1.9%] per
month for small-cap and [-0.5%, +2.4%] for mid-cap — both cross zero. Mean
monthly return, CAGR, and Sharpe all stay positive throughout their own
intervals; only the SPY comparison specifically isn't statistically
distinguishable from zero at this sample size. CAPM/FF5+Mom regressions add
one more qualifier: small-cap keeps a marginally significant alpha (+9.0%/yr,
\(t\)=2.14) after controlling for standard factors including momentum itself,
but mid-cap's alpha (+0.9%/yr, \(t\)=0.18) does not — most of mid-cap's edge
is factor exposure, not something beyond it. Full tables in `FINDINGS.md` and
`PAPER.md` Section 6.5.

## Self-improvement replay sensitivity

The two available caches answer different questions. The legacy cache keeps
the original promotion dates but contains pre-fix, leak-affected probabilities.
The v2 cache has the corrected probability pipeline, 26 instead of 24 folds,
one fewer ticker, and a shifted split. Values below are holdout CAGR; the common
window controls for dates but cannot isolate the purge, fold, data-vintage, and
universe changes from one another.

| Proposal | Legacy, original half | v2, shifted half | Legacy, common dates | v2, common dates |
|---|---:|---:|---:|---:|
| P5 lower gate | +6.3% | +25.9% | +5.0% | +22.8% |
| P8 inverse-vol weights | -2.0% | +17.7% | -3.1% | +14.9% |
| P9 top 30 | -2.1% | +17.7% | -3.1% | +14.5% |
| P12 top 40 | -2.0% | +17.9% | -3.0% | +14.8% |

The common period is February 28, 2020 through July 9, 2025. Because the
reversal remains on identical dates, the extra 2025–2026 market strength does
not explain it. A controlled refit on one current data snapshot then held the
folds, test rows, model, and seed constant and changed only purge logic:

| Proposal | Legacy purge holdout | Corrected purge holdout |
|---|---:|---:|
| P5 | +11.9% | +25.9% |
| P8 | +6.7% | +17.7% |
| P9 | +6.6% | +17.7% |
| P12 | +6.4% | +17.9% |

The purge repair itself causes a large positive shift, even though aggregate
AUC barely changes (0.553 to 0.551). Monthly selected-name overlap is only
about 39%. Tune Sharpes stay clearly under the original gate in five of six
cells; the corrected-purge P12 cell (0.41) lands almost exactly on it. Still
diagnostics rather than promotions. The original legacy cache cannot be fully
reconstructed because it has no data/code manifest; new caches now record one.

### Exploratory rank and consensus round

Four additional proposals were preregistered in code and evaluated on the
already inspected tune half of `sc_full_v2` (September 30, 2013 through
February 6, 2020). The tool marks them `exploratory-tune` and prevents their
promotion. The same runner recomputed a pure-momentum control using identical
stops, costs, dates, and portfolio accounting.

| Rule | CAGR | Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| Momentum control | 11.33% | 0.762 | -21.6% |
| P13 ML rank, no probability gate | 13.52% | 0.786 | -23.1% |
| P14 75% momentum / 25% ML rank | 12.05% | 0.788 | -21.5% |
| P15 top-40%-in-both consensus | **15.32%** | **0.903** | -22.2% |
| P16 P14 with inverse-volatility weights | 10.80% | 0.749 | -22.0% |

P15 is the candidate worth freezing: it improves CAGR by 3.99 percentage
points and Sharpe by 0.141 relative to the same-run momentum control, with a
0.6-point worse maximum drawdown. This is search output, not new evidence.
The period, universe, and model probabilities were already known when these
rules were designed, four alternatives were tried, and no untouched holdout
was opened. The lifetime budget is now 16/20 spent; four trials remain banked.

## What the evidence supports

The defensible research claim is narrow: cross-sectional momentum appears
stronger and more useful than the tested machine-learning, insider, value,
and ensemble variants. The ML rank contains a small statistical signal,
but it has not established a stable portfolio improvement. The promoted
overlays reverse sign between the legacy and corrected probability caches even
on common dates. That is evidence of pipeline sensitivity, while the strong v2
period is a real positive observation that warrants continued prospective
tracking. A Deflated Sharpe Ratio check adds a harder ceiling on the loop
specifically: its best historical result is statistically indistinguishable
from what 12 trials of pure noise would be expected to produce, and falls
below chance once every configuration this project has tried is counted.

The local stock lists contain companies that survive today. QuantConnect's
point-in-time fundamental universe and delisting-aware data provide a useful
independent check: the corrected 252/21 run remained strongly positive and no
longer produced the prior missing-price order errors. That makes a catastrophic
survivorship explanation unlikely.

The dynamic cloud magnitude is not a like-for-like estimate of the Yahoo
result: it uses the top 200 liquid US equities at each selection date. A second
cloud run fixes the universe to the same 99 current-constituent survivors and
the same broad 2011-2026 window as the local small-cap comparison. Its 23.62%
CAGR is close to the local 21.65%, while Sharpe is lower (0.69 vs 0.85) and
drawdown is worse (-48.10% vs -37.15%). The cloud engine converted daily market
orders to market-on-close/open fills, charged its IB fee model instead of the
local fixed 25 bps, adjusted several symbols to their mapping/data start dates,
and included an incomplete final June holding interval. Treat the CAGR
agreement as encouraging data/execution convergence, not exact replication.

## Promotion gate

Momentum can move from **provisional** to **validated for deployment** only after:

1. the remaining monthly-return differences between the completed fixed-
   universe cloud run and Yahoo implementation are reconciled, including fill
   timing, final-period truncation, and fees; and
2. prospective paper observations accumulate under the frozen model version.

Until those pass, this repository is a research product and audit trail,
not investment advice or a deployable trading system.

## Prospective protocol

`paper.py predict` accepts completed month-end data and clean pipeline code.
Each run writes a unique manifest containing its UTC creation time, git state,
input-file hashes, model backend, feature list, train/validation dates,
parameters, and complete signal set. Paper rows store the manifest hash;
grading verifies both the hash and each signal before use.

The clean series implements the documented execution rule: enter at the next
session's open, observe 20 trading sessions, exit at the first actual close at
or below the 15% stop or at the final close, and charge round-trip costs.
Results persist separately in `data/paper/log_v2.csv` and
`data/paper/grades_v2.csv`. No off-cycle September observation was inserted
while testing this change.

## Publication status

`PAPER.md` is the first complete working-paper draft. Its central claim is the
controlled result, not a claim of new alpha: correcting the purge changes AUC
from 0.553 to 0.551 while mean selected-name overlap is only 0.392. The draft
labels the local momentum magnitudes as provisional and separates the dynamic
survivorship check from the completed fixed-universe magnitude comparison.
The novelty review now also separates the broad known phenomenon from this
paper's narrower contribution. Predictive multiplicity, underspecification,
leakage-controlled financial ML, and seed-driven allocation instability all
have prior literature. A closely related August 2026 preprint also audits
validation-selected decisions, rank transfer, costs, and execution in index
ETFs. The potentially original result is the controlled
propagation of one label-purge repair, with data, folds, test rows, model, and
seed fixed, from nearly unchanged AUC to 0.392 selected-name overlap and changed
portfolio conclusions. `PAPER.md` includes a closest-prior-work table and does
not claim discovery of momentum, predictive multiplicity, or allocation
instability in general.
Remaining work is monthly return reconciliation, selection-stability figures,
artifact provenance, and prospective evidence.
