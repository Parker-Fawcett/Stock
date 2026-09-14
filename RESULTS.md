# Current results

This is the maintained product view of the research. `FINDINGS.md` keeps
the full history, including failed tests and superseded numbers.

**Last reconciled:** September 13, 2026  
**Research posture:** promising momentum result; not ready for real capital.

## Verdict board

| Line | Status | Current verdict |
|---|---|---|
| Textbook 12-1 momentum | **SURVIVED — provisional** | Positive in every tested half. Small-cap beats matched-window SPY in both halves; mid-cap beats it in holdout. Yahoo survivor bias remains. |
| Price-based tree ML | **WEAK / NOT USEFUL** | Ranking survives at mean AUC 0.554, but the portfolio is unstable and weaker than momentum. |
| ML + momentum | **FAILED** | Adding ML reduces CAGR and Sharpe in both halves. |
| Self-improvement overlays | **MIXED / PIPELINE-SENSITIVE** | On original dates with the fixed ledger, only P5 stays positive. With leak-fixed probabilities and shifted folds, all four are strongly positive later—but none passes that cache's tune gate. |
| Insider signals | **FAILED** | Raw and conviction-filtered Form 4 features do not improve the model. |
| Multi-asset trend | **SURVIVED — defensive** | Trails SPY on raw return, with smoother drawdowns and better measured Sharpe. |
| Graham value, mega caps | **FAILED FOR THIS UNIVERSE** | Few or no qualifying trades; paid point-in-time small-cap fundamentals would be a different test. |
| Paper portfolios | **LIVE / EARLY** | Logging is implemented; there are not enough clean forward observations to judge performance. |
| QuantConnect replication | **SURVIVED — different universe** | The guarded 252/21 cloud run completed without the prior missing-price order errors: +3,455.56% total return, 25.97% CAGR, 53.70% max drawdown, and 0.695 Sharpe. |

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
about 39%. Neither ablation passes the original tune gate, so these are
diagnostics rather than promotions. The original legacy cache cannot be fully
reconstructed because it has no data/code manifest; new caches now record one.

## What the evidence supports

The defensible research claim is narrow: cross-sectional momentum appears
stronger and more useful than the tested machine-learning, insider, value,
and ensemble variants. The ML rank contains a small statistical signal,
but it has not established a stable portfolio improvement. The promoted
overlays reverse sign between the legacy and corrected probability caches even
on common dates. That is evidence of pipeline sensitivity, while the strong v2
period is a real positive observation that warrants continued prospective
tracking.

The local stock lists contain companies that survive today. QuantConnect's
point-in-time fundamental universe and delisting-aware data provide a useful
independent check: the corrected 252/21 run remained strongly positive and no
longer produced the prior missing-price order errors. That makes a catastrophic
survivorship explanation unlikely.

The cloud magnitude is not a like-for-like estimate of the Yahoo result. It
uses the top 200 liquid US equities at each selection date, while the local
runs use today's small- and mid-cap constituent lists. Its January 2011 through
June 17, 2026 period and QuantConnect execution model differ too. Treat the
cloud result as confirmation of direction and fragility, not as a direct
replacement CAGR.

## Promotion gate

Momentum can move from **provisional** to **validated for deployment** only after:

1. the cloud and Yahoo implementations are compared over a common period and
   comparable universe; and
2. prospective paper observations accumulate under the frozen model version.

Until those pass, this repository is a research product and audit trail,
not investment advice or a deployable trading system.
