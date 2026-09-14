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

## What the evidence supports

The defensible research claim is narrow: cross-sectional momentum appears
stronger and more useful than the tested machine-learning, insider, value,
and ensemble variants. The ML rank contains a small statistical signal,
but it has not improved portfolio construction.

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
