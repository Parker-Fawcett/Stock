# When AUC Survives but Portfolios Do Not

## Leakage, accounting errors, and selection instability in a retail-scale machine-learning equity study

**Parker Fawcett**  
Working paper — September 2026  
Draft status: methods and historical results; prospective evidence pending

## Abstract

Machine-learning equity studies often report predictive discrimination and
portfolio performance as though improvements in one necessarily support the
other. This paper presents a reproducible audit of that assumption in a
retail-scale U.S. equity experiment. A tree classifier ranks stocks by the
probability of finishing in the top cross-sectional quintile of 20-trading-day
return relative to a market benchmark. The study reconstructs the research
pipeline after finding validation-label leakage, incorrect aggregation of
security-level log returns, incomplete turnover measurement, and risk controls
that changed concentration rather than exposure. A controlled refit changes
only the purge rule while holding the price snapshot, universe, folds, labels,
test observations, model code, and random seed fixed. Aggregate test AUC barely
changes, from 0.553 to 0.551, but the probability-rank correlation is 0.762 and
the mean Jaccard overlap of selected portfolios is only 0.392. Portfolio
conclusions move materially even though headline classifier accuracy does not.
A simpler 12-minus-1-month momentum rule is more stable: after correcting the
ledger, it produces positive results in both halves of two current-constituent
universes and outperforms matched-window SPY buy-and-hold in three of four
universe-period comparisons. A separate point-in-time, dynamic-universe
QuantConnect implementation remains strongly positive, although it is not a
matched estimate of the local backtest. The evidence supports a methodological
conclusion rather than a claim of newly discovered alpha: portfolio selection
near a model cutoff can be unstable even when aggregate AUC appears robust,
and implementation audits can dominate model choice in small quantitative
research programs.

**Keywords:** machine learning, equity selection, backtest overfitting,
look-ahead bias, momentum, reproducibility, portfolio accounting

## 1. Introduction

Financial machine learning has two evaluation problems. A statistical model
must generalize to future observations, and the decisions made from its output
must survive portfolio construction, trading costs, and market frictions. A
model can perform acceptably on a global classification statistic while being
economically unstable at the narrow cutoff where securities enter a portfolio.
Conversely, a correct signal can be misrepresented by an incorrect ledger.

This study documents both failures in one end-to-end equity research project.
The project began as an independent reconstruction of a publicly documented
retail stock-prediction system. It eventually tested price-based tree models,
insider features, long-short portfolios, risk overlays, ensembles, fundamental
value, multi-asset trend, and cross-sectional momentum. Negative results and
superseded results were retained. An external review then identified errors
that required the main strategies to be replayed without retuning.

The most useful result comes from a controlled purge ablation. The original
walk-forward routine approximated a 20-trading-day label horizon with calendar
days and failed to purge validation labels that crossed the test boundary.
Validation outcomes therefore entered calibration or early stopping even when
training observations were separated from the test period. Repairing that rule
on an otherwise identical data and model pipeline changes aggregate AUC by only
0.002. It nevertheless replaces most names in the investable portfolio each
month and changes the reported returns of previously selected overlays.

This distinction matters because AUC averages pairwise ranking quality across
all positive and negative observations. A portfolio uses only a small tail of
that ranking and often imposes a probability threshold as well. Two models can
therefore have similar global discrimination while disagreeing around the
investment boundary. In this experiment, mean selected-name overlap is about
39%, despite similar AUC.

The study makes three contributions. First, it gives a concrete, reproducible
example in which a valid correction leaves headline predictive accuracy nearly
unchanged while materially changing portfolio membership and economic results.
Second, it separates model error from accounting error through controlled
replays. Third, it compares the machine-learning pipeline with a frozen,
textbook momentum rule and with a dynamic point-in-time cloud implementation.
The comparison shows why a known simple signal can be a more demanding baseline
than a weakly informative machine-learning model.

The paper does not establish a deployable trading strategy. The local equity
universes use current constituents, the cloud comparison uses a different
liquidity-screened universe, formal factor-adjusted inference remains to be
completed, and the provenance-locked prospective series has not produced its
first observation. Those limits define the remaining research program.

## 2. Related literature

The project sits at the intersection of empirical asset pricing, financial
machine learning, momentum, and research-reliability studies.

Gu, Kelly, and Xiu compare machine-learning methods for expected-return
prediction and find that trees and neural networks can extract nonlinear
interactions from large characteristic panels. They also identify momentum,
liquidity, and volatility among the dominant signals. Their institutional-scale
setting—roughly 30,000 stocks and hundreds of predictors—is much broader than
the present experiment and provides a useful boundary on what a small public-
data system should be expected to reproduce [1].

Cross-sectional momentum is a well-established benchmark. Jegadeesh and Titman
document continuation among past winners over intermediate holding horizons
[2]. The present paper does not claim momentum as a new anomaly. It uses a
12-minus-1-month winner portfolio as a transparent control against which the
more complicated model must add value.

The risk of false discovery rises when many strategies are inspected. Harvey,
Liu, and Zhu argue that conventional significance thresholds are inadequate in
a literature containing hundreds of tested factors [3]. Bailey and López de
Prado develop the Deflated Sharpe Ratio to adjust an observed Sharpe ratio for
selection bias, non-normal returns, and the number of trials [4]. This project
keeps a lifetime proposal budget and an archive of failures, but the final
submission will also report a formal multiple-testing adjustment.

The present contribution differs from a factor-discovery paper. It asks how
model-validation details propagate through a selection threshold into realized
portfolio decisions. That focus also motivates the public audit trail: incorrect
results remain visible with explicit supersession labels instead of disappearing
from the record.

## 3. Research questions and hypotheses

The historical analysis is organized around four questions.

**RQ1.** Does correcting label purging materially change aggregate predictive
discrimination?

**H1.** The corrected and legacy pipelines have similar out-of-fold AUC.

**RQ2.** Can similar aggregate discrimination conceal unstable investable
rankings?

**H2.** Selected-name overlap is materially lower than the correlation of the
full probability rankings.

**RQ3.** Does the tested machine-learning rank improve a simple momentum
portfolio?

**H3.** A 50/50 machine-learning and momentum ensemble does not improve both
CAGR and Sharpe relative to momentum alone on the same periods.

**RQ4.** Does the direction of the local momentum result survive removal of the
current-constituent universe restriction?

**H4.** A frozen 252/21 momentum rule remains positive in a dynamic,
point-in-time U.S. equity universe.

These hypotheses describe the recorded experiments; they are not newly
preregistered claims. The separate `prospective-v2` protocol is the project's
first immutable forward test.

## 4. Data

### 4.1 Local equity panels

Local experiments use adjusted daily OHLCV data downloaded from Yahoo and
cached by ticker. The small-cap sample contains 100 names sampled from the
September 2026 S&P SmallCap 600 constituent list, with `LEG` subsequently
excluded because the endpoint returned only six stale observations. The
mid-cap replication contains 100 names sampled from the September 2026 S&P
MidCap 400 list. IWM supplies the market-relative label and benchmark for both
the small-cap model and the mid-cap momentum results reported in Table 3. An
earlier, separate mid-cap machine-learning generalization test (documented in
`FINDINGS.md`, not reported in this paper) used MDY as its market-relative
label; it is not the source of any table here. SPY is used for matched-window
buy-and-hold comparisons.

These lists are known today and therefore contain survivorship and membership
look-ahead bias when projected backward. Local return magnitudes are
provisional. The fixed lists remain useful for controlled comparisons in which
the same observations and securities appear on both sides.

The main momentum panel covers 188 monthly decisions from February 28, 2011,
through September 10, 2026. It is split into two equal 94-month halves, with
the boundary between November 30 and December 31, 2018. The terms *tune* and
*holdout* are retained from the experiment log, although the paper treats the
second half cautiously because the wider project inspected it during later
development.

### 4.2 Dynamic-universe cloud check

The QuantConnect implementation selects, at each universe update, the 200 most
liquid U.S. equities that have fundamental data, price above $5, and daily
dollar volume above $500,000. QuantConnect's Security Master tracks splits,
dividends, delistings, mergers, and ticker changes, while dynamic universe
selection reduces the selection bias inherent in a current-constituent list
[5]. The cloud experiment uses QuantConnect data and its Interactive Brokers
brokerage model.

The cloud universe, dates, order timing, and fee model differ from the local
test. It is a directional external check rather than a matched replication.

## 5. Methods

### 5.1 Prediction target and features

For security \(i\) on decision date \(t\), the target is an indicator for
whether its 20-trading-day forward log return in excess of the market lies in
the top cross-sectional quintile:

\[
y_{i,t}=\mathbb{1}\left(r_{i,t\rightarrow t+20}
-r_{m,t\rightarrow t+20}\text{ is in the top 20% at }t\right).
\]

The eleven price and volume features are trailing 1-, 5-, and 20-day log
returns; 20-day simple momentum; 20-day return volatility; daily range and its
20-day mean; volume relative to its 20-day mean; 14-day RSI; and price gaps
relative to 10- and 50-day moving averages. The feature set intentionally stays
small and public.

The classifier is a gradient-boosted decision-tree model. The implementation
uses LightGBM when installed and otherwise uses scikit-learn's histogram
gradient boosting with isotonic calibration. A saved result must therefore
record the backend and software versions; the newest cache and prospective-run
formats do so.

### 5.2 Walk-forward evaluation

Each fold uses approximately three years of training data, six months of
validation, and six months of test data. Test windows advance sequentially.
Because the target spans 20 trading sessions, a row's label end is computed
from the actual twentieth subsequent observation for that ticker.

The corrected split removes a training row if its label ends inside or near the
validation period and removes a validation row if its label ends inside or near
the test period. A 20-calendar-day embargo is added beyond the measured label
end. The test set is never used for calibration or early stopping.

The legacy routine made two different choices: it approximated the label end
with calendar time and purged training observations only. The controlled
ablation reconstructs that legacy routine without changing any other component.

### 5.3 Controlled purge ablation

Both ablation arms are refit on one frozen current price snapshot. They share:

- all 26 fold boundaries;
- identical training candidates before purging;
- identical test rows, labels, closing prices, and security universe;
- the same feature code, classifier implementation, backend, and seed; and
- the same portfolio and accounting code.

Only the split-purging rule differs. This design identifies the effect of the
purge repair within this pipeline. It does not identify a general causal effect
for other models or datasets.

We report pooled out-of-fold AUC, Pearson and Spearman probability association,
per-date rank association, and selected-name Jaccard overlap. For selected sets
\(A_t\) and \(B_t\), overlap is

\[
J_t=\frac{|A_t\cap B_t|}{|A_t\cup B_t|}.
\]

The selection rule takes up to 20 names whose predicted probability is at
least 0.25. This is the economically relevant tail used by the replayed
overlays.

### 5.4 Portfolio accounting

All corrected local strategies use simple security returns and compound
portfolio equity as \(E_t=E_{t-1}(1+R_t)\). An equal-weight portfolio return is
the arithmetic mean of constituent simple returns. This replaces the earlier
practice of averaging log returns and exponentiating, which does not represent
an equal-weight rebalanced portfolio.

Turnover is one half of the \(L_1\) distance between old and new security
weights:

\[
T_t=\frac{1}{2}\sum_i |w_{i,t}-w_{i,t-1}|.
\]

The local momentum study charges 25 basis points per unit of turnover. Risk
scaling changes total invested weight and holds the remainder in cash. Stops,
where used by an overlay, exit at the first observed close beyond the threshold
and use that actual close rather than the threshold price. Maximum drawdown is
computed from an initial equity value of 1.0 so that an immediate first-period
loss is counted.

### 5.5 Momentum and ensemble baselines

The frozen cross-sectional momentum rule ranks securities monthly by return
from 252 sessions before the decision to 21 sessions before it. It holds the
top decile, subject to at least five names, for 21 trading sessions with equal
weights. The rule is not fitted.

The ensemble averages the monthly returns of a frozen machine-learning book and
the momentum book at 50/50 weight. It introduces no optimized mixing parameter.
The relevant test is whether it improves both return and risk-adjusted return
relative to momentum alone on identical dates.

### 5.6 Performance measures

The paper reports compound annual growth rate, monthly-return Sharpe ratio
annualized by \(\sqrt{12}\), maximum drawdown, turnover, and exposure. The
historical tables do not yet report confidence intervals or factor alpha. Those
analyses are required before journal submission and are listed in Section 9.

## 6. Results

### 6.1 AUC stability conceals portfolio instability

Table 1 presents the central controlled result.

**Table 1. Controlled purge ablation**

| Measure | Legacy purge | Corrected purge | Cross-pipeline agreement |
|---|---:|---:|---:|
| Pooled out-of-fold AUC | 0.553 | 0.551 | — |
| Probability Pearson correlation | — | — | 0.762 |
| Probability rank correlation | — | — | 0.762 |
| Mean selected-name Jaccard | — | — | 0.392 |

The headline AUC declines by only 0.002 after the repair. The portfolios are
far less stable: at the threshold and top-20 cutoff, only about 39% of the
union of selected names appears in both arms on an average decision date. The
result supports H1 and H2. Aggregate classification quality is insufficient to
establish decision stability.

Previously promoted portfolio overlays also change materially, as shown in
Table 2. None of the two ablation arms passes the original tuning gate, so these
results are diagnostics and must not be interpreted as retroactive strategy
promotions.

**Table 2. Previously selected overlays under the controlled purge ablation**

| Overlay | Legacy tune | Legacy holdout | Corrected tune | Corrected holdout |
|---|---:|---:|---:|---:|
| P5: lower probability gate | -1.5% / 0.03 / -52.2% | +11.9% / 0.57 / -35.6% | +3.8% / 0.30 / -37.5% | +25.9% / 0.83 / -41.6% |
| P8: inverse-volatility weights | +1.1% / 0.16 / -50.6% | +6.7% / 0.40 / -39.4% | +4.6% / 0.36 / -35.6% | +17.7% / 0.65 / -40.8% |
| P9: top 30 | +1.2% / 0.16 / -50.4% | +6.6% / 0.40 / -39.7% | +4.6% / 0.35 / -34.9% | +17.7% / 0.65 / -40.8% |
| P12: top 40 | +1.2% / 0.16 / -50.4% | +6.4% / 0.39 / -39.7% | +4.7% / 0.36 / -34.5% | +17.9% / 0.66 / -40.8% |

*Cells report CAGR / annualized monthly Sharpe / maximum drawdown. “Tune” and
“holdout” describe the project's recorded split; neither ablation clears the
original tune Sharpe requirement.*

The direction of the holdout difference is unexpectedly favorable to the
corrected purge. That observation does not mean leakage was conservative in
general. Purging changes fitted rankings, and a single realized market path can
reward either set. The identified finding is instability, not the sign of the
change.

### 6.2 Momentum is stronger than the machine-learning portfolio

Correcting the equal-weight return ledger materially raises the measured
momentum result without changing its rule, dates, or securities. The old
formula is reproduced on the same observations, which attributes the change to
accounting rather than market-data drift.

**Table 3. Frozen 12-minus-1-month momentum after ledger correction**

| Universe | Tune CAGR | Tune Sharpe | Tune max DD | Holdout CAGR | Holdout Sharpe | Holdout max DD |
|---|---:|---:|---:|---:|---:|---:|
| Small cap | 17.8% | 0.84 | -26.5% | 21.7% | 0.78 | -37.1% |
| Mid cap | 9.8% | 0.54 | -27.8% | 24.7% | 0.81 | -31.0% |

Matched-window SPY buy-and-hold returns 11.9% in the first half and 16.9%
in the second. Small-cap momentum exceeds SPY by 5.9 and 4.8 percentage
points, respectively. Mid-cap momentum trails by 2.1 points in the first half
and exceeds SPY by 7.8 points in the second. These comparisons use identical
decision dates and annualization conventions, but the constituent lists still
contain survivorship bias.

### 6.3 Adding the machine-learning leg dilutes momentum

Table 4 compares the model, momentum, and their equal-weight ensemble on the
same window.

**Table 4. Machine learning, momentum, and the fixed 50/50 ensemble**

| Strategy | Tune CAGR | Tune Sharpe | Tune max DD | Holdout CAGR | Holdout Sharpe | Holdout max DD |
|---|---:|---:|---:|---:|---:|---:|
| Machine learning | -0.1% | 0.13 | -61.5% | 8.9% | 0.38 | -43.3% |
| Momentum | 17.3% | 0.86 | -25.3% | 20.3% | 0.71 | -37.4% |
| 50/50 ensemble | 9.8% | 0.60 | -42.3% | 16.5% | 0.63 | -38.2% |

The ensemble reduces CAGR and Sharpe relative to momentum alone in both halves
and increases maximum drawdown in the first. H3 is supported for this fixed
comparison. The model's weak discrimination does not translate into incremental
portfolio value.

### 6.4 Dynamic-universe QuantConnect check

The corrected QuantConnect strategy applies the same 252/21 momentum formation
rule to a dynamic top-200 liquid U.S. equity universe. The cloud result reports
3,455.56% total return, 25.97% annualized return, 53.70% maximum drawdown, and
a 0.695 Sharpe ratio. It completes without the missing-price order errors seen
in an earlier implementation. H4 is supported: the direction remains positive
when the current-constituent list is removed.

This result is intentionally not placed beside the local CAGRs as a direct
replication. Its dynamic universe contains larger and more liquid companies,
the backtest spans January 2011 through the cloud data available in 2026, and
QuantConnect applies a different execution and fee model. The magnitude could
reflect those differences. A common-period, comparable-universe experiment is
still required.

## 7. Discussion

The main result is a disconnect between global predictive stability and local
decision stability. AUC summarizes ranking performance across the full sample.
The portfolio acts on a small set near two discontinuities: a probability gate
and a top-count cutoff. Small probability changes can move many securities
across those boundaries even when the full-sample statistic barely moves.

This mechanism explains why “AUC survived the correction” is not sufficient
reassurance. The corrected classifier may retain weak information in aggregate,
yet a particular portfolio derived from it can remain fragile. Reporting the
overlap of actual selections, turnover, and return sensitivity alongside AUC
would make this fragility visible in other financial classification studies.

The accounting corrections carry a separate lesson. Averaging security-level
log returns and exponentiating creates the geometric mean of constituent gross
returns, not the return of an equal-weight portfolio. With dispersed small-cap
outcomes, the difference becomes economically large. Similarly, measuring
turnover from additions and deletions misses changes in position size, and
reducing the number of names without scaling weights can increase concentration
under the label of risk reduction. These are implementation errors rather than
subtle statistical disagreements, but they changed the project's conclusions
more than a new model did.

Momentum's relative success is a baseline result, not a novel anomaly claim.
It shows that complexity failed to earn its place in this experiment. The
machine-learning leg did not improve the simple rule, and its promotions were
sensitive to the probability pipeline. The proper inference is limited to the
tested public-data setup.

## 8. Limitations

Five limitations constrain the current evidence.

1. **Current-constituent local universes.** The local small- and mid-cap lists
   are sampled from September 2026 membership. Delisted and removed securities
   are absent from earlier years.
2. **Unmatched external validation.** The QuantConnect result uses a dynamic
   liquidity universe and different execution assumptions. It validates
   direction, not the local return estimate.
3. **Research-path dependence.** Many ideas were evaluated during the wider
   project. A 20-proposal lifetime budget records the formal improvement loop,
   but informal design choices also consume researcher degrees of freedom.
4. **Incomplete inference.** Historical results currently lack block-bootstrap
   confidence intervals, factor regressions, a Deflated Sharpe Ratio, and a
   probability-of-backtest-overfitting analysis.
5. **No clean prospective outcomes yet.** The immutable `prospective-v2`
   series begins at the next completed month end. Historical paper rows were
   generated under a superseded dating and grading implementation and are
   excluded from forward evidence.

These limitations prevent claims of investable alpha or live deployability.
They do not invalidate the controlled finding that similar AUC can coexist with
low portfolio overlap.

## 9. Prespecified work before journal submission

The following analyses will be completed without changing the frozen momentum
or machine-learning decision rules.

1. Report moving-block-bootstrap confidence intervals for mean monthly return,
   CAGR, Sharpe difference, and momentum-minus-SPY return.
2. Estimate CAPM and Fama-French five-factor-plus-momentum regressions using
   monthly portfolio returns, with heteroskedasticity and autocorrelation
   consistent standard errors.
3. Report the Deflated Sharpe Ratio using the documented research-trial count
   and return skewness and kurtosis; add a conservative sensitivity range that
   includes informal trials.
4. Run a QuantConnect comparison on a common period and a universe construction
   made as comparable as platform data allow. Record every remaining mismatch.
5. Add selection-stability plots by month and by probability distance from the
   cutoff.
6. Publish environment-lock information and immutable hashes for every table's
   source artifact.
7. Update the prospective section after at least 12 monthly cohorts while
   preserving the initial empty-series manuscript and all run manifests.

No remaining improvement-loop trials will be spent on the already inspected
historical sample before these analyses are complete.

## 10. Prospective protocol

The frozen prospective system contains two sleeves: `wide-seed0-v1` and
`mom12-1-v1`. A valid prediction run must use completed month-end data and a
clean pipeline. It writes an immutable manifest containing UTC creation time,
git state, input hashes, model backend and versions, feature list, training and
validation dates, parameters, and the complete signal set. Each ledger row
stores the manifest hash.

Positions enter at the next session's open and are observed for 20 trading
sessions. A position exits at the first actual close at or below the 15% stop,
or at the final close if no stop occurs. Round-trip costs are charged. Grading
recomputes and verifies the manifest hash and signal fields and is idempotent.
The clean records live separately from the superseded historical paper log.

At the time of this draft, the clean series contains no mature observations.
That empty state is a feature of the protocol: the historical audit can be
evaluated now, while the market supplies evidence that the research process can
no longer edit.

## 11. Conclusion

This audit finds that a repair to financial-model validation can leave
aggregate AUC almost unchanged while replacing most investable selections and
moving portfolio outcomes. In the controlled ablation, AUC changes from 0.553
to 0.551, while mean selected-name Jaccard overlap is 0.392. The result exposes
a blind spot in workflows that stop at predictive discrimination.

Correct portfolio accounting also changes the magnitude of the evidence. Once
simple-return aggregation, weight-based turnover, exposure scaling, gap-aware
stops, and initial-equity drawdown are enforced, a frozen 12-minus-1-month
momentum baseline dominates the tested machine-learning portfolio and its
ensemble. A dynamic-universe cloud implementation preserves the positive
direction, but a matched external comparison and prospective outcomes remain
necessary.

The publishable finding is therefore methodological. In small financial
machine-learning studies, the integrity of label boundaries, portfolio
selection, and the accounting ledger can matter more than the choice of model.
Researchers should report decision-level stability and preserve corrected
failures alongside headline accuracy.

## References

[1] Gu, S., Kelly, B., and Xiu, D. (2020). “Empirical Asset Pricing via
Machine Learning.” *The Review of Financial Studies*, 33(5), 2223–2273.
https://doi.org/10.1093/rfs/hhaa009

[2] Jegadeesh, N., and Titman, S. (1993). “Returns to Buying Winners and
Selling Losers: Implications for Stock Market Efficiency.” *The Journal of
Finance*, 48(1), 65–91.
https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

[3] Harvey, C. R., Liu, Y., and Zhu, H. (2016). “… and the Cross-Section of
Expected Returns.” *The Review of Financial Studies*, 29(1), 5–68.
https://doi.org/10.1093/rfs/hhv059

[4] Bailey, D. H., and López de Prado, M. (2014). “The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality.”
*The Journal of Portfolio Management*, 40(5), 94–107.
https://doi.org/10.3905/jpm.2014.40.5.094

[5] QuantConnect (2026). “Auxiliary Data: US Equity Security Master” and
“Research Guide: Survivorship Bias.” Accessed September 2026.
https://www.quantconnect.com/docs/v2/cloud-platform/datasets/quantconnect/auxiliary-data

## Appendix A. Reproduction map

| Result | Repository entry point | Evidence artifact |
|---|---|---|
| Controlled legacy/fixed purge refit | `purge_ablation.py` | `data/cache/purge_legacy_ablation/`, `data/cache/purge_fixed_ablation/` |
| Probability and selection agreement | `cache_compare.py` | Fold-level probability CSVs in the two ablation caches |
| Corrected momentum results | `mom_run.py` | Cached OHLCV files plus console summary recorded in `FINDINGS.md` |
| Corrected ML/momentum ensemble | `ens_test.py` | `data/cache/sc_full_v2/` plus console summary recorded in `FINDINGS.md` |
| Dynamic-universe momentum check | `qc_momentum.py` | QuantConnect result recorded in `FINDINGS.md` and `RESULTS.md` |
| Immutable prospective series | `paper.py` | `data/paper/runs/`, `data/paper/log_v2.csv`, `data/paper/grades_v2.csv` |

## Appendix B. Evidence-status vocabulary

- **Controlled:** one named mechanism changes while the relevant observations,
  model, and evaluation remain fixed.
- **Holdout:** evaluated once after a rule passed a previously stated gate;
  later inspection can contaminate that status but cannot restore it.
- **Directional replication:** the sign survives in a materially different
  universe or execution system; magnitudes are not treated as estimates of one
  another.
- **Prospective:** the signal, implementation, and complete manifest are frozen
  before the outcome begins.
- **Superseded:** a result produced by a known-invalid implementation and kept
  only as part of the audit trail.
