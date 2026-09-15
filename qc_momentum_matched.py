# Momentum sleeve for QuantConnect cloud -- MATCHED-UNIVERSE comparison
# (PAPER.md Section 9 item 4; cloud-run Sep 14, 2026).
# QuantConnect project: Calculating Fluorescent Yellow Owl
# Backtest: Determined Black Cow
# Result: +2,557.483% total return, 23.617% CAGR, 48.100% max drawdown,
# 0.690 daily-path Sharpe, $18,335.09 fees, 1,733 orders. On comparable
# common-month sampling, QC is 23.75% CAGR / 0.93 Sharpe / -39.30% maxDD
# versus 20.86% / 0.83 / -37.15% locally; returns correlate 0.933.
#
# Purpose: qc_momentum.py already validated DIRECTION on a dynamic top-200
# liquid-equity universe (survivor-free, but a different universe and period
# than the local backtest -- not a magnitude estimate). This file instead
# fixes the universe to the EXACT 99 tickers mom_run.py used for the local
# small-cap result (predictor/universe.py SMALLCAP, minus LEG -- see
# FINDINGS.md's data-hygiene note) and a common 2011-2026 calendar window, so
# a mismatch in the result is primarily attributable to data/execution details
# (adjusted-close methodology, fee model, fill timing, corporate actions),
# not to a different universe or period. It does NOT remove survivorship
# bias -- these are today's survivors on both sides -- it isolates whether
# QC's independent data/execution reproduces the local Yahoo-based number
# for the same names and dates. That is a different, complementary question
# from the dynamic-universe run's.
#
# Local reference to compare against (mom_run.py, same 99 tickers, IWM
# market-relative, 25bps, restricted to 2011-02-28..2026-05-29, 184 months,
# the actual trading dates that fall inside this file's start/end below):
#   CAGR +21.65%, Sharpe 0.85, maxDD -37.15%.
#   Matched-window buy-hold: SPY +14.0%, IWM +10.1% (same 184 months).
# QC's June 17 end leaves its final monthly position incomplete. The committed
# qc_reconcile.py artifact excludes that partial month from path comparison.
#
# Framework: fixed symbol list (AddEquity, no coarse/dynamic universe) +
# 12-1 momentum rank (lookback=252, skip=21, same as mom_run.py) + top
# decile (>=5 names) + monthly rebalance + IB fee model.
from AlgorithmImports import *

# predictor.universe.SMALLCAP minus LEG (see FINDINGS.md: Yahoo now returns
# only 6 stale rows for LEG post 2026-08-27, consistent with a delisting).
TICKERS = [
    "AAMI", "ACA", "AGX", "AMTM", "ARR", "AWR", "BFAM", "BMI", "BTU", "CC",
    "CON", "CRI", "CWEN", "DAN", "DFIN", "DXC", "EMN", "EPC", "FBP", "FMC",
    "GFF", "GPOR", "HAYW", "HMN", "IIPR", "JOE", "KMT", "KSS", "LTH", "MAN",
    "MHK", "MTX", "NGVT", "NPO", "OGN", "PBI", "PMT", "RCUS", "RHP", "SAFE",
    "SHO", "SM", "SXT", "TGTX", "UA", "UVV", "VSXY", "WHD", "WU", "ACAD",
    "ADAM", "AGYS", "ALRM", "AOSL", "AZTA", "BL", "CARG", "CENTA", "CLSK",
    "COLL", "CSW", "DAVE", "EBC", "EYE", "FHB", "FRPT", "GSHD", "HLIT",
    "HUBG", "INDV", "IRDM", "KLIC", "LKFN", "LYFT", "MBGL", "MIR", "MSEX",
    "NBHC", "NSSC", "OPLN", "PAYO", "PENN", "PLUS", "PRDO", "PSMT", "QDEL",
    "REYN", "RUSHA", "SDGR", "SHEN", "SPNT", "STRA", "TMDX", "UCB", "UPBD",
    "VCTR", "VSAT", "WDFC", "WSC",
]


class MomentumSleeveMatched(QCAlgorithm):
    def Initialize(self):
        # Common period: local run covers through Sep 2026, but QC's prior
        # completed run ("Creative Yellow Antelope") only had data through
        # 2026-06-17 -- use that as the shared end date rather than assuming
        # QC has caught up. Adjust if QC now has more recent data.
        self.SetStartDate(2011, 1, 1)
        self.SetEndDate(2026, 6, 17)
        self.SetCash(100000)
        self.SetBenchmark("SPY")
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        self._symbols = [self.AddEquity(t, Resolution.Daily).Symbol
                         for t in TICKERS]
        self.Schedule.On(self.DateRules.MonthStart(),
                         self.TimeRules.At(10, 30),
                         self.Rebalance)
        self.SetWarmUp(252, Resolution.Daily)

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        syms = [s for s in self._symbols if self.Securities[s].HasData]
        if len(syms) < 10:
            return
        hist = self.History(syms, 252, Resolution.Daily)
        if hist.empty:
            return
        mom = {}
        for sym in hist.index.get_level_values(0).unique():
            try:
                closes = hist.loc[sym]["close"].dropna()
                if len(closes) >= 252:
                    mom[sym] = closes.iloc[-21] / closes.iloc[-252] - 1
            except Exception:
                continue
        if len(mom) < 10:
            return
        ranked = sorted(mom, key=mom.get, reverse=True)
        n = max(5, int(len(ranked) * 0.10) + 1)
        picks = set(ranked[:n])
        for sym in list(self.Portfolio.Keys):
            if sym not in picks and self.Portfolio[sym].Invested:
                self.Liquidate(sym)
        tradable_picks = [sym for sym in picks
                          if self.Securities[sym].HasData
                          and self.Securities[sym].Price > 0]
        for sym in tradable_picks:
            self.SetHoldings(sym, 1.0 / len(tradable_picks))
        self.Log(f"picks: {','.join(sorted(str(s.Value) for s in tradable_picks))}")
