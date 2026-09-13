# Momentum sleeve for QuantConnect cloud (UNTESTED locally — paste and run).
# Purpose: survivor-free validation of mom12-1-v1 on QC's delisting-aware
# data. Compare vs our Yahoo run on overlap first; if they match, the
# survivor-free result is credible.
# Framework: coarse universe (dollar-volume filtered = our P7 idea built
# in) + 12-1 momentum rank + monthly rebalance + IB fee model.
# Free tier: cloud backtests included with limits.
from AlgorithmImports import *


class MomentumSleeve(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2011, 1, 1)
        self.SetEndDate(2026, 8, 1)
        self.SetCash(100000)
        self.SetBenchmark("SPY")
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseFilter)
        self._picks = []
        # monthly, first trading day, after open
        self.Schedule.On(self.DateRules.MonthStart(),
                         self.TimeRules.AfterMarketOpen("SPY", 30),
                         self.Rebalance)
        self.SetWarmUp(252, Resolution.Daily)

    def CoarseFilter(self, coarse):
        # liquid US equities only (P7 liquidity idea, point-in-time universe)
        liquid = [c for c in coarse if c.HasFundamentalData and c.Price > 5
                  and c.DollarVolume and c.DollarVolume > 500000]
        return sorted(liquid, key=lambda c: c.DollarVolume, reverse=True)[:200]

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        hist = self.History(self.Universe.Select(lambda c: c.Symbol),
                            273, Resolution.Daily)
        if hist.empty:
            return
        mom = {}
        for sym in hist.index.get_level_values(0).unique():
            try:
                closes = hist.loc[sym]["close"].dropna()
                if len(closes) >= 273:
                    mom[sym] = closes.iloc[-22] / closes.iloc[-273] - 1
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
        for sym in picks:
            self.SetHoldings(sym, 1.0 / len(picks))
        self._picks = sorted([str(s.Value) for s in picks])
        self.Log(f"picks: {','.join(self._picks)}")
