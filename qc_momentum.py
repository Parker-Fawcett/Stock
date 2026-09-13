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
        self.add_universe(self.coarse_filter)
        self._picks = []
        # monthly, first trading day, mid-morning (no symbol dependency)
        self.Schedule.On(self.DateRules.MonthStart(),
                         self.TimeRules.At(10, 30),
                         self.Rebalance)
        self.SetWarmUp(252, Resolution.Daily)

    def coarse_filter(self, fundamental):
        # liquid US equities only (P7 liquidity idea, point-in-time universe)
        liquid = [f for f in fundamental
                  if f.has_fundamental_data and f.price > 5
                  and f.dollar_volume and f.dollar_volume > 500000]
        ranked = sorted(liquid, key=lambda f: f.dollar_volume, reverse=True)[:200]
        return [f.symbol for f in ranked]

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        syms = [x.Symbol for x in self.ActiveSecurities.Values
                if x.Symbol.SecurityType == SecurityType.Equity]
        if len(syms) < 10:
            return
        # Same frozen 12m-1m spec as mom_run.py (lookback=252, skip=21).
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
        self._picks = sorted([str(s.Value) for s in tradable_picks])
        self.Log(f"picks: {','.join(self._picks)}")
