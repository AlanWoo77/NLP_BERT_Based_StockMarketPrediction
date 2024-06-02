import backtrader as bt


class Momentum(bt.Indicator):
    lines = ("momentum",)
    params = (("period", 30),)  # Momentum look-back period

    def __init__(self):
        self.lines.momentum = self.data.close / self.data.close(-self.params.period) - 1
