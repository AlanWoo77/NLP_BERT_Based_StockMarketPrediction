from backtrader.feeds import PandasData


class pandasData(PandasData):
    lines = ("signal",)
    params = (("signal", -1),)
