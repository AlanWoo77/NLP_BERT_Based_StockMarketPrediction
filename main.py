import pickle

import backtrader as bt
import pandas as pd
import quantstats as qs

from load_data import pandasData
from strategy import TestStrategy
import warnings
warnings.filterwarnings("ignore")
from functools import reduce

# Here, we read the backtesting data of different models: base_bt.pickle / distilbert_bt.pickle / finbert_bt.pickle

models = ["base", "distilbert", "finbert"]
dfs = []
for model in models:
    with open(f"pickles/{model}_bt.pickle", "rb") as file:
        datas = pickle.load(file)

    cerebro = bt.Cerebro()

    for stock_name, df in datas.items():
        data = pandasData(dataname=df)
        cerebro.adddata(data, name=stock_name)

    # add the trading strategy to the backtrader
    cerebro.addstrategy(TestStrategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn", timeframe=bt.TimeFrame.Days)

    cerebro.broker.set_cash(100_000)
    cerebro.broker.setcommission(commission=0.00002)

    # print the starting point
    results = cerebro.run()

    first_strategy = results[0]

    timereturn = results[0].analyzers._TimeReturn.get_analysis()

    df_returns = pd.DataFrame(list(timereturn.items()), columns=['date', model])

    dfs.append(df_returns)

    sharpe_ratio = first_strategy.analyzers.sharpe_ratio.get_analysis()
    drawdown_info = first_strategy.analyzers.drawdown.get_analysis()

merged_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
merged_df.set_index("date", inplace=True)

qs.reports.html(
    returns=merged_df, benchmark=None, output=rf"three_models.html",
    title="COMPARISION", periods_per_year=252
    )
