# NOTICE:
# please run the requirments.txt before start to run
import importlib
import json
import types
import csv
import numpy as np
import pandas as pd
import yfinance as yf
import yfinance.shared as shared
import backtrader as bt
import matplotlib.pyplot as plt

# import seaborn as sns

from broker import MyBroker
from analyzer.totalvalue import TotalValue

# Analyzers
# SharpeRatio: SQN: DrawDown: TimeReturn: VWR: TradeAnalyzer: PyFolio: AnnualReturn: Calmar: Omega: Sortino: TailRisk: 
# intervals for yf :[1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]

def prep_data_feed(symbol, fromdate, todate, freq, read_from_csv):
    if read_from_csv: 
        df = pd.read_csv(read_from_csv, index_col='Date', parse_dates=True)
        data = bt.feeds.PandasData(dataname = df)
    else:
        data =  bt.feeds.PandasData(dataname=yf.download(symbol, fromdate, todate, interval = freq))
        fails = list(shared._ERRORS.keys())
        if fails:
            raise RuntimeError("\n [Custom Error Msg] Fail to download %s symbol" % ("|".join(fails)))
    return data

def run(data_feed,strategy_class,broker_info,commission_info,strategy_args):
    cerebro = bt.Cerebro()

    cerebro.addstrategy(strategy_class,args=strategy_args)

    cerebro.adddata(data_feed)

    my_broker = MyBroker(args=broker_info)
    cerebro.setbroker(my_broker)

    cerebro.broker.setcommission(commission = commission_info.commission, margin = commission_info.margin,mult = commission_info.mult)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='myDD')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='mySQN')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='myVWR')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='myAR')
    cerebro.addanalyzer(TotalValue, _name='myTV')

    results = cerebro.run()

    cerebro.plot()

    return results

def load_config():
    strategy_args = types.SimpleNamespace()
    commission_args = types.SimpleNamespace()
    broker_args = types.SimpleNamespace()

    with open('config.json', 'r') as f:
        data = json.load(f)
        for key, value in data["strategy_args"].items():
            setattr(strategy_args, key, value)
        for key, value in data["commission_args"].items():
            setattr(commission_args, key, value)
    return data["data_args"],strategy_args,commission_args,data["broker_args"]

if __name__ == "__main__":
    data_config,strategy_config,commission_config,broker_config = load_config()

    data_feed = prep_data_feed(**data_config)

    module = importlib.import_module(strategy_config.strategy_path)
    Strategy = getattr(module, strategy_config.strategy_class)
    results = run(data_feed, Strategy, broker_config, commission_config, strategy_config.parameters)
    
    print('Sharpe Ratio :', results[0].analyzers.mysharpe.get_analysis())
    print('Draw Down    :', results[0].analyzers.myDD.get_analysis())
    print('SQN          :', results[0].analyzers.mySQN.get_analysis())
    print('VWR          :', results[0].analyzers.myVWR.get_analysis())
    print('Annual Return:', results[0].analyzers.myAR.get_analysis())

    od = results[0].analyzers.myTV.get_analysis()
    headers = list(od.keys())
    values = [str(val) for val in list(od.values())]
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for k, v in od.items():
            writer.writerow([k, (v-broker_config["cash"])/broker_config["cash"]])

    data = pd.read_csv("output.csv",index_col=0,squeeze=True)
    data.plot()
    plt.show()