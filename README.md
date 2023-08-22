# Simple_backtest_Binance
![Build Status](https://app.travis-ci.com/Anotherlynn/simple_backtest_Binance.svg?branch=main)
![](https://img.shields.io/badge/powered%20by-@Anotherlynn-green.svg)
![](https://img.shields.io/badge/language-Python-green.svg)
![](https://img.shields.io/badge/version-Python3.8-green.svg)
—————————————————————————————————————————————————————

The Simple_backtest_Binance is a high-frequency trading backtesting framework. It provides usages of Binance APIs and functions to analyse hour_level trading data of 2 cryptos, **Ethererum (ETH)** and **Cardano (ADA)**.

Data collection, alpha implementing, analysing models, and backtest on [backtrader](https://github.com/mementum/backtrader) and paper-trading on [Binance api](https://python-binance.readthedocs.io/en/latest/market_data.html) are all included, written in the Python language.

You can find usage examples [here](./examples.py).

## Installation
### Creating a Environment (Optional)
> Note: You are strongly suggested to build a virtual environment in `python3.8` and above.

To start a virtual environment, you can use [conda](https://github.com/conda/conda)
```bash
conda create -n your_env_name python=3.8
```
To activate or deactivate the enviroment, you can use:

On Linux
```bash
source activate your_env_namn
# To deactivate:
source deactivate
```
On Windows
```bash
activate your_env_name
# to deactivate
deactivate env_name # or activate root
```

### Building the Documentation
To use the tools, you need to install the packages in required version.
```bash
cd proj/
conda install -n your_env_nam requirements.txt # or python3.8 -m pip install -r requirements.txt
```

## Getting Started

- Tools and functions, see func_instruction [here](./_func/README.md)
  

- Data collection
    - [Kline and aggTrades data downloading & saving from **Binance Market Data** api](./_func/)
    - [Kline and aggTrades data merging](./data_gen.py)
    - merging on `number of trade (trades_at_current_ts)` and `buy/sell ratio (buy_sell_ratio_at_current_ts)`
    > Notice: the data size is more than 1.5GB per year, please pay attention to the storage of data. You can use `MangoDB` to store the data.
  
- Alpha implement & tuning (some of the alphas were inspired by [TradingView](https://www.tradingview.com/)
    - [MACD](./alpha/alpha_MACD.py)
    - [Swing Failure Pattern by EmreKb](./alpha/alpha_SWING.py)
    - [Garch(1,1)](./alpha/alpha_Garch.py)
    - [Momentum adjusted Moving Average](./alpha/alpha_Momentum.py)
    - updating…
    

- Alpha training
    - [Use different mark_out to train alphas on core models (parallel training available)](./model.py),available models:
      - Lasso regression
      - OLS & WLS
      - Transformer 
      - Random forest
      - LSTM
      - GRU
      - CNN
      
    - The mark_out (find [here](./func/Y.py) for defination) including:
      - previous 5, 10, 20, 40, 60, 100 bar and current bar diff
      - time weighted average price diff (tWap)
      - volume weighted average price diff (vWap)
      - updating…
    - Models are defined in a class `MyModel`(GPU available)
  

  
- BackTesting
  - [Backtrader framework](./backtrader/broker.py)
    - In this file, a customized broker class called `MyBroker`that extends the `bt.brokers.BackBroker` class from the Backtrader library, which will be used in the main file `run_strategy.py`. The purpose of this custom broker class is to simulate slippage in the backtest strategy.
    - The `MyBroker` class has the following attributes:
      - `params`: a parameter tuple that can be passed during initialization
      - `_last_prices`: a dictionary to store the last price for each asset
      - `slippage`: the amount of slippage to simulate (in dollars or percentage)
      - `init_cash`: the initial cash amount for the broker 
    - Several methods are overridden:
      - the `start` method to set the initial cash amount
      - the `buy` method to add slippage to the buy price and execute the order
      - the `sell` method to subtract slippage from the sell price and execute the order
  
  - [Trading strategy](./backtrader/strategy) 
    - To run the strategy, use `python run_strategy.py`
    - strategies available:
      - dual_thrust
      - rbreaker
      - SmaCross
      - follow the template [`./backtrader/strategy/template.py`](backtrader/strategy/template.py) to customize your strategy
  

- Paper trading
  - see [`./paper_trading/README.md`](./paper_trading/README.md) for instructions


- Analytics

  Since I didn't test on the whole dataset, I prefer to save the analysis at this point. However, according to the protocols for me to test on 2023 data, several problems occurred multiple times. So I would like to give some conclusions here:
  - In the modeling part, Lasso, RandomForest and CNN are usable examples for neither Regression task and Classification task, because they requires longer time to proceed while giving bad predictions. `Transformer`is good but seems costly for digital format data. What's more, the Transformer based model requires to be updated on a regular trading period ( normally a week or a monnth).
    - It is better used for NLP dataset. I have experience using Transformer to generate sentiment indicators and topic indicators for over 2 years, please let me know if you are interested.
  - Some of the links and modules are outdates. Please refers to `./paper_trading/README.md` for more information and better results.
  - To run the project, you may need to link to your cloud databse or drive to process. Load all models and data into `torch.cuda` might be a solution. Apply for API in [google clouds](https://console.cloud.google.com/) is the best way for training and uploading result at the same time. Refer [here](https://developers.google.com/drive/api/quickstart/python) for more details.

