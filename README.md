# simple_backtest_Binance
![Build Status](https://app.travis-ci.com/Anotherlynn/simple_backtest_Binance.svg?branch=main)
![](https://img.shields.io/badge/powered%20by-@Anotherlynn-green.svg)
![](https://img.shields.io/badge/language-Python-green.svg)
![](https://img.shields.io/badge/version-Python3.8-green.svg)
—————————————————————————————————————————————————————

**Important notice of the functions and analysis**

The simple_backtest_Binance provides protocols and functions to analyse temporary announcements from companies on stock market. 

Different LLMs, embeddings methods, testing models and convenient tools, such as access to the OpenAI API, are included, written in the Python language.

You can find usage examples [here](examples.py).

# Important notice of the functions and analysis



First, thank you for providing me this chance to build the project. Here I would like to give some instructions for you to use the project.

### Accessing the Data

The total size of `Data` is too large to send via email. Please download all the data in the original folder `data` [**here**]() and add them into  `src/` or replace the original `data` folder in the zip file to start use. 

Please contact me if any problem rises.

###  Adjustments before using

Please use the command to create a virtual environment **"bt"** to make sure the packages won't distract your base environment.
```
conda create --name bt python=3.8
conda activate bt
```
Use `python3.8 -m pip install XXX` to install all the packages.

### Complete of requirements

In the file `data_gen.py`, codes for:

- Data downloading & processing: 
  
  downloading the `aggTrades` and  `kline ` are provided and testified with no problem. However, the 1h data is too large for my pc and even $9.99 *GPU* on *Colab Pro* to process. I tried my best to meet the data format requirements of year 2023 and 2018. Please follow the codes to process the rest of years.
  

In the `model.py`:
- Modeling:
  
  all the required models are defined in a class `MyModel` and are defined to run on `torch.cuda`. Please follow the lasso example, train, evaluate and test the targets according to your needs.

In the folder `src/alpha/`:
- Alpha implementing:`alpha_SWING.py`,`alpha_MACD.py`
  
  following the indicators on *tradingview.com*, I finished the factor `MACD`, `Swing Failure Pattern` that are both testified with no problem. Yet I didn't have time to complete the following 2 factors of `Momentum` and `Garch(1,1)`. But I will continues to finished the project afteer the submission. Please contact me if you are still interested.

In the folder  `src/_func/`:
- function classes used be main files
- target definition: all the detailed class of two targets `DIFF` and `BMP` for different tasks are defined in `src/_func/Y.py`.


In the folder `backtrader`:
- BackTesting structure: `broker.py`
  
  in this file, a customized broker class called `MyBroker`that extends the `bt.brokers.BackBroker` class from the Backtrader library, which will be used in the main file `run_strategy.py`. The purpose of this custom broker class is to simulate slippage in the backtest strategy.

  The `MyBroker` class has the following attributes:
  - `params`: A parameter tuple that can be passed during initialization.
  - `_last_prices`: A dictionary to store the last price for each asset.
  - `slippage`: The amount of slippage to simulate (in dollars or percentage).
  - `init_cash`: The initial cash amount for the broker.
    
  
  Several methods are overridden:
    - the `start` method to set the initial cash amount.
    - the `buy` method to add slippage to the buy price and execute the order.
    - the `sell` method to subtract slippage from the sell price and execute the order.
  
- Three simple strategies:
  
  I tested single-point strategies like `dual_thrust`, `rbreaker` and `SmaCross`. The three simple strategies showed good results. Yet due to the data loading problem, I am unable to work on the whole dataset from 2018 to 2023. 

  More strategies could be used in the sturcture. Please follow the template in `backtrader/strategy/template.py`.
  

In the foler `paper_trading`:
- please refer to the `./paper_trading/README.md` for instructions.

### Analytics and suggestions
Since I didn't test on the whole dataset, I preefer to save the analysis at this point. However, according to the protocols for me to test on 2023 data, several problems occurred multiple times. So I would like to give some conclusions here:
1. In the modeling part, Lasso, RandomForest and CNN are usable examples for neither Regression task and Classification task, because they requires longer time to proceed while giving bad predictions. `Transformer`is good but seems costly for digital format data. What's more, the Transformer based model requires to be updated on a regular trading period ( normally a week or a monnth).
   
   It is better used for NLP dataset. I have experience using Transformer to generate sentiment indicators and topic indicators for over 2 years, please let me know if you are interested.


2. Some of the links and modules are outdates. Please refers to `./paper_trading/README.md` for more information and better results.

3. To run the project, you may need to link to your cloud databse or drive to process. Load all models and data into `torch.cuda` might be a solution. Apply for API in [google clouds](https://console.cloud.google.com/) is the best way for training and uploading result at the same time. Refer [here](https://developers.google.com/drive/api/quickstart/python) for more details.

