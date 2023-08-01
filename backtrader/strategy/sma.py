import backtrader as bt

    # "strategy_args":{
    #     "strategy_path": "strategy.sma",
    #     "strategy_class": "SmaCross",
    #     "parameters":{
    #         "l_period": 30,
    #         "s_period": 10
    #     }
    # },
    
class SmaCross(bt.SignalStrategy): 
    params = (('args', None),)
    def __init__(self):
        short_p = self.params.args["s_period"]
        long_p = self.params.args["l_period"]

        sma1, sma2 = bt.ind.SMA(period=short_p), bt.ind.SMA(period=long_p)
        self.crossover = bt.ind.CrossOver(sma1, sma2)
        self.order = None

    def next(self):
        if self.order:  
            self.cancel(self.order)  
    
        if not self.position:  
            if self.crossover > 0:  
                self.order = self.buy(size=10)
        elif self.crossover < 0:            
            self.order = self.close()

    def notify_order(self, order):
        pass
        

