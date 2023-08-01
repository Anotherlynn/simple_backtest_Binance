import backtrader as bt

    # "strategy_args":{
    #     "strategy_path": "strategy.dual_thrust",
    #     "strategy_class": "DualThrust",
    #     "parameters":{
    #         "period": 20,
    #         "k1": 0.5,
    #         "k2": 0.5
    #     }
    # },

class DualThrust(bt.Strategy):
    params = (('args', None),)
    def __init__(self):
        self.period = self.params.args['period']
        self.k1 = self.params.args['k1']
        self.k2 = self.params.args['k2']

        self.high = bt.indicators.Highest(self.data.high, period=self.period)
        self.low = bt.indicators.Lowest(self.data.low, period=self.period)

    def next(self):
        thrust_range = self.high[-1] - self.low[-1]
        buy_level = self.data.close[-1] + self.k1 * thrust_range
        sell_level = self.data.close[-1] - self.k2 * thrust_range

        current_price = self.data.close[0]
        
        # max_size = int(self.broker.cash / current_price)
        max_size = 1

        if not self.position:
            if self.data.open[0] > buy_level:
                self.buy(size = max_size)
        elif self.data.open[0] < sell_level:
            self.close()

        # if not self.position:
        #     if self.data.open[0] > buy_level:
        #         self.buy(size = max_size)
        #     elif self.data.open[0] < sell_level:
        #         self.sell(size = max_size)
        # elif self.position.size > 0 and self.data.open[0] < sell_level:
        #     self.close()
        #     self.sell(size = max_size)
        # elif self.position.size < 0 and self.data.open[0] > buy_level:
        #     self.close()
        #     self.buy(size = max_size)