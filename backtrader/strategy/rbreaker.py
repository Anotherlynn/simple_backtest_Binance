import backtrader as bt

    # "strategy_args":{
    #     "strategy_path": "strategy.rbreaker",
    #     "strategy_class": "RBreaker",
    #     "parameters":{
    #         "period": 1
    #     }
    # },

class RBreaker(bt.Strategy):
    params = (('args', None),)
    def __init__(self):
        self.period = self.params.args["period"]
        
        self.high = bt.indicators.Highest(self.data.high, period=self.period)
        self.low = bt.indicators.Lowest(self.data.low, period=self.period)
        
    def next(self):
        prev_high = self.high[-1] 
        prev_low = self.low[-1]
        prev_close = self.data.close[-1]

        pivot = (prev_high + prev_close + prev_low)/3

        break_buy_price = prev_high + 2 * (pivot - prev_low)
        observe_sell_price = pivot + prev_high - prev_low
        reverse_sell_price = 2 * pivot - prev_low

        reverse_buy_price =  2 * pivot - prev_high
        observe_buy_price = pivot - prev_high + prev_low
        break_sell_price =  prev_low - 2 * (prev_high - pivot)

        current_price = self.data.close[0]
        max_size = int(self.broker.cash / current_price)
        # max_size = 1

        if not self.position:
            if current_price > break_buy_price:
                self.buy(size = max_size)
            elif current_price < break_sell_price:
                self.sell(size = max_size)
        if self.position.size > 0:
            if self.high[0] > observe_sell_price and current_price < reverse_sell_price:
                self.close()
                self.sell(size = max_size)
        if self.position.size < 0:
            if self.low[0] < observe_buy_price and current_price > reverse_buy_price:
                self.close()
                self.buy(size = max_size)
        