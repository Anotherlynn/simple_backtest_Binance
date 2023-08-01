import backtrader as bt

class MyBroker(bt.brokers.BackBroker):
    params = (
        ("args", None),  # The amount of slippage to simulate (in dollars or percentage)
    )

    def __init__(self):
        super().__init__()
        self._last_prices = {}  # A dictionary to store the last price for each asset
        self.slippage = self.params.args["slippage"]
        self.init_cash = self.params.args["cash"]
    
    def start(self):
        self.set_cash(self.init_cash)

    def buy(self, owner, data, size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None,
            **kwargs):

        if price is None:
            price = data.close[0]

        # Add slippage to the price
        price += self.slippage * price

        # Execute the order
        return super().buy(owner, data, size, price=price, plimit=plimit,
            exectype=exectype, valid=valid, tradeid=tradeid, oco=oco,
            trailamount=trailamount, trailpercent=trailpercent,
            **kwargs)

    def sell(self, owner, data, size, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None,
             **kwargs):

        if price is None:
            price = data.close[0]

        # Subtract slippage from the price
        price -= self.slippage * price

        # Execute the order
        return super().sell(owner, data, size, price=price, plimit=plimit,
            exectype=exectype, valid=valid, tradeid=tradeid, oco=oco,
            trailamount=trailamount, trailpercent=trailpercent,
            **kwargs)