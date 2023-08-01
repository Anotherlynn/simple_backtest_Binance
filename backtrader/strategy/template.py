import backtrader as bt

# Due to the limit of time, I just to testify several simple strategy
# You can embed in your strategy using this template here, and adjust the config.json accordingly
class Template(bt.SignalStrategy):
    def __init__(self):
        pass

    def next(self):
        pass

    def notify_order(self, order):
        pass