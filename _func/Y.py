import math
import numpy as np

# data:
# data.columns
# Index(['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
#        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
#        'Taker buy quote asset volume', 'Ignore',
#        'trades_at_current_ts', 'trades_at_current_ts', 'trades_at_current_ts',
#        'Timestamp', 'Aggregate_tradeId', 'Price', 'Quantity', 'First_tradeId',
#        'Last_tradeId', 'Was_the_buyer_the_maker',
#        'Was_the_trade_the_best_price_match'],
#       dtype='object')

class DIFF:
    def __init__(self,data):
        self.data = data.copy()

    def diff(self, mark_out, ma_type='SMA', last_point='Close', first_point='Close', use_ma=True):
        """
        Technical Analyse Factor:diff
        using different mark out including [5, 10, 20, 40, 60, 80, 100]
        using different moving averages including ['SMA', 'EMA', 'WMA', 'VWMA', 'ALMA']
        :return:
        """
        # last_point = input('open', title="Current point of reference", options=['open', 'high', 'low', 'close'])
        # first_point = input('close', title="Previous point of reference", options=['open', 'high', 'low', 'close'])
        # ma_type = input('SMA', title="Moving average type", options=['SMA', 'EMA', 'WMA', 'VWMA', 'ALMA'])
        # ma_len = int(input("Moving average length"))

        # define the price to use, could be open, close, high, low and combination of them
        last = 'Open' if last_point == 'open' else ('High' if last_point == 'high'
                                                    else ('Low' if last_point == 'low' else 'Close'))
        first = 'Open' if first_point == 'open' else ('High' if first_point == 'high'
                                                      else ('Low' if first_point == 'low' else 'Close'))
        diff = (self.data[last]-self.data[first].shift(int(mark_out)))/self.data[first].shift(int(mark_out))

        if use_ma:
            # Notice that, here the technical moving averages could be easily used from talib.
            # However, considering the integrity, cost and efficiency of the whole project,
            # to use talib takes longer time to process.
            #  So here I simply constructed those moving averages by numpy

            if ma_type == 'SMA':
                weights = np.ones(mark_out) / mark_out
                return np.convolve(diff, weights, mode='valid')

            elif ma_type == 'EMA':
                weights = np.exp(np.linspace(-1., 0., mark_out))
                weights /= weights.sum()
                return np.convolve(diff, weights[::-1], mode='valid')

            elif ma_type == 'WMA':
                weights = np.arange(1, mark_out + 1)
                weights /= weights.sum()
                return np.convolve(diff, weights[::-1], mode='valid')

            elif ma_type == 'VWMA':
                weights = np.multiply(diff, data['Volume'])
                weights /= np.sum(data['Volume'])
                return np.convolve(weights, np.ones(mark_out) / mark_out, mode='valid')

            elif ma_type == 'ALMA':
                sigma, offset = 0.5, 5
                m = offset * (mark_out - 1)
                s = mark_out / sigma
                alpha = np.exp(-(np.square(np.arange(mark_out) - m) / (2 * np.square(s))))
                weights = alpha / np.sum(alpha)
                return np.convolve(diff, weights[::-1], mode='valid')
        else:
            return np.array(diff)



class BMP:
    def __init__(self, data, length = 800, mark_out=5, useLogReturns = True):
        self.data = data.copy()
        self.length = length
        self.mark_out = mark_out
        # using different mark out including [5, 10, 20, 40, 60, 80, 100]
        self.useLogReturns = useLogReturns

    def percent(self, a, b):
        return ((a - b) / b) * 100

    def logReturn(self, a, b):
        return math.log(a / b) * 100

    def change(self, close1, close2):
        return round(self.logReturn(close1, close2), 3) \
            if self.useLogReturns else round(self.percent(close1,close2),3)

    def cal_dist(self):
        data1 = self.data.copy().shift(self.mark_out)
        Change_ = self.change(self.data['Close'], data1['Close'])
        self.data['Change'] = Change_

        def cal_hist(close, open, change):
            """
            get the distribution of return in the base period
            the distribution will be used to calculate the probability in the following steps
            :return:
            """
            if close > open:
                if change in movement_size_green:
                    index = movement_size_green.index(change)
                    count_green[index] += 1
                else:
                    movement_size_green.append(change)
                    count_green.append(1)

            if close < open:
                if change in movement_size_red:
                    index = movement_size_red.index(change)
                    count_red[index] += 1
                else:
                    movement_size_red.append(change)
                    count_red.append(1)


        prob = []
        for i in self.data.rolling(window=self.length):
            dX = i.reset_index(drop=True)
            if dX.shape[0] >= self.length:
                movement_size_green = []
                count_green = []
                movement_size_red = []
                count_red =  []

                cal_hist(dX['Close'],dX['Open'],dX['Change'])

                probability_green = np.zeros(len(count_green))
                probability_red = np.zeros(len(count_red))

                count_green_sum = sum(count_green)
                count_red_sum = sum(count_red)

                for i in range(len(count_green)):
                    probability = count_green[i] / count_green_sum * 100
                    probability_green[i] = probability

                for i in range(len(count_red)):
                    probability = count_red[i] / count_red_sum * 100
                    probability_red[i] = probability

                price = dX.tail(1)['Price']
                open = dX.tail(1)['Open']

                if self.useLogReturns:
                    price = round(self.logReturn(price, open), 3)
                else:
                    price = round(self.percent(price, open), 3)

                green_index = movement_size_green.index(price) \
                    if price > 0 and price in movement_size_green \
                    else -1

                red_index = movement_size_red.index(price) \
                    if price < 0 and price in movement_size_red \
                    else -1

                probability_ = probability_green[green_index] \
                    if green_index != -1 \
                    else (probability_red[red_index]
                          if red_index != -1
                          else 0)
            else:
                probability_ = None

            prob.append(probability_)

        return np.array(prob)