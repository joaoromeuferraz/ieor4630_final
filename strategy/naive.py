from strategy import BaseStrategy
import numpy as np


class EqualWeights(BaseStrategy):
    def __init__(self, start_date, end_date, starting_capital=1000000, rebal_freq='M', top=None):
        super().__init__(start_date=start_date, end_date=end_date,
                         starting_capital=starting_capital, rebal_freq=rebal_freq, top=top)

    def on_day_close(self):
        pass

    def on_rebal_day(self):
        N = len(self.valid_tickers)
        weights = [1 / N for _ in range(N)]
        order = dict(zip(self.valid_tickers, weights))
        self.portfolio.clear_holdings()
        self.portfolio.trade(order=order, is_permno=True)


class ValueWeighted(BaseStrategy):
    def __init__(self, start_date, end_date, starting_capital=1000000, rebal_freq='M', top=None):
        super().__init__(start_date=start_date, end_date=end_date,
                         starting_capital=starting_capital, rebal_freq=rebal_freq, top=top)

    def on_day_close(self):
        pass

    def on_rebal_day(self):
        mv = self.prices.loc[:, 'prc'] * self.prices.loc[:, 'shrout']
        mv = mv.dropna()

        tickers = [p for p in mv.index]
        weights = np.array([mv.loc[p] for p in mv.index])
        weights /= np.sum(weights)

        order = dict(zip(tickers, weights))
        self.portfolio.clear_holdings()
        self.portfolio.trade(order=order, is_permno=True)