from strategy import BaseStrategy
import numpy as np
import pandas as pd
from data.helpers import loc_nearest
from scipy.optimize import minimize


class Markowitz(BaseStrategy):
    def __init__(self, start_date, end_date, min_ret, starting_capital=1000000, rebal_freq='M', lb_window=252, top=None):
        super().__init__(start_date=start_date, end_date=end_date,
                         starting_capital=starting_capital, rebal_freq=rebal_freq, top=top)
        self.lb_window = lb_window
        self.min_ret = min_ret
        self.min_ret = (1 + self.min_ret) ** (1 / 252) - 1

    def _estimate_params(self):
        current_idx = np.where(self.portfolio.data.dates_d == self.portfolio.current_date)[0][0]
        start_idx = current_idx - self.lb_window - 1
        end_idx = current_idx - 1

        if start_idx < 0:
            return None

        start = self.portfolio.data.dates_d[start_idx].strftime('%Y-%m-%d')
        end = self.portfolio.data.dates_d[end_idx].strftime('%Y-%m-%d')

        prices = self.portfolio.data.get_prices(start, permno=list(self.valid_tickers), end=end)
        valid_tickers = np.unique(prices['permno'])
        dates = np.unique(prices['date'])
        ret_df = pd.DataFrame(index=pd.DatetimeIndex(dates), dtype=float)

        for p in valid_tickers:
            prc = prices[prices['permno'] == p].set_index('date')
            prc = prc['prc'] / prc['cfacpr']
            prc = prc.sort_index(axis=0)
            ret = prc.pct_change()
            ret.name = p
            ret_df = ret_df.join(ret)

        ret_df = ret_df.dropna(axis=0, how='all').dropna(axis=1, how='any')
        self.valid_tickers = ret_df.columns
        return ret_df.mean(), ret_df.cov()

    def _optimize(self, mu, cov, r):
        min_ret = self.min_ret

        def obj(w):
            return w.T.dot(cov).dot(w)

        def cons_ret(w):
            return r + (mu - r).T.dot(w) - min_ret

        def cons_w(w):
            return np.sum(w) - 1

        N = mu.shape[0]
        bounds = None
        w_init = np.repeat(1/N, N)
        cons = [{'type': 'ineq', 'fun': cons_ret}, {'type': 'eq', 'fun': cons_w}]
        sol = minimize(fun=obj, x0=w_init, method='SLSQP', constraints=cons, bounds=bounds)
        return sol.x

    def on_rebal_day(self):
        current_date = self.portfolio.current_date.strftime('%Y-%m-%d')
        self.portfolio.clear_holdings()
        mu, cov = self._estimate_params()
        tickers = mu.index

        mu, cov = mu.values, cov.values
        nearest_dt = loc_nearest(current_date, self.portfolio.data.dates_f).strftime('%Y-%m-%d')
        r = self.portfolio.data.get_macro(nearest_dt)
        r = r.loc['tsy'] / 100

        r = (1 + r) ** (1 / 252) - 1

        weights = self._optimize(mu, cov, r)
        print(np.sum(weights))

        order = dict(zip(tickers, weights))
        self.portfolio.trade(order, is_permno=True)

    def on_day_close(self):
        pass


if __name__ == "__main__":
    start_date, end_date = '2006-01-31', '2020-12-30'
    starting_capital = 1000000
    rebal_freq = 'M'
    top = 1000  # Select 1000 most liquid stocks (by trading volume) each rebalancing period
    min_ret = 0.30
    lb_window = 252
    m_30 = Markowitz(
        start_date=start_date, end_date=end_date, min_ret=min_ret, starting_capital=starting_capital,
        rebal_freq=rebal_freq, lb_window=lb_window, top=top
    )
    m_30.run()
