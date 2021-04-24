from strategy import BaseStrategy
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParity(BaseStrategy):
    def __init__(self, start_date, end_date, max_vol=0.20, starting_capital=1000000,
                 rebal_freq='M', lb_window=252, top=1000):
        super().__init__(start_date=start_date, end_date=end_date,
                         starting_capital=starting_capital, rebal_freq=rebal_freq, top=top)
        self.max_vol = max_vol
        self.lb_window = lb_window
        self.max_vol /= np.sqrt(252)

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
        cum_df = (ret_df + 1).cumprod(axis=0) - 1

        self.valid_tickers = ret_df.columns
        return ret_df.cov(), cum_df.iloc[-1]

    def _optimize(self, cov, scores=None):
        raise NotImplementedError("Subclasses must implement optimize")

    def on_rebal_day(self):
        raise NotImplementedError("Subclasses must implement on_rebal_day")

    def on_day_close(self):
        pass


class RiskParityLongOnly(RiskParity):
    def __init__(self, start_date, end_date, max_vol=0.20, starting_capital=1000000,
                 rebal_freq='M', lb_window=252, top=1000):
        super().__init__(start_date=start_date, end_date=end_date, max_vol=max_vol, starting_capital=starting_capital,
                         rebal_freq=rebal_freq, lb_window=lb_window, top=top)

    def _optimize(self, cov, scores=None):
        max_vol = self.max_vol

        def obj(w):
            return -np.sum(np.log(w))

        def cons_vol(w):
            return max_vol - np.sqrt(w.T.dot(cov).dot(w))

        vols = np.sqrt(np.diag(cov))[:, np.newaxis]
        w_init = (1 / vols) / (np.sum(1 / vols))
        cons = [{'type': 'ineq', 'fun': cons_vol}]
        sol = minimize(fun=obj, x0=w_init, method='SLSQP', constraints=cons)
        return sol.x

    def on_rebal_day(self):
        self.portfolio.clear_holdings()
        cov, _ = self._estimate_params()
        weights = self._optimize(cov)
        weights /= np.sum(weights)
        order = dict(zip(self.valid_tickers, weights))
        self.portfolio.trade(order, is_permno=True)


class RiskParityLongShort(RiskParity):
    def __init__(self, start_date, end_date, max_vol=0.20, starting_capital=1000000,
                 rebal_freq='M', lb_window=252, top=1000):
        super().__init__(start_date=start_date, end_date=end_date, max_vol=max_vol, starting_capital=starting_capital,
                         rebal_freq=rebal_freq, lb_window=lb_window, top=top)

    def _optimize(self, cov, scores=None):
        max_vol = self.max_vol

        def obj(w):
            return -np.sum(np.abs(scores) * np.log(w))

        def cons_vol(w):
            return max_vol - np.sqrt(w.T.dot(cov).dot(w))

        def cons_long(w):
            return 1.0 - np.sum(w[w > 0])

        def cons_short(w):
            return np.sum(w[w < 0]) - 1.0

        vols = np.sqrt(np.diag(cov))
        w_init = np.sign(scores) * ((1 / vols) / (np.sum(1 / vols)))

        cons = [{'type': 'ineq', 'fun': cons_vol}, {'type': 'ineq', 'fun': cons_long},
                {'type': 'ineq', 'fun': cons_short}]
        sol = minimize(fun=obj, x0=w_init, method='SLSQP', constraints=cons)
        return sol.x

    def on_rebal_day(self):
        self.portfolio.clear_holdings()
        cov, ret = self._estimate_params()
        weights = self._optimize(cov, scores=ret.values)
        abs_weights = np.sum(np.abs(weights))
        weights /= abs_weights
        order = dict(zip(self.valid_tickers, weights))
        self.portfolio.trade(order, is_permno=True)


if __name__ == "__main__":
    start_date, end_date = '2006-01-31', '2020-12-30'
    starting_capital = 1000000
    rebal_freq = 'M'
    top = 1000  # Select 1000 most liquid stocks (by trading volume) each rebalancing period
    lb_window = 252
    max_vol = 0.50
    rp_ls_50 = RiskParityLongShort(
        start_date=start_date, end_date=end_date, max_vol=max_vol, starting_capital=starting_capital,
        rebal_freq=rebal_freq, lb_window=lb_window, top=top
    )
    rp_ls_50.run()



