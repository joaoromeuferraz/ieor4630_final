from portfolio import *
from functools import wraps


class BaseStrategy:
    def __init__(self, start_date, end_date, starting_capital, rebal_freq='M', top=None):
        self.start_date = start_date
        self.end_date = end_date
        self.top = top

        self.portfolio = Portfolio(start_date=start_date, starting_capital=starting_capital)
        rebal_dates = self.portfolio.dates[self.portfolio.dates <= end_date]
        rebal_dates = pd.Series(dtype=float, index=rebal_dates).resample(rebal_freq).last().index
        self.rebal_dates = rebal_dates
        self.valid_tickers = None
        self.prices = None

    def _pre_run(self):
        current_date = self.portfolio.current_date.strftime('%Y-%m-%d')
        if self.top is None:
            valid_tickers = self.portfolio.data.get_valid_tickers(current_date)
        else:
            valid_tickers = self.portfolio.data.get_tickers_by_liquidity(current_date, top=self.top)

        self.valid_tickers = valid_tickers
        self.prices = self.portfolio.data.get_prices(date=current_date, permno=list(self.valid_tickers), idx='permno')

    def _post_run(self):
        pass

    def _run_wrapper(func):
        @wraps(func)
        def wrapper(self):
            self._pre_run()
            func(self)
            self._post_run()
        return wrapper

    def on_day_close(self):
        return NotImplementedError("Function 'on_day_close'  must be implemented")


    def on_rebal_day(self):
        return NotImplementedError("Function 'on_rebal_day' must be implemented")

    @_run_wrapper
    def _run_day(self):
        if self.portfolio.current_date in self.rebal_dates:
            self.on_rebal_day()
        self.on_day_close()
        self.portfolio.update_nav()

    def run(self):
        end_nearest = loc_nearest(self.end_date, self.portfolio.dates)
        while self.portfolio.current_date <= end_nearest:
            print(f"Current date: {self.portfolio.current_date.strftime('%Y-%m-%d')}")
            self._run_day()
        print("Finished running strategy")
        print(self.portfolio.summary)

