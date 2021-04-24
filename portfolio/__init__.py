from data import *
from constants.portfolio import *
from data.helpers import *


class Portfolio:

    def __init__(self, start_date, starting_capital=1000):
        self.start_date = start_date
        self.data = Data()
        self.dates = self.data.dates_d[self.data.dates_d >= start_date]
        self.starting_capital = float(starting_capital)

        self.current_idx = 0
        self.current_date = self.dates[self.current_idx]

        self.returns = pd.DataFrame(columns=RETURNS_COLUMNS)
        self.returns.index.name = 'date'

        self.cash = self.starting_capital
        self.mv = 0.0
        self.nav = self.starting_capital

        self.cash_series = pd.Series([self.cash], index=[self.current_date])
        self.mv_series = pd.Series([self.mv], index=[self.current_date])
        self.nav_series = pd.Series([self.nav], index=[self.current_date])

        self.summary = pd.DataFrame(columns=SUMMARY_COLUMNS)

        self.active_holdings = pd.DataFrame(columns=HOLDINGS_COLUMNS)
        self.active_holdings.index.name = 'permno'

        self.historical_holdings = pd.DataFrame(columns=HISTORICAL_HOLDINGS_COLUMNS)
        self.trade_log = pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    def trade(self, order: dict, is_permno=True, prices=None):
        """
        :param order: dictionary with keys as tickers and values as target weights
            >>> order = {'AAPL': 0.5, 'GME': 0.5}
        :param is_permno: if keys of orders are given as permno (as opposed to stock tickers)
        :param prices: dataframe with trade prices
        :return:
        """
        date = self.current_date.strftime('%Y-%m-%d')

        if not is_permno:
            tickers = list(order.keys())
            permnos = self.data.get_id(tickers=tickers)
            permnos = list(permnos['permno'].values)
            ref = dict(zip(permnos, tickers))
        else:
            permnos = list(order.keys())
            ref = dict(zip(permnos, permnos))

        permnos = list(np.unique(permnos))

        if prices is None:
            prices = self.data.get_prices(date, permno=list(permnos), idx='permno')
            prices = prices.dropna()
            prices = prices.loc[:, 'prc']/prices.loc[:, 'cfacpr']

        for p in permnos:
            price = prices.loc[p] if p in prices.index else self.active_holdings.loc[p, 'price']
            weight = order[ref[p]]
            target_mv = self.nav * weight
            qty = target_mv/price

            prev_qty = self.active_holdings.loc[p]['qty'] if p in self.active_holdings.index else 0.
            qty_delta = qty - prev_qty

            mv = qty_delta * price
            if np.isnan(mv):
                mv = 0.0

            trade = pd.Series([date, p, qty_delta], index=TRADE_LOG_COLUMNS)
            self.trade_log = self.trade_log.append(trade, ignore_index=True)

            holding = pd.Series([qty, price], index=HOLDINGS_COLUMNS, name=p)

            if p in self.active_holdings.index:
                if target_mv == 0.:
                    self.active_holdings = self.active_holdings.drop(p, axis=0)
                else:
                    self.active_holdings.loc[p] = holding
            else:
                if not target_mv == 0.:
                    self.active_holdings = self.active_holdings.append(holding)

            self.mv += mv
            self.cash -= mv

    def clear_holdings(self):
        if len(self.active_holdings) > 0:
            permnos = list(self.active_holdings.index)
            weights = [0.0 for _ in range(len(permnos))]
            order = dict(zip(permnos, weights))
            self.trade(order=order, is_permno=True)

    def update_nav(self):
        current_date = self.current_date.strftime('%Y-%m-%d')
        print(f"Updating portfolio NAV: {current_date}")

        active_tickers = self.active_holdings.index
        prev_prices = self.active_holdings.loc[:, 'price']

        permno = list(active_tickers) if len(active_tickers) > 0 else None

        prices = self.data.get_prices(current_date, permno=permno, idx='permno')
        prices = prices.loc[:, 'prc']/prices.loc[:, 'cfacpr']

        valid_ticks = []
        for p in active_tickers:
            valid = p in prices.index and not np.isnan(prices.loc[p])
            if not valid:
                self.trade(order={p: 0.0}, is_permno=True, prices=self.active_holdings.loc[[p], 'price'])
            else:
                valid_ticks.append(p)

        prices = prices.loc[valid_ticks]
        for p in prices.index:
            price_delta = prices.loc[p] - prev_prices.loc[p]
            if np.isnan(price_delta):
                price_delta = 0.
            qty = self.active_holdings.loc[p, 'qty']
            mv_delta = price_delta*qty
            if np.isnan(mv_delta):
                mv_delta = 0.
            self.mv += mv_delta

        for p in self.active_holdings.index:
            updated_holdings = self.active_holdings.loc[p]
            if p in prices.index:
                updated_holdings['price'] = prices.loc[p]
                self.active_holdings.loc[p] = updated_holdings

        holdings = self.active_holdings.reset_index()
        holdings['date'] = pd.Series(np.repeat(self.current_date, len(holdings)), index=holdings.index)
        self.historical_holdings = self.historical_holdings.append(holdings, ignore_index=True)

        self.nav = self.cash + self.mv

        self.cash_series = self.cash_series.append(pd.Series([self.cash], index=[self.current_date]))
        self.mv_series = self.mv_series.append(pd.Series([self.mv], index=[self.current_date]))
        self.nav_series = self.nav_series.append(pd.Series([self.nav], index=[self.current_date]))

        summary = pd.Series([self.cash, self.mv, self.nav], index=SUMMARY_COLUMNS, name=self.current_date)
        print(summary)
        self.summary = self.summary.append(summary)
        self.next_day()
        print("-----------------------------------------")

    def get_trades(self, date):
        res = self.trade_log[self.trade_log['date'] == date]
        return res.set_index('permno').loc[:, ['qty']]

    def next_day(self):
        self.current_idx += 1
        self.current_date = self.dates[self.current_idx]

    def run_until(self, date):
        nearest = loc_nearest(date, self.dates)
        while self.current_date <= nearest:
            self.update_nav()

