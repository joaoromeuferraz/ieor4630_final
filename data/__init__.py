import wrds
from constants.data import *
from data.helpers import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


class Data:
    def __init__(self, wrds_username='jrferraz', freq='M', output_dir=None,
                 fields_ratios=None, fields_factors=None, all_chars=None):
        self.freq = freq
        self.db = wrds.Connection(wrds_username=wrds_username)

        self.fields_ratios = fields_ratios or FIELDS_RATIOS
        self.fields_factors = fields_factors or FIELDS_FACTORS
        self.fields_price = FIELDS_PRICE
        self.all_chars = all_chars or ALL_CHARS

        dates_d = pd.read_csv(DATES_D_PATH)
        dates_m = pd.read_csv(DATES_M_PATH)
        dates_f = pd.read_csv(DATES_F_PATH)

        self.dates_d = pd.DatetimeIndex(dates_d['date'])
        self.dates_m = pd.DatetimeIndex(dates_m['date'])
        self.dates_f = pd.DatetimeIndex(dates_f['date'])

        macro = pd.read_csv(MACRO_PATH)
        macro = macro.set_index('date')
        macro.index = pd.to_datetime(macro.index)
        self.macro = macro

        self.output_dir = output_dir or OUTPUT_DATA_DIR
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def get_features(self, date, tickers=None, fill_na=False):
        """

        :param date: date in format YYYY-MM-DD (str)
        :param tickers: list of relevant stock tickers (optional)
        :param fill_na: if True, then replace nans with cross-sectional median
        :return:

        """
        if tickers:
            permno = self.get_id(tickers=tickers)
            permno = list(permno.values)
        else:
            permno = None

        ratio_date = loc_nearest(date, self.dates_m)
        price_date = loc_nearest(ratio_date, self.dates_d)
        factor_date = loc_nearest(price_date, self.dates_f)

        # Getting all financial ratios, price data, and factor data
        print(f"Loading financial ratios...")
        ratios = self._get_ratios(ratio_date, permno=permno)
        print(f"Loading historical prices...")
        prices = self.get_prices(price_date, permno=permno, idx='permno')
        print(f"Loading factors...")
        factors = self._get_factors(price_date)

        prices = prices.drop(columns=['date'])

        assert len(factors) > 0, f"No factor portfolio data found for {date}"

        # Handling missing values
        if fill_na:
            ratios = self._handle_missing(ratios)
            prices = self._handle_missing(prices)
        else:
            ratios = ratios.dropna(axis=0, how='any')
            prices = prices.dropna(axis=0, how='any')

        prices = prices[~prices.index.duplicated(keep='first')]
        ratios = ratios[~ratios.index.duplicated(keep='first')]

        # Adjusting price series
        prc_adj = prices.loc[:, 'prc'] / prices.loc[:, 'cfacpr']
        prc_adj.name = 'prc_adj'
        mv = prices.loc[:, 'shrout'] * prices.loc[:, 'prc']
        mv.name = 'mv'
        prices = pd.concat((prices.loc[:, ['vol']], prc_adj, mv), axis=1)

        print(f"Loading additional features...")
        mom_facts = self._get_momentum(price_date, permno=permno, cur_prices=prices)
        other = self._get_other(price_date, permno=permno)
        print("Done")
        print("")

        all_features = ratios.join(mom_facts)
        all_features = all_features.join(other)
        all_features = self._handle_missing(all_features) if fill_na else all_features.dropna(axis=0, how='any')
        all_features = self._rank_features(all_features)
        return all_features

    def get_labels(self, date, permno=None, horizon=20):
        price_date = loc_nearest(date, self.dates_d)
        cur_idx = np.where(self.dates_d == price_date)[0][0]
        end = self.dates_d[cur_idx + horizon].strftime('%Y-%m-%d')
        prices_start = self.get_prices(price_date, permno=permno, idx='permno')
        prices_end = self.get_prices(end, permno=permno, idx='permno')
        print("Loading labels...")

        if permno is None:
            permno = list(prices_start.index)

        cols = [f"{horizon}d_ret"]
        labels = pd.DataFrame(index=permno, columns=cols, dtype=float)
        for p in permno:
            if (p in prices_end.index) and (p in prices_start.index):
                prc_1 = prices_start.loc[p, 'prc'] / prices_start.loc[p, 'cfacpr']
                prc_2 = prices_end.loc[p, 'prc'] / prices_end.loc[p, 'cfacpr']
                ret = prc_2 / prc_1 - 1
                labels.loc[p] = pd.Series([ret], index=cols)
            else:
                labels.loc[p] = pd.Series([np.nan], index=cols)
        print("Done")
        return labels

    def get_data(self, date, tickers=None, fill_na=True, horizon=20):
        features = self.get_features(date, tickers=tickers, fill_na=fill_na)
        permno = list(features.index)
        labels = self.get_labels(date, permno=permno, horizon=horizon)
        labels = labels.dropna()
        features = features.reindex(labels.index, method=None)

        return features, labels

    def get_all_data(self, start_date, end_date=None,  fill_na=False, horizon=20, save=False):
        dates = self.dates_m[self.dates_m >= start_date]
        if end_date:
            dates = dates[dates <= end_date]
        all_x, all_y = None, None
        for i, d in enumerate(dates):
            print("------------------------------------------")
            print(f"Generating features for {d.strftime('%Y-%m-%d')}")
            features, labels = self.get_data(d.strftime('%Y-%m-%d'), fill_na=fill_na, horizon=horizon)
            df = pd.concat((features, labels), axis=1)
            if save:
                df.to_csv(os.path.join(self.output_dir, d.strftime('%Y-%m-%d') + ".csv"))
            if all_x is None:
                all_x = features.values
                all_y = labels.values
            else:
                all_x = np.concatenate((all_x, features.values), axis=0)
                all_y = np.concatenate((all_y, labels.values), axis=0)
            print("")
        return all_x, all_y

    @staticmethod
    def _rank_features(df):
        ranked = df.rank()
        ranked = ranked / (len(ranked) / 2) - 1  # mapping features to [-1,1] interval based on ranking
        return ranked

    def _get_other(self, date, permno=None, window=60):
        assert np.where(self.dates_d == date) is not None, f"No price data for date {date}"
        ref_date = self.dates_d[np.where(self.dates_d == date)[0][0] - window].strftime('%Y-%m-%d')
        prices = self.get_prices(ref_date, permno=permno, end=date)
        factors = self._get_factors(ref_date, date)
        if permno is None:
            permno = list(prices[prices['date'] == date].loc[:, 'permno'].values)
        betas = pd.DataFrame(index=permno, columns=FIELDS_BETAS, dtype=float)
        others = pd.DataFrame(index=permno, columns=FIELDS_OTHER, dtype=float)
        betas.index.name = 'permno'
        others.index.name = 'permno'
        for p in permno:
            prc = prices[prices['permno'] == p].loc[:, ['date', 'prc', 'cfacpr', 'vol', 'shrout']].set_index('date')
            prc = prc.dropna(axis=0, how='any')
            if len(prc) >= 50:
                prc_adj = prc.loc[:, 'prc'] / prc.loc[:, 'cfacpr']
                prc_adj = prc_adj.pct_change(periods=1)
                prc_adj.name = 'y'
                sub_data = pd.concat((prc_adj, factors), axis=1)
                sub_data = sub_data.dropna()
                if len(sub_data) >= 50:
                    m = LinearRegression(fit_intercept=True)
                    m.fit(sub_data.loc[:, ['mktrf', 'smb', 'hml', 'umd']], sub_data.loc[:, 'y'])
                    params = [m.intercept_] + list(m.coef_)
                    betas.loc[p] = pd.Series(params, index=FIELDS_BETAS)

                    others_res = [prc_adj.std(), prc.loc[:, 'vol'].std(), prc.iloc[-1]['shrout'] * prc.iloc[-1]['prc']]
                    others.loc[p] = pd.Series(others_res, index=FIELDS_OTHER)
                else:
                    betas.loc[p] = pd.Series(np.repeat(np.nan, len(FIELDS_BETAS)), index=FIELDS_BETAS)
                    others.loc[p] = pd.Series(np.repeat(np.nan, len(FIELDS_OTHER)), index=FIELDS_OTHER)
            else:
                betas.loc[p] = pd.Series(np.repeat(np.nan, len(FIELDS_BETAS)), index=FIELDS_BETAS)
                others.loc[p] = pd.Series(np.repeat(np.nan, len(FIELDS_OTHER)), index=FIELDS_OTHER)

        df = pd.concat((betas, others), axis=1)
        return df

    def get_tickers_by_liquidity(self, date, top=1000):
        lib, tb = WRDS_TABLES['daily_price']
        q = f"SELECT permno FROM {lib}.{tb} WHERE date = '{date}' and prc is not null ORDER BY vol desc limit {top};"
        df = self.db.raw_sql(q)
        return pd.Index(df['permno'].values).drop_duplicates()

    def get_macro(self, date, end=None):
        date = loc_nearest(date, self.macro.index)
        if end is not None:
            end = loc_nearest(date, self.macro.index)
        df = self.macro.loc[date] if end is None else self.macro.loc[date:end]
        return df

    def _get_momentum(self, date, permno=None, windows=None, cur_prices=None):
        assert np.where(self.dates_d == date) is not None, f"No price data for date {date}"
        windows = windows or [20, 60, 120, 250, 750]

        idx = np.where(self.dates_d == date)[0][0]
        ref_dates = [self.dates_d[idx - w].strftime('%Y-%m-%d') for w in windows]
        if cur_prices is None:
            cur_prices = self.get_prices(date, permno, idx='permno')
            cur_prices = cur_prices.loc[:, 'prc'] / cur_prices.loc[:, 'cfacpr']
            cur_prices.name = 'prc_adj'
        else:
            cur_prices = cur_prices.loc[:, 'prc_adj']

        permno = list(cur_prices.index)

        mom_facts = []
        for i, r in enumerate(ref_dates):
            prc = self.get_prices(r, permno, idx='permno')
            prc_adj = prc.loc[:, 'prc'] / prc.loc[:, 'cfacpr']
            ret = cur_prices / prc_adj - 1
            ret.name = FIELDS_MOMENTUM[i]
            mom_facts.append(ret)
        mom_facts = pd.concat(mom_facts, axis=1)
        return mom_facts

    def _get_factors(self, date, end=None):
        lib, tb = WRDS_TABLES['factors']
        fields = ID_FIELDS_FACTORS + self.fields_factors
        end = end or date
        args = [{'type': '>=', 'col': 'date', 'val': date}, {'type': '<=', 'col': 'date', 'val': end}]
        df = self._select(lib, tb, fields=fields, args=args)
        df = df.set_index('date')
        return df

    @staticmethod
    def _handle_missing(df):
        fill_vals = df.median().to_dict()
        return df.fillna(fill_vals)

    def _get_ratios(self, date, permno=None):
        """
        Retrieves financial ratios from database
        :param date: relevant date
        :param permno: list of PERMNOs of relevant stocks
        :return:
        """
        lib, tb = WRDS_TABLES['ratios']
        fields = ID_FIELDS_RATIOS + self.fields_ratios
        args = [{'type': '=', 'col': 'public_date', 'val': date}]
        if permno is not None:
            args.append({'type': 'in', 'col': 'permno', 'val': permno})
        df = self._select(lib, tb, fields=fields, args=args)
        df = df.set_index('permno')
        return df

    def get_id(self, tickers=None, permnos=None):
        assert permnos is not None or tickers is not None, "Must provide list of tickers or permnos"
        ref = tickers if tickers is not None else permnos
        lib, tb = WRDS_TABLES['id']
        fields = ['permno', 'ticker']
        col = 'ticker' if tickers else 'permno'
        args = [{'type': 'in', 'col': col, 'val': ref}]
        df = self._select(lib, tb, fields=fields, args=args)
        df = df.set_index('ticker') if tickers else df.set_index('permno')
        return df

    def get_prices(self, date, permno=None, end=None, idx=None):
        lib, tb = WRDS_TABLES['daily_price']
        fields = ID_FIELDS_PRICE + self.fields_price
        end = end or date
        args = [{'type': '>=', 'col': 'date', 'val': date}, {'type': '<=', 'col': 'date', 'val': end}]
        if permno is not None:
            args.append({'type': 'in', 'col': 'permno', 'val': permno})
        df = self._select(lib, tb, fields=fields, args=args)
        if idx is not None:
            df = df.set_index(idx)
        df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'])
        return df

    def get_valid_tickers(self, date):
        lib, tb = WRDS_TABLES['daily_price']
        q = f"SELECT DISTINCT permno FROM {lib}.{tb} where date = '{date}' and prc is not Null"
        df = self.db.raw_sql(q)['permno']
        return pd.Index(df)

    def _select(self, lib, tb, fields=None, args=None):
        """
        Retrieves data from database
        :param lib: Library name
        :param tb: Table name
        :param fields: Name of columns to select
        :param args: Conditions for selection. Ex:
        >>> args = [{'type': '=', 'col': 'date', 'val': '2019-01-01'},
        >>>         {'type': 'in', 'col': 'tickers', 'val': ['AAPL', 'TSLA', 'AA']}]
        :return:
        """
        if fields:
            fields = ",".join(fields)
        else:
            fields = "*"

        q = f"SELECT {fields} FROM {lib}.{tb}"
        if args:
            q += " WHERE "
            args_list = []
            for d in args:
                s = f"{d['col']} {d['type']} "
                if d['type'] == 'in':
                    val = [f"'{x}'" for x in d['val']]
                    val = ",".join(val)
                    val = f"({val})"
                    s += f"{val}"
                else:
                    s += f"'{d['val']}'"
                args_list.append(s)
            q += " AND ".join(args_list)

        q += ";"
        df = self.db.raw_sql(q)
        return df


def load_features(path=None, features_col=None, label_col=None, macro_col=None):
    """

    :param path: directory that contains csv files
    :param label_col: name of column containing label
    :param features_col: name of columns containing characteristics
    :param macro_col: name of columns for macro variables
    :return:
    """
    path = path or OUTPUT_DATA_DIR
    if features_col is None:
        features_col = ALL_CHARS
    if label_col is None:
        label_col = LABEL_COL
    if macro_col is None:
        macro_col = FIELDS_MACRO

    macro_df = pd.read_csv(MACRO_PATH)
    macro_df = macro_df.set_index('date')
    macro_df.index = pd.to_datetime(macro_df.index)

    all_labels, all_features, all_dates = None, None, None

    print(f"Loading features...")
    fnames = os.listdir(path)
    for f in fnames:
        cur_date = f[:-4]
        df = pd.read_csv(os.path.join(path, f))
        df = df.set_index(df.columns[0])
        df.index.name = 'permno'
        labels = df.loc[:, label_col].values
        chars = df.loc[:, features_col].values

        try:
            nearest_dt = loc_nearest(cur_date, macro_df.index)
            macro = macro_df.loc[nearest_dt.strftime('%Y-%m-%d'), macro_col].values
            macro = np.concatenate((np.ones(1), macro), axis=0)

            features = np.empty((chars.shape[0], macro.shape[0]*chars.shape[1]))
            for i in range(chars.shape[0]):
                tick_f = chars[i]
                res = []
                for j in range(tick_f.shape[0]):
                    for k in range(macro.shape[0]):
                        res.append(macro[k]*tick_f[j])
                features[i] = np.array(res)

            dates = np.empty(labels.shape[0], dtype='<U10')
            dates[:] = cur_date

            if all_labels is None:
                all_labels = labels
                all_features = features
                all_dates = dates
            else:
                all_labels = np.concatenate((all_labels, labels), axis=0)
                all_features = np.concatenate((all_features, features), axis=0)
                all_dates = np.concatenate((all_dates, dates), axis=0)
        except IndexError:
            # print(f"No macro data for {cur_date}")
            pass

    print(f"Done")

    return all_features, all_labels, all_dates


if __name__ == "__main__":
    d = Data(output_dir='data/outputs_5/')
    X, y = d.get_all_data('2006-01-01', None, fill_na=True, horizon=5, save=True)

