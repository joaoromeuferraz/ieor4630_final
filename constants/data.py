WRDS_TABLES = {
    'daily_price': ('crsp_q_stock', 'dsf'),
    'ratios': ('wrdsapps_finratio', 'firm_ratio'),
    'id': ('wrdsapps_finratio', 'id'),
    'factors': ('ff', 'factors_daily')
}

ID_FIELDS_RATIOS = ['permno']

FIELDS_RATIOS = [
    'capei', 'be', 'bm', 'evm', 'pe_op_basic', 'ps', 'pcf', 'npm', 'gpm', 'cfm', 'roa', 'roe', 'gprof',
    'capital_ratio', 'cash_lt', 'debt_ebitda', 'short_debt', 'curr_debt', 'lt_debt', 'cash_debt', 'debt_assets',
    'de_ratio', 'intcov_ratio', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'at_turn', 'accrual'
]

ID_FIELDS_PRICE = ['permno', 'date']

FIELDS_PRICE = ['prc', 'vol', 'cfacpr', 'shrout']

ID_FIELDS_FACTORS = ['date']

FIELDS_FACTORS = ['mktrf', 'smb', 'hml', 'umd']

FIELDS_MOMENTUM = ['mom1m', 'mom3m', 'mom6m', 'mom12m', 'mom36m']
FIELDS_BETAS = ['alpha', 'beta_mkt', 'beta_smb', 'beta_hml', ' beta_umd']
FIELDS_OTHER = ['ret_vol', 'std_dolvol', 'mv']
FIELDS_MACRO = ['sp500_20', 'tsy_20']

ALL_CHARS = FIELDS_RATIOS + FIELDS_MOMENTUM + FIELDS_BETAS + FIELDS_OTHER
LABEL_COL = '20d_ret'

OUTPUT_DATA_DIR = 'data/outputs/'
DATES_D_PATH = 'data/dates_d.csv'
DATES_M_PATH = 'data/dates_m.csv'
DATES_F_PATH = 'data/dates_f.csv'
MACRO_PATH = 'data/raw_data/macro.csv'
