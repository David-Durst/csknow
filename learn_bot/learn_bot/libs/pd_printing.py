import pandas as pd


def set_pd_print_options():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
