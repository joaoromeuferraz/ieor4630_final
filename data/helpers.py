import pandas as pd


def loc_nearest(date, idx_arr, method='forward'):
    if method == 'forward':
        return idx_arr[idx_arr <= date][-1]
    elif method == 'backward':
        return idx_arr[idx_arr >= date][0]
    else:
        raise ValueError(f"Method {method} is unavailable")
