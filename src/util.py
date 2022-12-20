import os

import akshare as ak
import numpy as np
import pandas as pd
import pickle
import torch


def load_stock_data(code: str) -> pd.DataFrame:
    # data = np.arange(100)
    data_path = f'data/ak/stock_zh_{code}.pkl'
    if not os.path.exists(data_path):
        data = ak.stock_zh_index_daily(symbol=code)
        with open(data_path, 'wb') as fp:
            pickle.dump(data, fp)
    else:
        with open(data_path, 'rb') as fp:
            data = pickle.load(fp)
    return data

def get_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return device
    