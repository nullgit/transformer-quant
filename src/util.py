
import numpy as np
import os
import akshare as ak


def load_data(code: str) -> np.ndarray:
    # data = np.arange(100)
    data_path = f'data/ak/stock_zh_{code}.npy'
    if not os.path.exists(data_path):
        data = ak.stock_zh_index_daily(symbol=code)['close'].to_numpy()
        np.save(data_path, data)
    else:
        data = np.load(data_path)
    return data
