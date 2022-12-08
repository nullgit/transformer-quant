from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import akshare as ak
from torch import Tensor
from torch import nn
from torch.nn import LSTM, Linear
from torch.optim import Adam


def load_data(code: str) -> np.ndarray:
    # data = np.arange(100)
    data_path = f'data/ak/stock_zh_{code}.npy'
    if not os.path.exists(data_path):
        data = ak.stock_zh_index_daily(symbol=code)['close'].to_numpy()
        np.save(data_path, data)
    else:
        data = np.load(data_path)
    return data


device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

# TRAIN_CODE = '100'
TRAIN_CODE = 'sh000300'
EVAL_CODE = 'sh000300'
EVAL = False
# EVAL = True
WINDOW_SIZE = 30
PRED_DAY = 1
EPOCH = 3
MODEL_PATH = f'lstm_{TRAIN_CODE}_{EPOCH}.h5'
BATCH_SIZE = 32


class LSTMModel(nn.Module):
    def __init__(self) -> None:
        super(LSTMModel, self).__init__()

        self.num_layers = 1
        self.lstm1 = LSTM(input_size=WINDOW_SIZE, hidden_size=128, num_layers=self.num_layers)
        self.lstm2 = LSTM(input_size=128, hidden_size=64, num_layers=self.num_layers)
        self.fc1 = Linear(64, 25)
        self.fc2 = Linear(25, 1)
        # self.model = Sequential(
        #     LSTM(input_size=WINDOW_SIZE, hidden_size=128),
        #     LSTM(input_size=128, hidden_size=64),
        #     Linear(64, 25),
        #     Linear(25, 1),
        # )

    def forward(self, x) -> Tensor:
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_data_label_min_max(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = []
    y = []
    x_min = []
    x_max = []
    for i in range(len(data) - WINDOW_SIZE - PRED_DAY + 1):
        prev_n_x = data[i:i + WINDOW_SIZE]
        prev_n_max = max(prev_n_x)
        prev_n_min = min(prev_n_x)
        x_max.append([prev_n_max])
        x_min.append([prev_n_min])
        after_pred_day_price = data[i + WINDOW_SIZE + PRED_DAY - 1]
        prev_n_x = (prev_n_x - prev_n_min) / (prev_n_max - prev_n_min) * 2 - 1
        after_pred_day_price = (after_pred_day_price - prev_n_min) / (prev_n_max - prev_n_min) * 2 - 1
        y.append([after_pred_day_price])
        x.append(prev_n_x)
    return np.array(x), np.array(y), np.array(x_min), np.array(x_max)


if __name__ == '__main__':
    # all_data = load_data(TRAIN_CODE)
    all_data = load_data(TRAIN_CODE)[0:50]
    # all_data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    x, y, x_min, x_max = get_data_label_min_max(all_data)

    if not EVAL:
        train_data_len = int(len(x) * 0.8)
        train_x = x[:train_data_len]
        train_y = y[:train_data_len]
        test_x = x[train_data_len:]
        test_y = y[train_data_len:]
        model = LSTMModel().to(device)
        criterion = nn.BCELoss().to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)

        # Train the model
        model.train()
        for i in range(len(train_x)):
            x = torch.Tensor(train_x[i:i+1]).to(device)
            y = torch.Tensor(train_y[i:i+1]).to(device)

            output = model(x)
            optimizer.zero_grad()

            loss = criterion(output, y)
            loss.backward()
            print(i, loss)

            optimizer.step()

        # Eval
        model.eval()
        x = torch.Tensor(test_x).to(device)
        y = torch.Tensor(test_y).to(device)

        pred: Tensor = model(x)
        loss = criterion(pred, y)
        print(loss)

        # pred = pred * (x_max - x_min) + x_min
        # y = y * (x_max - x_min) + x_min
        line1, = plt.plot(pred.cpu().detach().numpy().reshape((-1)))
        line1.set_label('pred')
        line2, = plt.plot(y.cpu().detach().numpy().reshape((-1)))
        line2.set_label('real')
        plt.legend()
        plt.show()
