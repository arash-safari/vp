from torch import nn
from video.ConvLstmCell import ConvLstmCell
import torch
import torch.nn.functional as F


class LSTM_CNN_VIDEOMNIST(nn.Module):

    def __init__(self, input_channel, hidden_channel, lstm_kernel_size, cnn_kernel_size, device):
        super().__init__()
        self.type = type
        self.device = device
        self.cnn = nn.Conv2d(
            hidden_channel,
            input_channel,
            cnn_kernel_size,
            stride=1,
            padding=cnn_kernel_size // 2,
        )
        self.lstm1 = ConvLstmCell(input_channel, hidden_channel, 3, device)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lstm2 = ConvLstmCell(hidden_channel, hidden_channel, 3, device)
        self.bn2 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lstm3 = ConvLstmCell(hidden_channel, hidden_channel, 3, device)
        self.bn3 = nn.BatchNorm2d(num_features=hidden_channel)

    def forward(self, inputs_, cells_state=[None] * 3):
        for input_ in inputs_:
            cells_state[0] = self.lstm1(input_, cells_state[0])
            h0, _ = cells_state[0]

            cells_state[1] = self.lstm1(self.bn1(F.relu(h0)), cells_state[1])
            h1, _ = cells_state[1]
            cells_state[2] = self.lstm1(self.bn2(F.relu(h1)), cells_state[2])
            h2, _ = cells_state[2]

        return self.cnn(self.bn3(h2))
