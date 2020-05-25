from torch import nn
from video.ConvLstmCell import ConvLstmCell
import torch.nn.functional as F


class LSTM3(nn.Module):

    def __init__(self, input_channel, hidden_channel, device):
        super().__init__()
        self.type = type
        self.device = device
        self.lstm1 = ConvLstmCell(input_channel, hidden_channel, 3, device)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lstm2 = ConvLstmCell(hidden_channel, hidden_channel, 3, device)
        self.bn2 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lstm3 = ConvLstmCell(hidden_channel, hidden_channel, 3, device)
        self.bn3 = nn.BatchNorm2d(num_features=hidden_channel)

    def forward(self, input_, cells_state=None):
        if cells_state is None:
            cells_state = [None] * 3
        cells_state[0] = self.lstm1(input_, cells_state[0])
        h0, _ = cells_state[0]
        cells_state[1] = self.lstm2(self.bn1(F.relu(h0)), cells_state[1])
        h1, _ = cells_state[1]
        cells_state[2] = self.lstm3(self.bn2(F.relu(h1)), cells_state[2])
        h2, _ = cells_state[2]

        return h2, cells_state
