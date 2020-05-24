from torch import nn
from video.ConvLstmCell import ConvLstmCell
import torch


class LSTM_CNN_VIDEOMNIST(nn.Module):

    def __init__(self, input_channel, hidden_channel, lstm_kernel_size, cnn_kernel_size, device):
        super().__init__()
        self.type = type
        self.lstm = ConvLstmCell(input_channel, hidden_channel, lstm_kernel_size, device)
        self.device = device
        self.cnn = nn.Conv2d(
            hidden_channel,
            input_channel,
            cnn_kernel_size,
            stride=1,
            padding=cnn_kernel_size // 2,
        )

    def forward(self, frames, cell_state=None):
        result = torch.Tensor()

        for frame in frames:
            input_ = frame.copy()
            cell_state = self.lstm(input_, cell_state)
            hidden, cell = cell_state
            result.append(self.cnn(hidden))

        return result
