from torch import nn
from video.ConvLstmCell import ConvLstmCell
import torch.nn.functional as F


class LSTM_PixelSnail(nn.Module):

    def __init__(self, lstm_model, cnn_model, pixel_model):
        super().__init__()
        self.type = type
        self.pixel_model = pixel_model
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model

    def forward(self, inputs_, cells_state):
        input_ = inputs_[:, 0, :, :]
        target = inputs_[:, 1, :, :]
        lstm_out, cells_state = self.lstm_model(input_, cells_state)
        cnn_out = self.cnn_model(lstm_out)
        out, _ = self.pixel_model(target, condition=cnn_out)

        return out, cells_state
