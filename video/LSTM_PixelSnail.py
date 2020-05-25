from torch import nn
from video.ConvLstmCell import ConvLstmCell
import torch.nn.functional as F


class LSTM_PixelSnail(nn.Module):

    def __init__(self, lstm_laysers, lstm_model, pixel_model):
        super().__init__()
        self.type = type
        self.pixel_model = pixel_model
        self.lstm_model = lstm_model
        self.lstm_layers = lstm_laysers

    def forward(self, inputs_,cells_state):

        input_ = inputs_[:,0,:,:]
        target = inputs_[:, 1, :, :]
        out,cells_state = self.lstm_model(input_, cells_state)
        out, _ = self.pixel_model(target, condition=cells_state[0])

        return out, cells_state
