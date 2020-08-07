from torch import nn
from video.ConvLstmCell import ConvLstmCell
import torch


class LSTM_PixelSnail3(nn.Module):
    def _to_one_hot(self, y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to('cuda')

        return zeros.scatter(scatter_dim, y_tensor, 1)

    def __init__(self, lstm_model, pixel_model):
        super().__init__()
        self.type = type
        self.pixel_model = pixel_model
        self.lstm_model = lstm_model

    def forward(self, inputs_, cells_state):

        ls_input = inputs_[:, 0, :, :, :]
        target= inputs_[:, 1, :, :, :]
        last_frame = inputs_[:, 2, :, :, :]

        lstm_out, cells_state = self.lstm_model(ls_input, cells_state)
        pixel_input = torch.cat([ls_input,last_frame],dim=1)
        out, _ = self.pixel_model(target, condition=pixel_input)
        return out, cells_state

