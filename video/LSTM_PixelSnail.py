from torch import nn
from video.ConvLstmCell import ConvLstmCell
import torch
from tqdm import tqdm


class LSTM_PixelSnail(nn.Module):

    def __init__(self, lstm_model, cnn_model, pixel_model):
        super().__init__()
        self.type = type
        self.pixel_model = pixel_model
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model

    def forward(self, inputs_, cells_state):
        input_ = inputs_[:, 0, :, :, :]
        target = inputs_[:, 1, :, :, :]
        lstm_out, cells_state = self.lstm_model(input_, cells_state)

        cnn_out = self.cnn_model(lstm_out)
        out, _ = self.pixel_model(target, condition=cnn_out)
        return out, cells_state

    @torch.no_grad()
    def sample_model(self, input_, cells_state, device, batch, size, temperature, condition=None):
        row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
        cache = {}
        lstm_out, cells_state = self.lstm_model(input_, cells_state)
        cnn_out = self.cnn_model(lstm_out)
        size = input_.size()

        for i in tqdm(range(size[0])):
            for j in range(size[1]):
                out, cache = self.pixel_model(row[:, : i + 1, :], condition=cnn_out, cache=cache)
                prob = torch.softmax(out[:, :, i, j] / temperature, 1)
                sample = torch.multinomial(prob, 1).squeeze(-1)
                row[:, i, j] = sample

        return row, cells_state
