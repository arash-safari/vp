from torch import nn
from video.ConvLstmCell import ConvLstmCell
import torch
from tqdm import tqdm


class LSTM_PixelSnail(nn.Module):
    def _to_one_hot(self, y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to('cuda')

        return zeros.scatter(scatter_dim, y_tensor, 1).permute(0, 3, 1, 2)

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
    def sample(self, input_, cells_state, temperature=1.0):
        size = input_.size()
        row = torch.zeros( *size).to('cuda')
        cache = {}
        lstm_out, cells_state = self.lstm_model(input_, cells_state)
        cnn_out = self.cnn_model(lstm_out)

        for i in tqdm(range(size[0])):
            for j in range(size[1]):
                out, cache = self.pixel_model(row[: ,:, : i + 1, :], condition=cnn_out, cache=cache)
                print('out {}'.format(out.size()))
                prob = torch.softmax(out[:, :, i, j] / temperature, 1)
                # print(prob.size())
                sample = torch.multinomial(prob, 1).squeeze(-1)
                # sample = self._to_one_hot(sample, size[1]).float()
                # print(sample.size())

                # print(row.size())
                row[:,0, i, j] = sample


        return row, cells_state
