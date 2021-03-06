import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as f


class ConvLstmCell(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel_size, device):
        super().__init__()
        self.device = device
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.Gates = nn.Conv2d(input_channel + hidden_channel, 4 * hidden_channel, kernel_size,
                               padding=kernel_size // 2)

    def forward(self, input, state=None):
        batch_size = input.data.size()[0]
        spatial_size = input.data.size()[2:]

        if state is None:
            state_size = [batch_size, self.hidden_channel] + list(spatial_size)
            state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = state
        prev_cell = prev_cell.to(self.device)
        prev_hidden = prev_hidden.to(self.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input.float(), prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        hidden = out_gate * f.tanh(cell)
        return hidden, cell
