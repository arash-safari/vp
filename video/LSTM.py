from torch import nn
from video.ConvLstmCell import ConvLstmCell


class LSTM(nn.Module):

    def __init__(self, layers_num, input_channel, hidden_channel, kernel_size,device ):
        super().__init__()
        self.type = type
        self.layers_num = layers_num
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.cells = nn.ModuleList()
        self.device = device
        for i in range(layers_num):
            cell = ConvLstmCell(input_channel,hidden_channel,kernel_size, device)
            self.cells.append(cell)

    def forward(self, input, cells_state = None):
        input_ = input
        if cells_state is None:
            cells_state = [None] * len(self.cells)
        for i, cell in enumerate(self.cells):
            cells_state[i] = cell(input_, cells_state[i])
            hidden, cell = cells_state[i]
            input_ = hidden

        return cells_state

