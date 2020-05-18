from torch import nn
from video.ConvLstmCell import ConvLstmCell


class LSTM(nn.Module):

    def __init__(self, layers_num, input_channel, hidden_channel, kernel_size ):
        super().__init__()
        self.type = type
        self.layers_num = layers_num
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.cells = []
        self.cells_state = []
        for i in range(layers_num):
            cell = ConvLstmCell(input_channel,hidden_channel,kernel_size)
            self.cells.append(cell)
            self.cells_state.append(None)

    def forward(self, input, cells_state = None):
        input_ = input
        if not cells_state is None:
            self.cells_state = cells_state

        for i, cell in enumerate(self.cells):
            self.cells_state[i] = cell(input_, self.cells_state[i])
            hidden, cell = self.cells_state[i]
            input_ = hidden

        return self.cells_state

