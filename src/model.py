from torch.nn import Module
import torch.nn as nn


class CharsRnn(Module):
    def __init__(self, no_chars, hidden_layers=2,hidden_nodes=None, train_on_gpu=False):
        super().__init__()
        self.input_size = no_chars
        if hidden_nodes is None:
            hidden_nodes = no_chars * 2
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes

        self.lstm = nn.LSTM(no_chars, hidden_nodes, hidden_layers, dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_nodes, no_chars)
        self.train_on_gpu = train_on_gpu
        if train_on_gpu:
            self.cuda()


    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_nodes)

        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (weight.new(self.hidden_layers, batch_size, self.hidden_nodes).zero_().cuda(),
                      weight.new(self.hidden_layers, batch_size, self.hidden_nodes).zero_().cuda())
        else:
            hidden = (weight.new(self.hidden_layers, batch_size, self.hidden_nodes).zero_(),
                      weight.new(self.hidden_layers, batch_size, self.hidden_nodes).zero_())

        return hidden
