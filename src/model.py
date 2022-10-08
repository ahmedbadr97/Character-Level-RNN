import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn

import torch.nn.functional as F


class CharsRnn(Module):
    def __init__(self, chars, hidden_layers=2, hidden_nodes=None, train_on_gpu=False):
        super().__init__()
        no_chars = len(chars)
        self.chars = chars
        self.chars_to_int = {}
        self.int_to_chars = {}
        for num, char in enumerate(chars):
            self.chars_to_int[char] = num
            self.int_to_chars[num] = char
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

    def forward(self, x, hidden) -> (torch.Tensor, torch.Tensor):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = out.contiguous().view(-1, self.hidden_nodes)

        out = self.fc(out)
        return out, hidden

    def one_hot_encode(self, char):
        vector = np.zeros((len(self.chars_to_int)))
        vector[self.chars_to_int[char]] = 1
        return vector

    def predict_text(self, context: str, no_chars: int, hidden=None):
        self.eval()
        predicted_txt=[]
        with torch.no_grad():
            if hidden is None:
                hidden = self.init_hidden(1)
            # feed the context
            for c in context:
                _, hidden = self._next_char(c, hidden)
                predicted_txt.append(c)
            # the one after the last char in given context
            predicted_txt.append(_)
            # predict next chars
            for i in range(no_chars):
                char, hidden = self._next_char(predicted_txt[-1], hidden)
                predicted_txt.append(char)

        return "".join(predicted_txt)

    def _next_char(self, current_char, hidden):
        char_encoding = self.one_hot_encode(current_char)
        # add batch_dim, seq_dim
        char_encoding.resize((1, 1, char_encoding.shape[0]))
        model_input = torch.tensor(char_encoding, dtype=torch.float32)
        if self.train_on_gpu:
            model_input=model_input.cuda()
        out, hidden = self.forward(model_input, hidden)
        hidden = tuple([each.data for each in hidden])

        out = F.softmax(out, dim=1).data
        # size (1,no_chars) batch size=1 seq_len=1
        # top_char = torch.argmax(out, dim=1).item()
        # top_char = self.int_to_chars[top_char]
        p, top_ch_idx = out.topk(5)
        top_ch_idx = top_ch_idx.numpy().squeeze()

        # select the likely next character with some element of randomness

        p = p.numpy().squeeze()
        char = np.random.choice(top_ch_idx, p=p / p.sum())

        return self.int_to_chars[char], hidden

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

    def load_weights(self, path, cuda_weights):
        if cuda_weights:
            device = torch.device('cpu')
            state_dict = torch.load(path, map_location=device)
        else:
            state_dict = torch.load(path)
        self.load_state_dict(state_dict)
