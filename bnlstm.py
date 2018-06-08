import torch
from torch import nn

class BNLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
            super(BNLSTMCell, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.weight_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
            self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
            self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

            self.bn_ih = nn.BatchNorm1d(4 * self.hidden_size, affine=False)
            self.bn_hh = nn.BatchNorm1d(4 * self.hidden_size, affine=False)
            self.bn_c = nn.BatchNorm1d(self.hidden_size)

            self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_hh.data[:, :self.hidden_size])
        nn.init.orthogonal_(self.weight_hh.data[:, self.hidden_size:2 * self.hidden_size])
        nn.init.orthogonal_(self.weight_hh.data[:, 2 * self.hidden_size:3 * self.hidden_size])
        nn.init.eye_(self.weight_hh.data[:, 3 * self.hidden_size:])
        self.weight_hh.data[:, 3 * self.hidden_size:] *= 0.95
        

    def forward(self, input, hx):
        h, c = hx
        ih = torch.matmul(input, self.weight_ih)
        hh = torch.matmul(h, self.weight_hh)
        bn_ih = self.bn_ih(ih)
        bn_hh = self.bn_hh(hh)
        hidden = bn_ih + bn_hh + self.bias

        i, f, o, g = torch.split(hidden, split_size_or_sections=self.hidden_size, dim=1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        new_h = torch.sigmoid(o) * torch.tanh(self.bn_c(new_c))
        return (new_h, new_c)


class BNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False):
        super(BNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.lstm_f = BNLSTMCell(input_size, hidden_size)
        if bidirectional:
            self.lstm_b = BNLSTMCell(input_size, hidden_size)
        self.h0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
        nn.init.normal_(self.h0, mean=0, std=0.1)
        nn.init.normal_(self.c0, mean=0, std=0.1)
    
    def forward(self, input, hx=None):
        if not self.batch_first:
            input = input.transpose(0, 1)
        batch_size, seq_len, dim = input.size()
        if hx: init_state = hx
        else: init_state = (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1))
        
        hiddens_f = []
        final_hx_f = None
        hx = (init_state[0][0], init_state[1][0])
        for i in range(seq_len):
            hx = self.lstm_f(input[:, i, :], hx)
            hiddens_f.append(hx[0])
            final_hx_f = hx
        hiddens_f = torch.stack(hiddens_f, 1)
        
        if self.bidirectional:
            hiddens_b = []
            final_hx_b = None
            hx = (init_state[0][1], init_state[1][1])
            for i in range(seq_len-1, -1, -1):
                hx = self.lstm_b(input[:, i, :], hx)
                hiddens_b.append(hx[0])
                final_hx_b = hx
            hiddens_b.reverse()
            hiddens_b = torch.stack(hiddens_b, 1)
        
        if self.bidirectional:
            hiddens = torch.cat([hiddens_f, hiddens_b], -1)
            hx = (torch.stack([final_hx_f[0], final_hx_b[0]], 0), torch.stack([final_hx_f[1], final_hx_b[1]], 0))
        else:
            hiddens = hiddens_f
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(1))
        if not self.batch_first:
            hiddens = hiddens.transpose(0, 1)
        return hiddens, hx
        
        
if __name__ == '__main__':
    lstm = BNLSTM(input_size=10, hidden_size=7, batch_first=False, bidirectional=False)#.cuda(0)
    input = torch.randn(3, 11, 10)#.cuda(0)
    o, h = lstm(input)
    o = torch.sum(o)
    o.backward()