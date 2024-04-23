import torch

class Network(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Network, self).__init__()
        self.affine = torch.nn.Linear(23*5, hidden_size)
        self.act1 = torch.nn.Tanh()
        self.lstm = torch.nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers = 3, bidirectional= False, batch_first=True)
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, output_size),
        )
    def forward(self, x, lx):
        x = self.affine(x)
        x = self.act1(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lx, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        logits = self.classification(x)
        return logits
    

