
import torch
import torch.nn as nn

neurons = 2176

class NMPCPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(NMPCPredictor, self).__init__()
        self.hidden1 = nn.Linear(input_size, neurons)
        self.act1 = nn.ELU()
        self.hidden2 = nn.Linear(neurons, neurons)
        self.act2 = nn.ELU()
        self.hidden3 = nn.Linear(neurons, neurons)
        self.act3 = nn.ELU()
        self.output = nn.Linear(neurons, output_size)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x

