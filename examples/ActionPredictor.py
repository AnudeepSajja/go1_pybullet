import torch
import torch.nn as nn

# Define a Neural Network
class ActionPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActionPredictor, self).__init__()
        self.hidden1 = nn.Linear(input_size, 256)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(256, 512)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(512, 1024)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(1024, 1024)
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(1024, 1024)
        self.act5 = nn.ReLU()
        self.hidden6 = nn.Linear(1024, 512)
        self.act6 = nn.ReLU()
        self.hidden7 = nn.Linear(512, 256)
        self.act7 = nn.ReLU()
        self.output = nn.Linear(256, output_size)
        # self.act_output = nn.Sigmoid()  # Uncomment if needed for specific output activation

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.act6(self.hidden6(x))
        x = self.act7(self.hidden7(x))  # Corrected line

        x = self.output(x)  # No activation function needed for MSE loss
        return x
