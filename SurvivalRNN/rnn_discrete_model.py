import torch
from torch import nn


class RNNDiscreteSurvival(nn.Module):
    # ------------------------------
    # RNN-Based Discrete-Time Survival Model
    # ------------------------------

    def __init__(self, input_dim, hidden_dim,
                 fc_hidden_dim1, fc_hidden_dim2,
                 fc_hidden_dim3,num_layers=1):

        super(RNNDiscreteSurvival, self).__init__()

        # hidden dimension for the RNN model
        self.hidden_dim = hidden_dim

        # the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim1)
        self.fc2 = nn.Linear(fc_hidden_dim1, fc_hidden_dim2)
        self.fc3 = nn.Linear(fc_hidden_dim2, fc_hidden_dim3)
        self.fc4 = nn.Linear(fc_hidden_dim3, 1)

        # the multilayer perceptron
        self.fc = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu, self.fc3,
                                self.relu, self.fc4)


    def forward(self, x):
        # x: (batch_size, seq_length, feature_dim)
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_dim)

        logits = self.fc(out.data).squeeze(-1)  # (batch_size, seq_length)

        # I think that's the issue here (probability distribution over all instances, which is wrong)
        hazard = torch.sigmoid(logits)

        return logits, hazard