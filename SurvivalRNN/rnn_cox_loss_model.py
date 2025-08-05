import torch.nn as nn

class SurvivalRNNCox(nn.Module):
    """
        The RNN model for survival analysis with Cox Proportional Hazard Loss that
         follows the RNN model in RNN-Surv paper for FLCHAIN
    """
    def __init__(self, input_size, hidden_size, hidden_size1, hidden_size2,
                 hidden_size3, num_layers=2):
        """
            Initializes the survival RNN model
            :param input_size: the size of the input (i.e. number of input features)
            :param hidden_size: the size of the hidden layer for LSTM
            :param hidden_size1: the size of the first hidden layer for Fully Connected Network
            :param hidden_size2: the size of the second hidden layer for Fully Connected Network
            :param num_layers: the number of hidden layers
        """
        super(SurvivalRNNCox, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # first linear layer
        self.fc1 = nn.Linear(hidden_size, hidden_size1)

        # second linear layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # third linear layer
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)

        # fourth linear layer
        self.fc4 = nn.Linear(hidden_size3, 1)

    def forward(self, x):
        """
            Performs forward propagation through the network
            :param x: the input
            :return: the output of the network (the logits)
        """
        # LSTM output
        out, _ = self.lstm(x)

        # Pass the LSTM output through the first fully Connected Layer
        x = self.fc1(out)

        # ReLU activation function in between
        x = nn.ReLU()(x)

        # Pass through the second fully connected layer
        x = self.fc2(x)

        # ReLU activation function
        x = nn.ReLU()(x)

        # Pass through the third fully connected layer
        x = self.fc3(x)

        # ReLU activation function
        x = nn.ReLU()(x)

        # get the logits after passing through the fourth fully connected layer
        logits = self.fc4(x)

        # risk score (should it not be only the final timestep but the event timestep)
        #risk_score = logits[:, -1]

        return logits