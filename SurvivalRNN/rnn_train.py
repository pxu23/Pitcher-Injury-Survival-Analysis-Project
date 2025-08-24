import os

import numpy as np
import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from SurvivalRNN.rnn_prepare_input import prepare_time_varying_input_for_rnn, prepare_time_invariant_input_for_rnn
from SurvivalRNN.pitcher_injury_dataset import PitcherInjuryDataset
from SurvivalRNN.rnn_discrete_model import RNNDiscreteSurvival

device = "cuda" if torch.cuda.is_available() else "cpu"

def cox_loss(risk_scores, T, E, eps=1e-8):
    """
    Compute Cox Proportional Hazard loss with numerical stability.
    - risk_scores: Predicted risk scores (higher = more risk)
    - T: Survival times
    - E: Event indicator (1 if event occurred, 0 if censored)
    """
    risk_scores = risk_scores.squeeze()  # Ensure correct shape
    T = T.squeeze()
    E = E.squeeze()

    # Sort data by descending survival time
    risk_order = torch.argsort(T, descending=True)
    risk_scores = risk_scores[risk_order]
    E = E[risk_order]

    # Normalize risk scores to prevent overflow
    #risk_scores = risk_scores - risk_scores.mean()

    # Compute log-cumulative sum of exponential risk scores
    log_risk_sum = torch.logcumsumexp(risk_scores, dim=0)

    # Compute loss (ensure no division by zero)
    loss = -torch.sum((risk_scores - log_risk_sum) * E) / (torch.sum(E) + eps)

    return loss

def train_rnn_cox(model, X_padded, T, E,
                  num_epoches, lr):
    """
        Function to train the RNN model with Cox PH Loss
        :param model: the RNN model to train
        :param X_padded: the padded features for the RNN
        :param T: the times for the RNN model
        :param E: the event indicators for the RNN model
        :param loss_fct: the loss function to use (likelihood or cox PH for now)
    """
    # Optimizer (Adam optimizer)
    # learning rate too high
    # maybe decrease according to schedule
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # add scheduler
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    # Training loop
    # loss history
    loss_history = []

    print(f"Start training")
    for epoch in range(num_epoches):
        model.train()  # Set model to training mode

        # zeros the gradient
        optimizer.zero_grad()

        # Forward pass: Get predicted risk scores
        logits = model(X_padded)  # Get predictions

        predictions = logits[torch.arange(logits.shape[0]), T - 1]

        # Compute Cox PH loss (based on negative Partial Likelihood Function)
        loss = cox_loss(predictions, T, E)

        # add to the loss history
        loss_history.append(loss.item())

        # Backward pass: Compute gradients
        loss.backward()

        # Optimizer step: Update weights
        optimizer.step()

        # scheduler step
        scheduler.step()

        # Print loss for each epoch
        print(f"Epoch [{epoch + 1}/{num_epoches}], Loss: {loss.item()}")

    return loss_history



# =============================================================================
# Collate Function for Variable Length Sequences (Padding)
# =============================================================================
def collate_fn(batch):
    # Each item in batch is a tuple (sequence, label)
    features, target, mask = zip(*batch)
    # Sort batch by sequence length in descending order
    #sorted_batch = sorted(zip(features, target, mask), key=lambda features: features[0].size(0), reverse=True)
    #features, target, mask= zip(*sorted_batch)
    #lengths = [seq.size(0) for seq in features]
    # Pad sequences to the length of the longest sequence
    features = [torch.tensor(seq) for seq in features]
    target = [torch.tensor(tgt) for tgt in target]
    mask = [torch.tensor(msk) for msk in mask]
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(target, batch_first=True, padding_value=0)
    padded_mask = pad_sequence(mask, batch_first=True, padding_value=0)

    # Add an extra dimension for input_size (here, input_size = 1)
    #padded_features = padded_features.unsqueeze(-1)
    return padded_features, padded_targets, padded_mask


def train_rnn_discrete_model(training_season, num_epochs, lr, input_dim, output_file_model,
                             output_file_loss_history, w1, w2, time_invariant):
    """
        Train the Survival RNN model with the discrete survival loss
        :param training_season: the season the survival rnn model is trained on
        :param num_epochs: the number of epochs to train for
        :param lr: the learning rate
        :param output_file_model: the output file to save the model to
        :param output_file_loss_history: the output file to save the loss history to
        :param w1: the weight w1 for the Discrete survival loss
        :param w2: the weight w2 for the Discrete Survival Loss
        :param time_invariant: whether we are using the time-invariant or time-varying RNN model
    """

    # prepare the time-varying input for RNN training for the training season
    if time_invariant:
        X_train, T_train, E_train = prepare_time_invariant_input_for_rnn(training_season)
    else:
        X_train, T_train, E_train = prepare_time_varying_input_for_rnn(training_season)

    # create the dataset for training season
    train_dataset = PitcherInjuryDataset(X_train, T_train, E_train)

    # create the dataloaders for training season
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True ,collate_fn=collate_fn)

    # ------------------------------
    # Training Loop
    # ------------------------------
    # Increase model capacity and train longer for near-perfect fitting.
    if not os.path.exists(output_file_model):
        # initialize the survival RNN model with Discrete Survival Loss
        model = RNNDiscreteSurvival(input_dim, hidden_dim=32, num_layers=2,
                                    fc_hidden_dim1=45, fc_hidden_dim2=40,
                                    fc_hidden_dim3=35).to(device)

        # set the Adam optimizer and the Exponential learning rate scheduler with gamma = 0.99
        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = ExponentialLR(optimizer, gamma=0.99)

        loss_history = []
        eps = 1e-7

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            # loop through each batch in the training dataloader
            for (features, target, mask) in train_loader:
                features = features.to(device)
                target = target.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()

                # pack padded sequence
                #packed_x= pack_padded_sequence(features, seq_length, batch_first=True)
                #packed_target = pack_padded_sequence(target, seq_length, batch_first=True)
                #packed_mask = pack_padded_sequence(mask, seq_length, batch_first=True)

                # here it's wrong with softmax
                risk_score, hazard = model.forward(features)  # (batch_size, seq_length)
                #print(f"hazard_packed.shape is {hazard_packed.shape}")
                # pad the packed hazard output
                #hazard = pad_packed_sequence(hazard_packed, batch_first=True)
                # does the sum across all rows equal one? If so, this can cause the loss issue and the indistinguisable issue

                batch_size_actual = features.size()[0]
                batch_loss = 0.0

                # Compute loss per sample batch
                for i in range(batch_size_actual):
                    sample_hazard = hazard[i]  # (seq_length,)
                    sample_target = target[i]  # (seq_length,)
                    valid_intervals = int(mask[i].sum().item())  # Number of valid time steps

                    if sample_target.sum() > 0:  # Event observed
                        # Find the event time (first occurrence of 1)
                        event_time = int(torch.argmax(sample_target).item())

                        # Compute cumulative log survival before event by the produce of the
                        survival_log = torch.sum(
                            torch.log(1 - sample_hazard[:event_time + 1] + eps)) if event_time > 0 else 0.0

                        # Log likelihood at the event time
                        event_log = torch.log(sample_hazard[event_time] + eps)
                        sample_loss = - (w2* event_log + w1* survival_log)
                    else:
                        # For censored instance, sum survival logs over valid intervals (no event here)
                        sample_loss = - w1 * torch.sum(torch.log(1 - sample_hazard[:valid_intervals] + eps))

                    batch_loss += sample_loss

                # why do we need to divide by batch_size_actual
                batch_loss = batch_loss / batch_size_actual

                # compute the back propagation
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            # update the scheduler every epoch
            scheduler.step()

            loss_history.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # save the trained RNN model and the loss history
        torch.save(model, output_file_model)
        np.savetxt(output_file_loss_history, loss_history)