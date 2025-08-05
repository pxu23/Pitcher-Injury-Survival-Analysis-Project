import os
import time

from rnn_cox_loss_model import SurvivalRNNCox
from rnn_prepare_input import prepare_time_varying_input_for_rnn
from rnn_train import train_rnn_cox
from rnn_eval import evaluate_rnn_cox
from rnn_compute_survival_prob import compute_survival_function_single_season

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# sets seed to ensure reproducibility (maybe look across multiple seeds)
torch.manual_seed(100)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ("2021", "2022"),("2022", "2023"),

for training_season, evaluation_season in [("2023", "2024")]:

    print(f"\nTrain on {training_season} season, "
          f"Evaluate on {evaluation_season} season")

    # Initialize model
    input_size = 11  # Number of features (game stats) (including 10th and 90th percentile)
    hidden_size = 32

    hidden_size1 = 45
    hidden_size2 = 40
    hidden_size3 = 35

    # create the survival RNN model with Cox Proportional Hazard loss
    model = SurvivalRNNCox(input_size, hidden_size, hidden_size1, hidden_size2, hidden_size3)

    model = model.to(device)

    # STEP 2: Create the time-varying RNN input for the training season
    X_train, T_train, E_train = prepare_time_varying_input_for_rnn(training_season)
    X_train = X_train.to(device)
    T_train = T_train.to(device)
    E_train = E_train.to(device)

    # if model does not exist, train it otherwise directly read it
    if not os.path.exists(f"RNN_Models/rnn_model_cox_loss_{training_season}_{evaluation_season}.pt"):
        # calculate the training time
        start_time = time.time()

        num_epoches = 100

        # STEP 3:  train the RNN model on the corresponding season
        loss_history = train_rnn_cox(model,X_train, T_train, E_train, num_epoches, lr=0.01)

        end_time = time.time()

        print(f"Training time: {round(end_time - start_time,3)} seconds")

        # plot of training loss over the epoches
        plt.figure()
        # epoches
        epoches = np.array([i for i in range(1, num_epoches + 1)])

        # STEP 4: Plot the loss history
        plt.plot(epoches, loss_history)
        plt.xlabel("Epoches")
        plt.ylabel('Training loss')
        plt.savefig(f'RNN_Loss_History_Curves/loss_history_{training_season}_rnn_cox_ph_loss.png')

        # STEP 5: Save the trained RNN model
        torch.save(model, f"RNN_Models/rnn_model_cox_loss_{training_season}_{evaluation_season}.pt")

    # STEP 6: Load the trained RNN model
    model = torch.load(f"RNN_Models/rnn_model_cox_loss_{training_season}_{evaluation_season}.pt")

    # STEP 7: Get the predictions for the training season
    logits_train = model(X_train).squeeze(-1)
    predictions_train = logits_train[torch.arange(logits_train.shape[0]), T_train - 1]

    # STEP 8: Get the time-varying input for the evaluation season for RNN
    X_test, T_test, E_test = prepare_time_varying_input_for_rnn(evaluation_season)
    X_test = X_test.to(device)
    T_test = T_test.to(device)
    E_test = E_test.to(device)

    # STEP 9: evaluate the RNN model on the subsequent season in terms of the predicted risk scores
    concordance_index, ibs_score = evaluate_rnn_cox(model, X_test, T_test, E_test,
                                                    training_season, evaluation_season)

    print(f"Concordance Index: {concordance_index}, Training: {training_season}, Evaluation: {evaluation_season}")
    print(f"IBS Score: {ibs_score}, Training: {training_season}, Evaluation: {evaluation_season}")

    # STEP 10: Get the predicted risk scores for the test instances
    logits = model(X_test)  # Get predictions
    test_predictions = logits[torch.arange(logits.shape[0]), T_test - 1].squeeze(-1)
    test_predictions = test_predictions.to(device)
    print(test_predictions.shape)

    # STEP 11: Compute the individual survival curves and the overall survival curves
    S_x_t_overall, S_x_t = compute_survival_function_single_season(predictions_train,test_predictions, training_season,
                                                                   evaluation_season)
    print(f"S_x_t_overall: {S_x_t_overall}")
    print(f"S_x_t: {S_x_t}")

    # STEP 12: Save the predicted survival probabilities for each individual to the .txt file
    np.savetxt(f"../Survival_Probabilities/survival_prob_rnn_cox_loss_{training_season}_{evaluation_season}.txt",
               S_x_t)