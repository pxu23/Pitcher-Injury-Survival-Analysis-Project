
import torch

from SurvivalRNN.rnn_train import train_rnn_discrete_model
from SurvivalRNN.rnn_eval import evaluate_rnn_discrete

device = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings("ignore")

for training_season, evaluation_season in [("2021", "2022"), ("2022", "2023"), ("2023", "2024")]:
    w1 = 1
    w2 = 4

    output_file_model = f"RNN_Models/survivalrnn_{training_season}_{evaluation_season}_w1_{w1}_w2_{w2}.pt"
    output_file_loss_history = f"RNN_Loss_History_Curves/survivalrnn_{training_season}_{evaluation_season}_loss_history_w1_{w1}_w2_{w2}.txt"
    output_file_survival_probability = (f"../Survival_Probabilities/survivalrnn_all_individual_survival"
                                        f"_{training_season}_{evaluation_season}_w1_{w1}_w2_{w2}.txt")
    #output_file_hazard = f"R/survivalrnn_all_individual_hazard_{training_season}_{evaluation_season}_w1_{w1}_w2_{w2}.txt"

    input_dim =11

    # Train for 100 epochs at learning rate of 0.01
    num_epochs = 100
    lr = 0.01

    # perform training of the RNN Discrete Survival model
    train_rnn_discrete_model(training_season,num_epochs, lr, input_dim,
                             output_file_model, output_file_loss_history,
                             w1=w1, w2=w2, time_invariant=False)

    # evaluate the RNN model and get the integrated brier score, as well as the survival and hazard functions
    c_index, ibs_score, all_survival, overall_survival = evaluate_rnn_discrete(training_season, evaluation_season,
                                                                               output_file_model, output_file_survival_probability,
                                                                               time_invariant=False)

    print(f"\nThe integrated brier score for the RNN Model (Training: {training_season}, Evaluation: {evaluation_season}) "
          f"is: {ibs_score:.4f}")
    print(f"The achieved Concordance Index for the RNN Model (Training: {training_season},"
          f"Evaluation: {evaluation_season}) is {c_index:.4f}")

