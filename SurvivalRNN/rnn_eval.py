import numpy as np
import torch
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from sksurv.util import Surv

from SurvivalRNN.rnn_compute_survival_prob import compute_survival_function_single_season
from SurvivalRNN.rnn_prepare_input import prepare_time_varying_input_for_rnn, prepare_time_invariant_input_for_rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_integrated_brier_score_rnn_cox(rnn_model, training_season, evaluation_season):
    X_train, T_train, E_train = prepare_time_varying_input_for_rnn(training_season)
    X_test, T_test, E_test = prepare_time_varying_input_for_rnn(evaluation_season)

    # Create a structured array for survival data
    y_train = Surv.from_arrays(event=E_train, time=T_train)
    y_test = Surv.from_arrays(event=E_test, time=T_test)

    # predict the survival function from the RNN model
    #model = torch.load(f"rnn_model_{training_season}_{evaluation_season}.pth").to(device)

    # Forward pass: Get predicted risk scores from RNN model
    train_logits = rnn_model(X_train.to(device))
    train_predictions = train_logits[torch.arange(train_logits.shape[0]), T_train - 1]

    logits = rnn_model(X_test.to(device))
    predictions = logits[torch.arange(logits.shape[0]), T_test - 1]

    # Compute the overall and individual survival functions based on the predicted risk scores
    _, S_x_t = compute_survival_function_single_season(train_predictions.cpu(),predictions.cpu(),
                                                       training_season, evaluation_season)
    print(f"S_x_t.shape: {S_x_t.shape}")
    num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())

    # Sort data by event time
    sorted_indices = np.argsort(T_test)

    # should it be unique? 78 unique times
    # maybe go from 1 to max
    #print(f"even_time_sorted.max(): {event_times_sorted.max()}")
    times = np.linspace(max(T_train.min(), T_test.min()),
                        min(T_train.max(), T_test.max()), num_time_points, endpoint=False)


    score = integrated_brier_score(y_train, y_test, S_x_t, times)

    return score



def evaluate_rnn_cox(rnn_model, X_padded, T, E,
                 training_season, evaluation_season):
    # Forward pass: Get predicted hazards (risk scores)
    logits = rnn_model(X_padded)  # Get predictions
    predictions = logits[torch.arange(logits.shape[0]), T - 1].squeeze(-1)
    predictions = predictions.to(device)

    print(f"Number of predictions is {len(predictions.cpu().detach().numpy())}")

    # write the predicted hazard and the ground truth times to file
    #print(f"Model predictions are {predictions}")
    #with open("rnn_prediction_ground_truth.txt", "w") as f:
    #    f.write("Instance \t Predicted Risk \t Ground Truth Time \t Event\n")
    #    for i in range(len(predictions.cpu().detach().numpy())):
    #        f.write(f"{i+1} \t {predictions.cpu().detach().numpy()[i]} \t {T.cpu().detach().numpy()[i]}"
    #                f"\t {int(E.cpu().detach().numpy()[i])}\n")
    #f.close()
    print(f"predictions.shape: {predictions.shape}")

    # compute the concordance index
    risk_scores = predictions.cpu().detach().numpy()  # Get model's predicted risk scores
    c_index = round(concordance_index_censored(E.cpu().numpy().astype(bool), T.cpu(), risk_scores)[0], 3)

    print(f"The achieved concordance index for the evaluation is {c_index}")

    # integrated brier score
    ibs = compute_integrated_brier_score_rnn_cox(rnn_model, training_season, evaluation_season)
    print(f"The achieved Integrated Brier Score for the evaluation is {ibs}")

    return c_index, ibs

import numpy as np
import torch
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from sksurv.metrics import integrated_brier_score, concordance_index_censored

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_rnn_discrete(training_season, evaluation_season,
                          model_file, output_file_survival_probability,
                          time_invariant):
    """
        Perform the evaluation of the RNN survival model with discrete survival loss on
        the evaluation season. Note that this is trained on the training_season
        :param training_season: the season that the RNN model is fitted on
        :param evaluation_season: the season that the RNN model is evaluated on
        :param model_file: the file for the trained RNN model
        :param output_file_survival_probability: the output file survival probability file
        :return concordance_index, integrated brier score, overall survival curve, individual survival curve
    """
    # prepare the features, event times, and events for the training
    if time_invariant:
        X_train, T_train, E_train = prepare_time_invariant_input_for_rnn(training_season)
    else:
        X_train, T_train, E_train = prepare_time_varying_input_for_rnn(training_season)

    # prepare the time-varying input for RNN evaluation for the evaluation season
    if time_invariant:
        X_test, T_test, E_test = prepare_time_invariant_input_for_rnn(evaluation_season)
    else:
        X_test, T_test, E_test = prepare_time_varying_input_for_rnn(evaluation_season)

    # load in the saved RNN model
    model = torch.load(model_file)

    X_test = X_test.to(device)

    # compute the logits, hazards output of the saved RNN model on the test dataset
    logits, hazards = model(X_test)
    
    # survival curve computation from MATRX 2025 (cumulative product of 1 - hazards)
    #all_survival = torch.cumprod(1 - hazards, dim=1).detach().numpy()

    # predicted risk score of each instances in the test dataset
    test_risk_predictions = logits[torch.arange(logits.shape[0]), T_test - 1]

    X_train = X_train.to(device)

    # logits of the RNN model on the training features
    logits, _ = model(X_train)

    # predicted risk score of each instances in the training dataset
    train_risk_predictions = logits[torch.arange(logits.shape[0]), T_train - 1]

    # compute the overall survival and the individual survival probabilities
    average_survival, all_survival = compute_survival_function_single_season(train_risk_predictions,
                                                                             test_risk_predictions,
                                                                             training_season,
                                                                             evaluation_season)
    print(f"All survival.shape is {all_survival.shape}")

    #all_survival_ibs = all_survival[:,max(T_train.min(), T_test.min()):
    #min(T_train.max(), T_test.max())]

    # Computes the Integrated Brier Score
    ibs_score = compute_integrated_brier_score(all_survival, E_train, T_train,
                                               E_test, T_test)

    # compute the Concordance Index
    c_index = round(concordance_index_censored(E_test.cpu().numpy().astype(bool), T_test.cpu(),
                                               test_risk_predictions.detach().cpu())[0], 3)

    # save the individual survival probabilities to a file
    np.savetxt(output_file_survival_probability,all_survival)

    return c_index, ibs_score, all_survival, average_survival


def compute_integrated_brier_score(all_survival, E_train, T_train,
                                   E_test, T_test):
    """
        Computes the Integrated Brier Score
        :param all_survival: the survival probability for all individual instances
        :param E_train: the event indicators for the training dataset
        :param T_train: the event times for all instances in training dataset
        :param E_test: the event indicators for the test dataset
        :param T_test: the event times for all instances in test dataset
    """

    # Convert ground truth to a structured array required by scikit-survival.
    #y_train = np.array([(bool(e), t) for e, t in zip(E_train, T_train)],
    #                    dtype=[('event', bool), ('time', float)])
    #y_test = np.array([(bool(e), t) for e, t in zip(E_test, T_test)],
    #                    dtype=[('event', bool), ('time', float)])
    y_train = Surv.from_arrays(event=E_train, time=T_train)
    y_test = Surv.from_arrays(event=E_test, time=T_test)

    # number of time points for integrated brier score
    num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())

    # grid of time points for integrated brier score
    times = np.linspace(max(T_train.min(), T_test.min()),
                        min(T_train.max(), T_test.max()), num_time_points, endpoint=False)

    # compute the integrated brier score for the survival functions at the time grid
    ibs_value = integrated_brier_score(y_train, y_test, all_survival, times)
    return ibs_value