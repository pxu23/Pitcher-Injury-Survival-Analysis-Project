import numpy as np
import torch
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from sksurv.util import Surv

from SurvivalRNN.rnn_compute_survival_prob import compute_survival_function_single_season
from SurvivalRNN.rnn_prepare_input import prepare_time_varying_input_for_rnn
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
                          model_file, output_file_survival_probability):
    """
        Perform the evaluation of the RNN survival model with discrete survival loss on
        the evaluation season. Note that this is trained on the training_season
    """
    X_train, T_train, E_train = prepare_time_varying_input_for_rnn(training_season)

    # prepare the time-varying input for RNN evaluation for the evaluation season
    X_test, T_test, E_test = prepare_time_varying_input_for_rnn(evaluation_season)

    # load in the saved model and loss history
    model = torch.load(model_file)

    X_test = X_test.to(device)
    logits, _ = model(X_test)
    test_risk_predictions = logits[torch.arange(logits.shape[0]), T_test - 1]

    X_train = X_train.to(device)
    logits, _ = model(X_train)
    train_risk_predictions = logits[torch.arange(logits.shape[0]), T_train - 1]

    average_survival, all_survival = compute_survival_function_single_season(train_risk_predictions,
                                                                             test_risk_predictions,
                                                                             training_season,
                                                                             evaluation_season)
    print(f"All survival.shape is {all_survival.shape}")

    # Computes the Integrated Brier Score
    ibs_score = compute_integrated_brier_score(all_survival, E_train, T_train,
                                               E_test, T_test)
    #print(f"The achieved IBS score is {ibs_score}")

    # compute the Concordance Index
    c_index = round(concordance_index_censored(E_test.cpu().numpy().astype(bool), T_test.cpu(),
                                               test_risk_predictions.detach().cpu())[0], 3)

    # save the individual survival probabilities to a file
    np.savetxt(output_file_survival_probability,all_survival)

    return c_index, ibs_score, all_survival, average_survival


def compute_integrated_brier_score(all_survival, E_train, T_train,
                                   E_test, T_test):
    # ------------------------------
    # Integrated Brier Score using scikit-survival
    # ------------------------------

    # Convert ground truth to a structured array required by scikit-survival.
    y_train = np.array([(bool(e), t) for e, t in zip(E_train, T_train)],
                        dtype=[('event', bool), ('time', float)])
    y_test = np.array([(bool(e), t) for e, t in zip(E_test, T_test)],
                        dtype=[('event', bool), ('time', float)])

    num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())

    times = np.linspace(max(T_train.min(), T_test.min()),
                        min(T_train.max(), T_test.max()), num_time_points, endpoint=False)

    ibs_value = integrated_brier_score(y_train, y_test, all_survival, times)
    return ibs_value