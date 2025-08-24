import numpy as np
import torch

from SurvivalRNN.rnn_prepare_input import prepare_time_varying_input_for_rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_survival_function_single_season(train_predictions, test_predictions,
                                            X_train, T_train, E_train,
                                            X_test, T_test, E_test):
    X_train, T_train, E_train = prepare_time_varying_input_for_rnn(training_season)
    X_test, T_test, E_test = prepare_time_varying_input_for_rnn(evaluation_season)

    # train risk scores
    train_risk_scores = train_predictions.cpu().detach().numpy()

    # test risk scores
    test_risk_scores = test_predictions.cpu().detach().numpy()

    T_train = T_train.cpu().numpy()
    E_train = E_train.cpu().numpy()

    T_test = T_test.cpu().numpy()

    # Sort training data by event time
    sorted_indices = np.argsort(T_train)
    event_times_sorted = T_train[sorted_indices]

    # events corresponding to training instances sorted by event time
    event_sorted = E_train[sorted_indices]

    # exponential of risk scores for training instances sorted by event time
    train_risk_exp_sorted = np.exp(train_risk_scores[sorted_indices])  # Exponentiate risk scores

    # Compute Breslow's baseline cumulative hazard
    h_0 = np.zeros_like(event_times_sorted, dtype=np.float64)
    risk_sum = np.zeros_like(event_times_sorted, dtype=np.float64)

    for i, t in enumerate(event_times_sorted):
        # all the at risk individuals at current time
        at_risk = train_risk_exp_sorted[i:]

        # total exponential risk of the at risk individuals
        risk_sum[i] = at_risk.sum()

        # if the instance expressed event, then the hazard is 1 / risk sum
        # otherwise, the hazard is zero
        if event_sorted[i] == 1:  # Only count actual observed injures
            h_0[i] = 1 / risk_sum[i]

    # Compute cumulative hazard as the cumulative sum of the individual hazards
    H_0 = np.cumsum(h_0)

    # Compute baseline survival function S_0 from the cumulative hazard
    S_0 = np.exp(-H_0)

    def make_step_function(times, values):
        times = np.asarray(times)
        values = np.asarray(values)

        def step_func(t_query):
            t_query = np.atleast_1d(t_query)
            idx = np.searchsorted(times, t_query, side="right") - 1
            idx = np.clip(idx, 0, len(values) - 1)
            return values[idx]

        return step_func

    # Compute individual survival functions
    S_x_t = np.array([S_0**np.exp(r_score) for r_score in test_risk_scores])

    # Make the step functions for the individual survival probabilities
    step_functions = []
    for i in range(S_x_t.shape[0]):
        sf = make_step_function(event_times_sorted, S_x_t[i, :])
        step_functions.append(sf)

    # Evaluate the survival function at the times for the instances
    num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())
    times = np.linspace(max(T_train.min(), T_test.min()), min(T_train.max(), T_test.max()),
                        num_time_points, endpoint=False)
    S_x_t = np.asarray([fn(times) for fn in step_functions])

    # the overall average survival function for the training instances
    S_x_t_overall = np.mean(S_x_t, axis=0)

    return S_x_t_overall, S_x_t