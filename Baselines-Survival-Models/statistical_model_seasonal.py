## for each of the season, fit various survival analysis models
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, CoxPHFitter
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv
import warnings
warnings.filterwarnings("ignore")

# use the following pairs of training and evaluation seasons for apper
train_evaluate_seasons = [("2021", "2022"), ("2022", "2023"), ("2023", "2024")]

# Cox Proportional Hazard and Weibull Accelerated Failure Time Models
models = ['Cox', "Weibull AFT"]

for (train_season, evaluate_season) in train_evaluate_seasons:
    # read in the time-invariant game-level survival dataframe for the training season
    survival_df_time_invariant_game_level_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                               f"survival_df_time_invariant_game_{train_season}_level_processed.csv")

    # drop the player name column since we are not interested (also drop the batting hand and throwing hand columns)
    training_data = survival_df_time_invariant_game_level_season.drop(columns=["player_name", "bats", "throws"])

    # read in the time-invariant game-level survival dataframe for the evaluation season
    survival_df_time_invariant_game_level_evaluation_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                          f"survival_df_time_invariant_game_{evaluate_season}_level_processed.csv")
    # drop the player name column since we are not interested (also drop the batting hand and throwing hand columns)
    testing_data = survival_df_time_invariant_game_level_evaluation_season.drop(columns=["player_name", "bats", "throws"])

    for model_name in models:
        if model_name == "Cox":
            # Cox Proportional Hazard model
            model = CoxPHFitter()
        elif model_name == "Weibull AFT":
            # Weibull Accelerated Failure Time model
            model = WeibullAFTFitter()
        else:
            raise ValueError("Invalid model name")

        # fit the model on the training data
        model.fit(training_data, event_col="event", duration_col="num_games")

        # get the achieved Concordance index on the testing data
        concordance_index = model.score(testing_data, scoring_method="concordance_index")
        print(f"The Concordance Index of the game-level {model_name} model with recurrence fitted on the {train_season} season on the"
              f" held out {evaluate_season} season is {round(concordance_index,2)}")

        T_train = training_data["num_games"].to_numpy()
        T_test = testing_data["num_games"].to_numpy()
        E_train = training_data["event"].to_numpy()
        E_test = testing_data["event"].to_numpy()

        # training and testing labels
        y_train = Surv.from_arrays(event=E_train,time=T_train)
        y_test = Surv.from_arrays(event=E_test,time=T_test)

        # training features
        X_train = training_data.drop(columns=["num_games", "event"])

        # number of time points
        num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())

        # grid of times
        times = np.linspace(max(T_train.min(), T_test.min()), min(T_train.max(), T_test.max()), num_time_points, endpoint=False)

        # testing features
        X_test = testing_data.drop(columns=["num_games", "event"])

        # predicted survival function for the model
        model_survival = model.predict_survival_function(X_test, times=times)

        # taking the transpose
        pred_survs = model_survival.T.values

        # save the predicted survival probabilites for each instance to txt file
        np.savetxt(f"../Survival_Probabilities/weibull_aft_survival_train_{train_season}_eval_{evaluate_season}.txt",
                   pred_survs)

        # compute the integrated brier score
        ibs_score = integrated_brier_score(y_train, y_test, pred_survs, times)
        print(f"The Integrated Brier Score of the game-level {model_name} model with recurrence fitted on the {train_season} season "
              f"on the held out {evaluate_season} season is {round(ibs_score,2)}")
        print("\n")