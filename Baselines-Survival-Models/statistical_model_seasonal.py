## for each of the season, fit various survival analysis models
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, CoxPHFitter
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv
import warnings
warnings.filterwarnings("ignore")

train_evaluate_seasons = [("2021", "2022"), ("2022", "2023"), ("2023", "2024")]
models = ['Cox', "Weibull AFT"]

for (train_season, evaluate_season) in train_evaluate_seasons:
    # read in the time-invariant game-level survival dataframe for the season
    survival_df_time_invariant_game_level_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                               f"survival_df_time_invariant_game_{train_season}_level_processed.csv")

    # drop the player name column since we are not interested
    training_data = survival_df_time_invariant_game_level_season.drop(columns=["player_name", "bats", "throws"])

    for model_name in models:

        # Weibull Accelerated Failure Time model
        if model_name == "Cox":
            model = CoxPHFitter()
        elif model_name == "Weibull AFT":
            model = WeibullAFTFitter()
        else:
            raise ValueError("Invalid model name")

        model.fit(training_data, event_col="event", duration_col="num_games")

        survival_df_time_invariant_game_level_evaluation_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                              f"survival_df_time_invariant_game_{evaluate_season}_level_processed.csv")
        testing_data = survival_df_time_invariant_game_level_evaluation_season.drop(columns=["player_name", "bats", "throws"])
        concordance_index = model.score(testing_data, scoring_method="concordance_index")
        print(f"The Concordance Index of the game-level {model_name} model with recurrence fitted on the {train_season} season on the"
              f" held out {evaluate_season} season is {round(concordance_index,2)}")

        # Integrated Brier Score
        T_train = training_data["num_games"].to_numpy()
        T_test = testing_data["num_games"].to_numpy()
        E_train = training_data["event"].to_numpy()
        E_test = testing_data["event"].to_numpy()

        y_train = Surv.from_arrays(event=E_train,time=T_train)
        y_test = Surv.from_arrays(event=E_test,time=T_test)

        X_train = training_data.drop(columns=["num_games", "event"])

        num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())

        times = np.linspace(max(T_train.min(), T_test.min()), min(T_train.max(), T_test.max()), num_time_points, endpoint=False)
        X_test = testing_data.drop(columns=["num_games", "event"])
        model_survival = model.predict_survival_function(X_test, times=times)

        pred_survs = model_survival.T.values

        np.savetxt(f"../Survival_Probabilities/weibull_aft_survival_train_{train_season}_eval_{evaluate_season}.txt",
                   pred_survs)
        ibs_score = integrated_brier_score(y_train, y_test, pred_survs, times)
        print(f"The Integrated Brier Score of the game-level {model_name} model with recurrence fitted on the {train_season} season "
              f"on the held out {evaluate_season} season is {round(ibs_score,2)}")
        print("\n")
