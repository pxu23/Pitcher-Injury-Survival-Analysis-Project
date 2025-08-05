# Fit the Random Survival Forest model on the colon dataset (using train test split of 70-30%
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv
import warnings
warnings.filterwarnings("ignore")

train_evaluate_seasons = [("2021", "2022"), ("2022", "2023"), ("2023", "2024")]
models = ['rsf', "gradient_boosting"]

for (training_season, evaluation_season) in train_evaluate_seasons:
    for model_name in models:

        # read in the time-invariant game-level survival dataframe for the season
        survival_df_time_invariant_game_level_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                   f"survival_df_time_invariant_game_{training_season}_level_processed.csv")
        survival_df_time_invariant_game_level_season["event"] = survival_df_time_invariant_game_level_season["event"].astype(bool)

        X_train = survival_df_time_invariant_game_level_season.drop(columns=["event", "num_games", "player_name", "bats", "throws"])
        y_train = survival_df_time_invariant_game_level_season[["event", "num_games"]]

        # fit the random survival forest model on X_train, y_train
        if model_name == "rsf":
            model = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=100)
        elif model_name == "gradient_boosting":
            model = GradientBoostingSurvivalAnalysis(n_estimators=100, min_samples_split=10, min_samples_leaf=15,random_state=100)
        else:
            raise ValueError("Invalid model name")

        model.fit(X_train.to_numpy(),y_train.to_records(index=False))

        survival_df_time_invariant_game_level_evaluation_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                              f"survival_df_time_invariant_game_{evaluation_season}_level_processed.csv")
        survival_df_time_invariant_game_level_evaluation_season['event'] = survival_df_time_invariant_game_level_evaluation_season["event"].astype(bool)
        X_test = survival_df_time_invariant_game_level_evaluation_season.drop(columns=["event", "num_games", "player_name", "bats", "throws"])
        y_test = survival_df_time_invariant_game_level_evaluation_season[["event", "num_games"]]

        # Evaluate the random survival forest model on X_test, y_test (C-index and IBS Score)
        concordance_index = model.score(X_test, y_test.to_records(index=False))
        print(f"The achieved concordance index on the {model_name} model fitted on the "
              f"{training_season} on the held-out {evaluation_season} season is {round(concordance_index, 2)}")

        T_train = y_train['num_games'].to_numpy()
        E_train = y_train['event'].to_numpy()
        y_train = Surv.from_arrays(event=E_train, time=T_train)

        T_test = y_test['num_games'].to_numpy()
        E_test = y_test['event'].to_numpy()
        y_test = Surv.from_arrays(event=E_test, time=T_test)

        num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())

        # Define a grid of times from the minimum to maximum observed in test data.
        # Use endpoint=False to ensure we exclude the maximum time (per requirements).
        times = np.linspace(max(T_train.min(), T_test.min()),
                            min(T_train.max(), T_test.max()), num_time_points, endpoint=False)
        #print(len(times))
        rsf_survival = model.predict_survival_function(X_test.to_numpy())

        pred_surv = np.asarray([fn(times) for fn in rsf_survival])

        overall_survival = np.mean(pred_surv, axis=0)

        np.savetxt(f"../Survival_Probabilities/{model_name}_survival_train_{training_season}_eval_{evaluation_season}.txt", pred_surv)
        #print(pred_surv.shape)
        ibs_score = integrated_brier_score(y_train, y_test, pred_surv, times)

        print(f"The IBS score of the {model_name} model fitted on the "
              f"{training_season} on the held-out {evaluation_season} season is {round(ibs_score, 2)}")
        print("\n")
