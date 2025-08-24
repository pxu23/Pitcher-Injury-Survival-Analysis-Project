import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv
import warnings
warnings.filterwarnings("ignore")

# use the following pairs of training and evaluation seasons for apper
train_evaluate_seasons = [("2021", "2022"), ("2022", "2023"), ("2023", "2024")]

# use the Random Survival Forest and Gradient Boosting models
models = ['rsf', "gradient_boosting"]

for (training_season, evaluation_season) in train_evaluate_seasons:
    for model_name in models:
        # read in the time-invariant game-level survival dataframe for the training season
        survival_df_time_invariant_game_level_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                   f"survival_df_time_invariant_game_{training_season}_level_processed.csv")

        # convert the event column ot boolean
        survival_df_time_invariant_game_level_season["event"] = survival_df_time_invariant_game_level_season["event"].astype(bool)

        # training features (drops the batting and throwing hand too)
        X_train = survival_df_time_invariant_game_level_season.drop(columns=["event", "num_games", "player_name", "bats", "throws"])
        # training labels
        y_train = survival_df_time_invariant_game_level_season[["event", "num_games"]]

        if model_name == "rsf":
            # fit the random survival forest model on X_train, y_train
            model = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=100)
        elif model_name == "gradient_boosting":
            # fit the gradient boosting model on X-train, y_train
            model = GradientBoostingSurvivalAnalysis(n_estimators=100, min_samples_split=10, min_samples_leaf=15,random_state=100)
        else:
            raise ValueError("Invalid model name")

        # Fit the model
        model.fit(X_train.to_numpy(),y_train.to_records(index=False))

        # read in the time invariant game level survival dataframe for evaluation season
        survival_df_time_invariant_game_level_evaluation_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                              f"survival_df_time_invariant_game_{evaluation_season}_level_processed.csv")
        # convert the event feature to boolean
        survival_df_time_invariant_game_level_evaluation_season['event'] = survival_df_time_invariant_game_level_evaluation_season["event"].astype(bool)
        # get the test features and labels
        X_test = survival_df_time_invariant_game_level_evaluation_season.drop(columns=["event", "num_games",
                                                                                       "player_name", "bats", "throws"])
        y_test = survival_df_time_invariant_game_level_evaluation_season[["event", "num_games"]]

        # Evaluate the random survival forest model on X_test, y_test (C-index and IBS Score)
        concordance_index = model.score(X_test, y_test.to_records(index=False))
        print(f"The achieved concordance index on the {model_name} model fitted on the "
              f"{training_season} on the held-out {evaluation_season} season is {round(concordance_index, 2)}")

        concordance_index_train = model.score(X_train, y_train.to_records(index=False))
        print(f"The achieved concordance index of the {model_name} model fitted on the {training_season} on {training_season} season is {round(concordance_index_train, 2)}")

        T_train = y_train['num_games'].to_numpy()
        E_train = y_train['event'].to_numpy()
        y_train = Surv.from_arrays(event=E_train, time=T_train)

        T_test = y_test['num_games'].to_numpy()
        E_test = y_test['event'].to_numpy()
        y_test = Surv.from_arrays(event=E_test, time=T_test)

        # get the number of time points for IBS Score computation
        num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())

        # Grid of time (exclude the upper bound
        times = np.linspace(max(T_train.min(), T_test.min()),
                            min(T_train.max(), T_test.max()), num_time_points, endpoint=False)

        # get the predicted survival function for the test instances
        model_survival = model.predict_survival_function(X_test.to_numpy())

        # get the predicted survival probabilities for each instance at the time points
        pred_surv = np.asarray([fn(times) for fn in model_survival])

        # get the overall survival function for the model
        overall_survival = np.mean(pred_surv, axis=0)

        # saves the individual survival function to the .txt file
        np.savetxt(f"../Survival_Probabilities/"
                   f"{model_name}_survival_train_{training_season}_eval_{evaluation_season}.txt", pred_surv)

        # compute the Integrated Brier Score
        ibs_score = integrated_brier_score(y_train, y_test, pred_surv, times)

        print(f"The IBS score of the {model_name} model fitted on the "
              f"{training_season} on the held-out {evaluation_season} season is {round(ibs_score, 2)}")
        print("\n")