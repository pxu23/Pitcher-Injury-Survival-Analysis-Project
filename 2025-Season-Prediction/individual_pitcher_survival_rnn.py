# use the SurvivalRNN model to predict the survival curves for individual pitchers for the 2025 season
## Survival curve predictions for individual pitchers
## the list of pitchers

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, WeibullAFTFitter
from matplotlib import pyplot as plt
import warnings

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from compute_survival_curve_rnn_pitcher import compute_survival_function_pitcher

device = "cuda" if torch.cuda.is_available() else "cpu"

from prepare_rnn_input_for_pitcher import prepare_time_varying_pitcher_input_for_rnn
from SurvivalRNN.rnn_prepare_input import prepare_time_varying_input_for_rnn

warnings.filterwarnings("ignore")

train_season = "2023"

# read in the time invariant survival data
survival_df_time_invariant_train_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                      f"survival_df_time_invariant_game_{train_season}_level_processed.csv")

survival_df_time_invariant_train_season.drop(columns=['player_name', 'bats', 'throws'], inplace=True)

pitcher_list = [("Blake", "Snell"), ("Cade", "Povich"), ("Chris", "Sale"), ("Cole", "Ragans"),                ("Corbin", "Burnes"), ("Framber", "Valdez"), ("Gerrit", "Cole"), ("Jack", "Flaherty"),
                ("Jacob", "deGrom"), ("Logan", "Gilbert"), ("Logan", "Webb"),
                ("Max", "Fried"), ("Spencer", "Strider"), ("Tarik", "Skubal"),
                ("Yoshinobu", "Yamamoto"), ("Zack", "Wheeler"), ("Paul", "Skenes"), ("Kevin", "Gausman"),
                ("Shane", "McClanahan")]

pitcher_list = [("Pablo", "Lopez")]

for first_name, last_name in pitcher_list:
        player_name = first_name.lower() + "_" + last_name.lower()

        plt.figure(figsize=(8,6))

        # STEP 1: REad in the game-level characteristics data for 2025
        avg_pitch_characteristics_game_level_2025 = pd.read_csv(
            "../Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2025.csv")

        # STEP 2: Get the average number of pitches in each game for 2025
        avg_pitches_per_game = avg_pitch_characteristics_game_level_2025['num_pitches'].mean()
        print(avg_pitches_per_game)

        # STEP 1: prepare the time varying input for the RNN model for the pitcher
        X_pitcher, T_pitcher, E_pitcher = prepare_time_varying_pitcher_input_for_rnn(player_name)


        # STEP 2: Get the trained RNN model
        model_file = f"../SurvivalRNN/RNN_Models/survivalrnn_2023_2024_w1_1_w2_4.pt"
        rnn_model = torch.load(model_file)

        # STEP 3: Get the predicted risk scores for the RNN model for that pitcher for each recurrence
        logits, hazard = rnn_model(X_pitcher.to(device))

        #pitcher_survival = torch.cumprod(1 - hazard, dim=1).detach().numpy()

        #pitcher_survival_overall = pitcher_survival.mean(axis=0)

        pitcher_risk_predictions = logits[torch.arange(logits.shape[0]), T_pitcher - 1]

        #final_hazard = hazard[0, T_pitcher - 1].item()

        # STEP 4: Get the predicted training risks (should actually save to a file and load for efficiency)
        # STEP 3: get the predicted risks for the RNN model for the train season
        X_train, T_train, E_train = prepare_time_varying_input_for_rnn(train_season)
        X_train = X_train.to(device)

        #for i in range(T_pitcher.item(), T_train.max()):
        #    final_survival = pitcher_survival_overall[-1]
        #    new_survival = final_survival * (1 - final_hazard)
        #    pitcher_survival_overall = np.append(pitcher_survival_overall, new_survival)

        #print(pitcher_survival_overall)

        logits, _ = rnn_model(X_train)
        train_risk_predictions = logits[torch.arange(logits.shape[0]), T_train - 1]
        #print(train_risk_predictions)
        #print(pitcher_risk_predictions)

        # STEP 4: Use the predicted risk to compute the predicted survival curve for the pitcher for each
        # recurrence.
        S_x_t_overall, S_x_t = compute_survival_function_pitcher(T_pitcher, T_train, E_train,
                                          train_risk_predictions, pitcher_risk_predictions)

        #S_x_t_overall[:T_pitcher] = pitcher_survival_overall

        #print(S_x_t)
        #print(S_x_t_overall)
        # STEP 5: Average the predicted survival curves for each injury recurrence to get the ovreall
        # survival curve for the pitcher

        # STEP 6: Save the survival curves for the pitcher in the .png file
        plt.figure(figsize=(8,6))
        num_time_points = T_train.max() - T_train.min()
        #print(T_pitcher.max())
        #print(T_pitcher.min())
        times = np.linspace(T_train.min(), T_train.max(), num_time_points, endpoint=False)
        print(times)
        num_pitches = avg_pitches_per_game * times
        plt.step(num_pitches, S_x_t_overall, where="post", label=f"SurvivalRNN")

        # STEP 7: Compare the RNN survival curve with that of the baseline models
        # get the row in the survival dataframe for the pitcher
        survival_df_time_invariant_game_level_pitcher = pd.read_csv(f"Survival_Dataframe_Time_Invariant_2025_Pitchers/"
                                                                    f"survival_df_time_invariant_game_level_2025_{player_name}_processed.csv")

        X = survival_df_time_invariant_game_level_pitcher.drop(
            columns=['player_name', 'num_games', 'event', 'Batting Hand', 'Throwing Hand'])
        X.rename(columns={"Height": "height", "Weight": "weight"}, inplace=True)

        T_train = survival_df_time_invariant_train_season['num_games']
        num_time_points = T_train.max() - T_train.min() + 1
        time_points = np.linspace(T_train.min(), T_train.max() + 1, num_time_points, endpoint=False)
        s_ensemble = [[0] * len(time_points) for _ in range(5)]

        i = 0
        for model_name in ["cox", "weibull_aft"]:
            model = CoxPHFitter() if model_name == "cox" else WeibullAFTFitter()
            model.fit(survival_df_time_invariant_train_season, event_col="event", duration_col="num_games")
            model_survival = model.predict_survival_function(X, times=times)
            pred_survs = model_survival.T.values
            s_model = pred_survs[0]
            s_ensemble[i] = s_model
            plt.step(num_pitches, s_model, where="post", label=f"{model_name}")
            i += 1

        # the X_train for fitting the random survival forests
        survival_df_time_invariant_train_season['event'] = survival_df_time_invariant_train_season['event'].astype(bool)
        X_train = survival_df_time_invariant_train_season.drop(columns=['num_games', 'event'])

        # the y_train for fitting the random survival forests
        y_train = survival_df_time_invariant_train_season[['event', 'num_games']].to_records(index=False)

        for model_name in ["rsf", "gradient_boosting"]:
            model = (RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1,
                                          random_state=100) if model_name == "rsf"
                     else GradientBoostingSurvivalAnalysis(n_estimators=100, min_samples_split=10, min_samples_leaf=15,
                                                           random_state=100))

            model.fit(X_train.to_numpy(), y_train)
            model_survival = model.predict_survival_function(X)
            pred_surv = np.asarray([fn(times) for fn in model_survival])
            s_model = pred_surv[0]
            s_ensemble[i] = s_model
            plt.step(num_pitches, s_model, where="post", label=f"{model_name}")
            i += 1
        s_ensemble[4] = S_x_t_overall
        plt.ylabel("Survival probability")
        plt.xlabel("Number of Pitches")
        plt.title(f"{first_name} {last_name}")
        plt.legend()
        plt.savefig(f"Survival_Curves_Individual_Pitcher/survival_curve_pitcher_{player_name}_2025.png",
                    bbox_inches="tight")

        # STEP 8: Get the mean +- standard deviation of the survival model predictions
        s_ensemble = np.array(s_ensemble)

        mean_ensemble = np.mean(s_ensemble, axis=0)
        std_ensemble = np.std(s_ensemble, axis=0)

        plt.figure(figsize=(8, 6))
        plt.step(num_pitches, mean_ensemble, where="post")
        plt.fill_between(num_pitches, mean_ensemble - std_ensemble, mean_ensemble + std_ensemble, alpha=0.3)
        plt.ylabel("Survival probability")
        plt.xlabel("Number of Pitches")
        plt.title(f"{first_name} {last_name}")
        plt.savefig(f"Survival_Curves_Individual_Pitcher/mean_std_{player_name}_2025.png",
                    bbox_inches="tight")



    



