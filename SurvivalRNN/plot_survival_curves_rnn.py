# Plotting the overall survival curves for the Cox PH, Weibull AFT, Random Survival Forest, and Gradient Boosting Models
import numpy as np
from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt

train_evaluate_seasons = [("2021", "2022"), ("2022", "2023"), ("2023", "2024")]

for training_season,evaluation_season in train_evaluate_seasons:
    survival_df_time_varying_game_level = pd.read_csv(f"../Survival-Dataframes/Time-Varying/"
                                                      f"survival_df_time_varying_game_level_{evaluation_season}_processed.csv")
    # STEP 2: Get the average number of pitches in each game for season
    avg_pitches_per_game = survival_df_time_varying_game_level['num_pitches'].mean()
    print(f"There are on average {round(avg_pitches_per_game, 2)} pitches per game in {evaluation_season}")

    survival_df_time_invariant_train_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                          f"survival_df_time_invariant_game_{training_season}_level_processed.csv")
    survival_df_time_invariant_evaluation_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                               f"survival_df_time_invariant_game_{evaluation_season}"
                                                               f"_level_processed.csv")

    survival_df_time_invariant_evaluation_season.drop(columns=['player_name', 'bats', 'throws'], inplace=True)

    T_train = survival_df_time_invariant_train_season["num_games"]
    T_test = survival_df_time_invariant_evaluation_season["num_games"]

    num_time_points = min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())
    time_points = np.linspace(max(T_train.min(), T_test.min()), min(T_train.max(), T_test.max()),
                              num_time_points, endpoint=False)

    num_pitches = avg_pitches_per_game * time_points
    print(num_pitches)

    survival_df_time_invariant_game_level_evaluation_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                          f"survival_df_time_invariant_game_{evaluation_season}_level_processed.csv")

    plt.figure(figsize = (8,6))
    y_test = survival_df_time_invariant_game_level_evaluation_season[['event', 'num_games']]
    kmf = KaplanMeierFitter()
    kmf.fit(durations = y_test["num_games"], event_observed=y_test["event"])
    print(kmf.timeline)
    print(avg_pitches_per_game * kmf.timeline)
    plt.step(avg_pitches_per_game*kmf.timeline, kmf.survival_function_, label="Kaplan-Meier")
    #plt.fill_between(kmf.lower_bound, kmf.upper_bound)

    #plt.step(num_pitches, kmf.survival_function_)
    #plt.plot(kmf.survival_function_, label="Kaplan-Meier Fit")

    # Cox survival curve
    cox_survival = np.loadtxt(f"../Survival_Probabilities/cox_survival_train_{training_season}_eval_{evaluation_season}.txt")
    #print(cox_survival.shape)
    overall_survival_cox = cox_survival.mean(axis=0)
    #plt.step(time_points, overall_survival_cox, where="post", label="Cox Proportional Hazard")

    # Weibull AFT survival curve
    weibull_aft_survival = np.loadtxt(f"../Survival_Probabilities/weibull_aft_survival_train_{training_season}_eval_{evaluation_season}.txt")
    overall_survival_weibull_aft = weibull_aft_survival.mean(axis=0)
    #plt.step(time_points, overall_survival_weibull_aft, where="post", label="Weibull Accelerated Failure Time")

    # RSF survival curve
    rsf_survival = np.loadtxt(f"../Survival_Probabilities/rsf_survival_train_{training_season}_eval_{evaluation_season}.txt")
    overall_survival_rsf = rsf_survival.mean(axis=0)
    #plt.step(time_points,overall_survival_rsf, where="post", label="Random Survival Forest")

    # Gradient Boosting survival curve
    gradient_boosting_survival = np.loadtxt(f"../Survival_Probabilities/gradient_boosting_survival_train_{training_season}_eval_{evaluation_season}.txt")
    overall_survival_gradient_boosting = gradient_boosting_survival.mean(axis=0)
    #plt.step(time_points, overall_survival_gradient_boosting, where="post", label="Gradient Boosting")

    # RNN (Discrete Survival survival curve)
    rnn_survival = np.loadtxt(f"../Survival_Probabilities/survivalrnn_all_individual_survival_{training_season}_{evaluation_season}_w1_1_w2_4.txt")
    overall_survival_rnn = rnn_survival.mean(axis=0)
    #plt.step(time_points, overall_survival_rnn, where="post", label="SurvivalRNN")
    plt.step(num_pitches, overall_survival_rnn, where="post", label="Survival RNN")
    plt.xlabel("Number of Pitches")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.savefig(f"../Survival_Curves/survival_curve_overall_training_{training_season}_"
                f"evaluation_{evaluation_season}.png",bbox_inches='tight')

    # RNN (Cox Loss survival curves)
    rnn_cox_survival = np.loadtxt(f"../Survival_Probabilities/"
                                  f"survival_prob_rnn_cox_loss_{training_season}_{evaluation_season}.txt")
    overall_survival_rnn_cox = rnn_cox_survival.mean(axis=0)
    plt.step(time_points, overall_survival_rnn_cox, where="post", label="SurvivalRNN (Cox)")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.savefig(f"../Survival_Curves/survival_curve_overall_training_{training_season}_"
                f"evaluation_{evaluation_season}_rnn_cox.png", bbox_inches='tight')