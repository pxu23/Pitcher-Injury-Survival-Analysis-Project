# read in the time invariant survival data
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

for (train_season, evaluation_season) in [("2021", "2022"), ("2022", "2023"), ("2023", "2024")]:
    model_name = "survivalrnn"
    survival_df_time_invariant_train_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                  f"survival_df_time_invariant_game_{train_season}"
                                                                  f"_level_processed.csv")
    survival_df_time_invariant_evaluation_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                                       f"survival_df_time_invariant_game_{evaluation_season}"
                                                                       f"_level_processed.csv")

    survival_df_time_invariant_evaluation_season.drop(columns=['player_name', 'bats', 'throws'], inplace=True)

    T_train = survival_df_time_invariant_train_season["num_games"]
    T_test = survival_df_time_invariant_evaluation_season["num_games"]

    num_time_points =  min(T_train.max(), T_test.max()) - max(T_train.min(), T_test.min())
    time_points = np.linspace(max(T_train.min(), T_test.min()), min(T_train.max(), T_test.max()),
                                     num_time_points, endpoint=False)
    # STEP 2: Get the average number of pitches in each game for season
    survival_df_time_varying_game_level = pd.read_csv(f"../Survival-Dataframes/Time-Varying/"
                                                              f"survival_df_time_varying_game_level_{evaluation_season}_processed.csv")
    avg_pitches_per_game = survival_df_time_varying_game_level['num_pitches'].mean()
    num_pitches = avg_pitches_per_game * time_points

    survival_probabilities = np.loadtxt(f"../Survival_Probabilities/"
                                        f"survivalrnn_all_individual_survival_{train_season}_{evaluation_season}_w1_1_w2_4.txt")

    ## AGE SUBGROUP ANALYSIS
    # threshold for the subgroups for the age (maybe the 50th percentile)
    threshold_age = survival_df_time_invariant_evaluation_season["age"].quantile(0.75)
    # are the survival curves different among the instances for the younger vs older pitchers
    group1_instance_season_age = survival_df_time_invariant_evaluation_season.loc[(survival_df_time_invariant_evaluation_season['age'] < threshold_age)]

    print(f"There are {len(group1_instance_season_age)} instances where age < {threshold_age}")
    survival_probabilities_group1_age = survival_probabilities[group1_instance_season_age.index]

    group2_instance_season_age = survival_df_time_invariant_evaluation_season.loc[
                (survival_df_time_invariant_evaluation_season['age'] >= threshold_age)]
    print(f"There are {len(group2_instance_season_age)} instances where age >= {threshold_age}")
    survival_probabilities_group2_age = survival_probabilities[group2_instance_season_age.index]

    overall_survival_group1_age = survival_probabilities_group1_age.mean(axis=0)
    overall_survival_group2_age = survival_probabilities_group2_age.mean(axis=0)

    plt.figure()
    plt.step(num_pitches, overall_survival_group1_age, where='post', label=f"Age < {round(threshold_age, 2)}")
    plt.step(num_pitches, overall_survival_group2_age, where='post', label=f"Age >= {round(threshold_age, 2)}")
    plt.xlabel("Number of Pitches")
    plt.ylabel("Survival Probability")
    plt.legend()
    if not os.path.exists(f"Subgroup_Survival_Curves/"
                        f"{model_name}_survival_age_train_{train_season}_evaluate_{evaluation_season}.png"):
        plt.savefig(f"Subgroup_Survival_Curves/"
                            f"{model_name}_survival_age_train_{train_season}_evaluate_{evaluation_season}.png",
                            bbox_inches='tight')

    ## PRIOR INJURY SUBGROUP ANALYSIS
    group1_instances_previous_injury = survival_df_time_invariant_evaluation_season.loc[
                survival_df_time_invariant_evaluation_season['recurrence'] != 0]
    group2_instances_no_previous_injury = survival_df_time_invariant_evaluation_season.loc[
                survival_df_time_invariant_evaluation_season['recurrence'] == 0]
    print(f"There are {len(group1_instances_previous_injury)} instances with previous injury")
    print(f"There are {len(group2_instances_no_previous_injury)} instances without previous injury")
    print(group1_instances_previous_injury.index)
    survival_probabilities_group1_previous_injury = survival_probabilities[group1_instances_previous_injury.index]
    survival_probabilities_group2_no_previous_injury = survival_probabilities[group2_instances_no_previous_injury.index]
    overall_survival_group1_previous_injury = survival_probabilities_group1_previous_injury.mean(axis=0)
    overall_survival_group2_no_previous_injury = survival_probabilities_group2_no_previous_injury.mean(axis=0)

    plt.figure()
    plt.step(num_pitches, overall_survival_group1_previous_injury, label="Previous Injury")
    plt.step(num_pitches, overall_survival_group2_no_previous_injury, label="No Previous Injury")
    plt.xlabel("Number of Pitches")
    plt.ylabel("Survival Probability")
    plt.legend()
    if not os.path.exists(f"Subgroup_Survival_Curves/"
                        f"{model_name}_survival_previous_injury_train_{train_season}_evaluate_{evaluation_season}.png"):
        plt.savefig(f"Subgroup_Survival_Curves/"
                            f"{model_name}_survival_previous_injury_train_{train_season}_evaluate_{evaluation_season}.png",
                            bbox_inches='tight')

            ## VELOCITY subgroup analysis
    threshold_velocity = survival_df_time_invariant_evaluation_season["avg_release_speed"].quantile(0.75)
    group1_instances_velocity = survival_df_time_invariant_evaluation_season.loc[
                survival_df_time_invariant_evaluation_season['avg_release_speed'] < threshold_velocity
            ]
    group2_instances_velocity = survival_df_time_invariant_evaluation_season.loc[
                survival_df_time_invariant_evaluation_season['avg_release_speed'] >= threshold_velocity
            ]
    print(f"There are {len(group1_instances_velocity)} with average velocity below {threshold_velocity}")
    print(f"There are {len(group2_instances_velocity)} with average velocity at least {threshold_velocity}")

    survival_probabilities_group1_velocity = survival_probabilities[group1_instances_velocity.index]
    survival_probabilities_group2_velocity = survival_probabilities[group2_instances_velocity.index]
    overall_survival_group1_velocity = survival_probabilities_group1_velocity.mean(axis=0)
    overall_survival_group2_velocity = survival_probabilities_group2_velocity.mean(axis=0)
    plt.figure()
    plt.step(num_pitches, overall_survival_group1_velocity,
                     label=f"Average Velocity Below {round(threshold_velocity,2)}")
    plt.step(num_pitches, overall_survival_group2_velocity,
                     label=f"Average Velocity At Least {round(threshold_velocity,2)}")
    plt.xlabel("Number of Pitches")
    plt.ylabel("Survival Probability")
    plt.legend()
    if not os.path.exists(f"Subgroup_Survival_Curves/"
                        f"{model_name}_survival_velocity_train_{train_season}_evaluate_{evaluation_season}.png"):
        plt.savefig(f"Subgroup_Survival_Curves/"
                            f"{model_name}_survival_velocity_train_{train_season}_evaluate_{evaluation_season}.png",
                            bbox_inches='tight')

    # Combination of Age, Previous Injury, and Velocity
    # Older, Throw Faster, and Injured Previously
    group1 = survival_df_time_invariant_evaluation_season.loc[
                (survival_df_time_invariant_evaluation_season['avg_release_speed'] >= threshold_velocity) &
                (survival_df_time_invariant_evaluation_season['age'] >= threshold_age) &
                (survival_df_time_invariant_evaluation_season['recurrence'] != 0)
            ]

    group2 = survival_df_time_invariant_evaluation_season.loc[
                (survival_df_time_invariant_evaluation_season['avg_release_speed'] < threshold_velocity) &
                (survival_df_time_invariant_evaluation_season['age'] < threshold_age) &
                (survival_df_time_invariant_evaluation_season['recurrence'] == 0)
            ]

    print(f"There are {len(group1)} older, throws faster, and is previously injured")
    print(f"There are {len(group2)} younger, throws slower, and not previously injured")

    survival_probabilities_group1 = survival_probabilities[group1.index]
    survival_probabilities_group2 = survival_probabilities[group2.index]
    overall_survival_group1 = survival_probabilities_group1.mean(axis=0)
    overall_survival_group2 = survival_probabilities_group2.mean(axis=0)
    plt.figure()
    plt.step(num_pitches, overall_survival_group1,
                     label=f"Older, Faster, and Previously Injured")
    plt.step(num_pitches, overall_survival_group2,
                     label=f"Younger, Slower, and Not Previously Injured")
    plt.xlabel("Number of Pitches")
    plt.ylabel("Survival Probability")
    plt.legend()
    if not os.path.exists(f"Subgroup_Survival_Curves/"
                        f"{model_name}_survival_more_features_train_{train_season}_evaluate_{evaluation_season}.png"):
        plt.savefig(f"Subgroup_Survival_Curves/"
                            f"{model_name}_survival_more_features_train_{train_season}_evaluate_{evaluation_season}.png",
                            bbox_inches='tight')