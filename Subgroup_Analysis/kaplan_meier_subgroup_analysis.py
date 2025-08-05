import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

import pandas as pd
from lifelines.statistics import logrank_test

#season_list = ["2021", "2022", "2023", "2024"]
season_list = ["2024"]
for evaluation_season in season_list:

    # read in the time invariant survival data
    survival_df_time_invariant_evaluation_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                               f"survival_df_time_invariant_game_{evaluation_season}"
                                                               f"_level_processed.csv")

    ## AGE SUBGROUP ANALYSIS
    # threshold for the subgroups for the age (maybe the 50th percentile)
    threshold_age = survival_df_time_invariant_evaluation_season.age.quantile(0.75)
    # are the survival curves different among the instances for the younger vs older pitchers
    group1_instance_season_age = survival_df_time_invariant_evaluation_season.loc[
        (survival_df_time_invariant_evaluation_season['age'] < threshold_age)]
    print(f"There are {len(group1_instance_season_age)} instances where age < {threshold_age}")

    group2_instance_season_age = survival_df_time_invariant_evaluation_season.loc[
        (survival_df_time_invariant_evaluation_season['age'] >= threshold_age)]
    print(f"There are {len(group2_instance_season_age)} instances where age >= {threshold_age}")


    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))

    kmf.fit(durations=group1_instance_season_age['num_games'], event_observed=group1_instance_season_age['event'],
            label=f"Age Group < {round(threshold_age,2)}")

    kmf.plot_survival_function()

    kmf.fit(durations=group2_instance_season_age['num_games'], event_observed=group2_instance_season_age['event'],
            label=f"Age Group >= {round(threshold_age, 2)}")
    kmf.plot_survival_function()

    plt.xlabel(f"Number of Games for {evaluation_season} Season")
    plt.ylabel("Survival Probability")

    plt.savefig(f"Subgroup_Survival_Curves/kmf_survival_curve_age_{evaluation_season}_season.png")

    # Perform the log rank test
    results = logrank_test(
        group1_instance_season_age['num_games'],
        group1_instance_season_age['num_games'],
        event_observed_A=group1_instance_season_age['event'],
        event_observed_B=group1_instance_season_age['event'],
    )

    # Print the results from the log rank test
    results.print_summary()

    ## PRIOR INJURY SUBGROUP ANALYSIS
    group1_instances_previous_injury = survival_df_time_invariant_evaluation_season.loc[
        survival_df_time_invariant_evaluation_season['recurrence'] != 0]
    group2_instances_no_previous_injury = survival_df_time_invariant_evaluation_season.loc[
        survival_df_time_invariant_evaluation_season['recurrence'] == 0]
    print(f"There are {len(group1_instances_previous_injury)} instances with previous injury")
    print(f"There are {len(group2_instances_no_previous_injury)} instances without previous injury")
    print(group1_instances_previous_injury.index)

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))

    kmf.fit(durations=group1_instances_previous_injury['num_games'], event_observed=group1_instances_previous_injury['event'],
            label=f"Previous Injury")

    kmf.plot_survival_function()

    kmf.fit(durations=group2_instances_no_previous_injury['num_games'], event_observed=group2_instances_no_previous_injury['event'],
            label=f"No Previous Injury")
    kmf.plot_survival_function()

    plt.xlabel(f"Number of Games for {evaluation_season} Season")
    plt.ylabel("Survival Probability")

    plt.savefig(f"Subgroup_Survival_Curves/kmf_survival_curve_previous_injury_{evaluation_season}_season.png")

    # Perform the log rank test
    results = logrank_test(
        group1_instances_previous_injury['num_games'],
        group2_instances_no_previous_injury['num_games'],
        event_observed_A=group1_instances_previous_injury['event'],
        event_observed_B=group2_instances_no_previous_injury['event'],
    )

    # Print the results from the log rank test
    results.print_summary()

    ## VELOCITY subgroup analysis
    threshold_velocity = survival_df_time_invariant_evaluation_season.avg_release_speed.quantile(0.75)
    group1_instances_velocity = survival_df_time_invariant_evaluation_season.loc[
        survival_df_time_invariant_evaluation_season['avg_release_speed'] < threshold_velocity
    ]
    group2_instances_velocity = survival_df_time_invariant_evaluation_season.loc[
        survival_df_time_invariant_evaluation_season['avg_release_speed'] >= threshold_velocity
    ]
    print(f"There are {len(group1_instances_velocity)} with average velocity below {threshold_velocity}")
    print(f"There are {len(group2_instances_velocity)} with average velocity at least {threshold_velocity}")

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))

    kmf.fit(durations=group1_instances_velocity['num_games'], event_observed=group1_instances_velocity['event'],
            label=f"Average Velocity Below {round(threshold_velocity,2)}")

    kmf.plot_survival_function()

    kmf.fit(durations=group2_instances_velocity['num_games'], event_observed=group2_instances_velocity['event'],
            label=f"Average Velocity At Least {round(threshold_velocity,2)}")
    kmf.plot_survival_function()

    plt.xlabel(f"Number of Games for {evaluation_season} Season")
    plt.ylabel("Survival Probability")

    plt.savefig(f"Subgroup_Survival_Curves/kmf_survival_curve_velocity_{evaluation_season}_season.png")

    # Perform the log rank test
    results = logrank_test(
        group1_instances_velocity['num_games'],
        group2_instances_velocity['num_games'],
        event_observed_A=group1_instances_velocity['event'],
        event_observed_B=group2_instances_velocity['event'],
    )

    # Print the results from the log rank test
    results.print_summary()

    # Combination of Age, Previous Injury, and Velocity
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

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    kmf.fit(durations=group1['num_games'], event_observed=group1['event'],
            label=f"Older, Previously Injured, and Throws Faster")

    kmf.plot_survival_function()

    kmf.fit(durations=group2['num_games'], event_observed=group2['event'],
            label=f"Younger, Not Previously Injured, and Throws Slower")
    kmf.plot_survival_function()

    plt.xlabel(f"Number of Games for {evaluation_season} Season")
    plt.ylabel("Survival Probability")

    plt.savefig(f"Subgroup_Survival_Curves/kmf_survival_curve_multiple_features_{evaluation_season}_season.png")

    # Perform the log rank test
    results = logrank_test(
        group1['num_games'],
        group2['num_games'],
        event_observed_A=group1['event'],
        event_observed_B=group2['event'],
    )

    # Print the results from the log rank test
    results.print_summary()