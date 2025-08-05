# read in the time-invariant survival dataframe for the season
import pandas as pd

for season_year in ["2021", "2022", "2023", "2024"]:
    output_file = open(f"Feature_Analysis/survival_df_analysis_{season_year}.txt", "w")
    file_name = f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{season_year}_level_processed.csv"
    survival_df_time_invariant_game_level_season = pd.read_csv(file_name)

    # get the number of instances
    num_instances = len(survival_df_time_invariant_game_level_season)

    injured_instances = survival_df_time_invariant_game_level_season[survival_df_time_invariant_game_level_season["event"] == 1]
    censored_instances = survival_df_time_invariant_game_level_season[survival_df_time_invariant_game_level_season["event"] == 0]

    # numerical features used
    numerical_feature_names = ["age", "avg_release_speed", "avg_release_spin_rate", "num_games",
                     "avg_effective_speed", "avg_vx0", "avg_vy0", "avg_vz0", "height", "weight"]

    for i, feature in enumerate(numerical_feature_names):
        if i > 0:
            print("\n\n", file=output_file)
        print(f"Processing {feature} feature", file=output_file)
        print("Overall", file=output_file)
        feature_min = round(survival_df_time_invariant_game_level_season[feature].min(), 2)
        feature_10_percentile = round(survival_df_time_invariant_game_level_season[feature].quantile(0.1), 2)
        feature_25_percentile = round(survival_df_time_invariant_game_level_season[feature].quantile(0.25), 2)
        feature_50_percentile = round(survival_df_time_invariant_game_level_season[feature].quantile(0.5), 2)
        feature_75_percentile = round(survival_df_time_invariant_game_level_season[feature].quantile(0.75), 2)
        feature_90_percentile = round(survival_df_time_invariant_game_level_season[feature].quantile(0.9), 2)
        feature_max = round(survival_df_time_invariant_game_level_season[feature].max(), 2)
        print(f"Feature: {feature}", file=output_file)
        print(f"Minimum for {feature}: {feature_min}", file=output_file)
        print(f"10th Percentile for {feature}: {feature_10_percentile}", file=output_file)
        print(f"25th Percentile for {feature}: {feature_25_percentile}", file=output_file)
        print(f"50th Percentile for {feature}: {feature_50_percentile}", file=output_file)
        print(f"75th Percentile for {feature}: {feature_75_percentile}", file=output_file)
        print(f"90th Percentile for {feature}: {feature_90_percentile}", file=output_file)
        print(f"Maximum for {feature}: {feature_max}", file=output_file)

        print("\nInjured", file=output_file)
        feature_min_injured = round(injured_instances[feature].min(), 2)
        feature_10_percentile_injured = round(injured_instances[feature].quantile(0.1), 2)
        feature_25_percentile_injured = round(injured_instances[feature].quantile(0.25), 2)
        feature_50_percentile_injured = round(injured_instances[feature].quantile(0.5), 2)
        feature_75_percentile_injured = round(injured_instances[feature].quantile(0.75), 2)
        feature_90_percentile_injured = round(injured_instances[feature].quantile(0.9), 2)
        feature_max_injured = round(injured_instances[feature].max(), 2)
        print(f"Feature: {feature}", file=output_file)
        print(f"Minimum for {feature}: {feature_min_injured}", file=output_file)
        print(f"10th Percentile for {feature}: {feature_10_percentile_injured}", file=output_file)
        print(f"25th Percentile for {feature}: {feature_25_percentile_injured}", file=output_file)
        print(f"50th Percentile for {feature}: {feature_50_percentile_injured}", file=output_file)
        print(f"75th Percentile for {feature}: {feature_75_percentile_injured}", file=output_file)
        print(f"90th Percentile for {feature}: {feature_90_percentile_injured}", file=output_file)
        print(f"Maximum for {feature}: {feature_max_injured}", file=output_file)

        print("\nCensored", file=output_file)
        feature_min_censored = round(censored_instances[feature].min(), 2)
        feature_10_percentile_censored = round(censored_instances[feature].quantile(0.1), 2)
        feature_25_percentile_censored = round(censored_instances[feature].quantile(0.25), 2)
        feature_50_percentile_censored = round(censored_instances[feature].quantile(0.5), 2)
        feature_75_percentile_censored = round(censored_instances[feature].quantile(0.75), 2)
        feature_90_percentile_censored = round(censored_instances[feature].quantile(0.9), 2)
        feature_max_censored = round(censored_instances[feature].max(), 2)
        print(f"Feature: {feature}", file=output_file)
        print(f"Minimum for {feature}: {feature_min_censored}", file=output_file)
        print(f"10th Percentile for {feature}: {feature_10_percentile_censored}", file=output_file)
        print(f"25th Percentile for {feature}: {feature_25_percentile_censored}", file=output_file)
        print(f"50th Percentile for {feature}: {feature_50_percentile_censored}", file=output_file)
        print(f"75th Percentile for {feature}: {feature_75_percentile_censored}", file=output_file)
        print(f"90th Percentile for {feature}: {feature_90_percentile_censored}", file=output_file)
        print(f"Maximum for {feature}: {feature_max_censored}", file=output_file)

    # look at the categorical variables
    categorical_features = ["recurrence","bats", "throws"]
    for i, feature in enumerate(categorical_features):
        if i > 0:
            print("\n\n", file=output_file)
        print(f"Processing {feature} feature", file=output_file)
        print("Overall", file=output_file)
        print(survival_df_time_invariant_game_level_season[feature].value_counts(), file=output_file)
        print("\nInjured", file=output_file)
        print(injured_instances[feature].value_counts(), file=output_file)
        print("\nCensored", file=output_file)
        print(censored_instances[feature].value_counts(), file=output_file)

    output_file.close()
