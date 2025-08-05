## check the number of instances in which at least one column has all null values (for each game)
# read in the time-varying survival_dataframe
import os

import numpy as np
import pandas as pd
import tqdm
from Survival_Dataframe_Construction.demographics import get_demographic_information

def create_time_invariant_survival_dataframe(season_year, min_max, more_features, pitch_type):
    """
        Constructs the time invariant survival dataframe for the season
        :param season_year: the season year for the survival dataframe
        :param min_max: whether min and max of the features are considered
        :param more_features: whether more features are considered
        :param pitch_type: whether pitch type is considered
    """
    # read in the time-varying survival dataframe
    if min_max:
        survival_df_time_varying_game_level_processed = pd.read_csv(f"../Survival-Dataframes/Time-Varying/"
                                                                    f"survival_df_time_varying_game_level_{season_year}_min_max_processed.csv")
    elif more_features:
        survival_df_time_varying_game_level_processed = pd.read_csv(f"../Survival-Dataframes/Time-Varying/"
                                                                    f"survival_df_time_varying_game_level_{season_year}_more_features_processed.csv")
    elif pitch_type:
        survival_df_time_varying_game_level_processed = pd.read_csv(f"../Survival-Dataframes/Time-Varying/"
                                                                    f"survival_df_time_varying_game_level_{season_year}_processed_pitch_type.csv")
    else:
        survival_df_time_varying_game_level_processed = pd.read_csv(f"../Survival-Dataframes/Time-Varying/"
                                                                    f"survival_df_time_varying_game_level_{season_year}_processed.csv")

    # pitcher injury recurrence
    pitcher_injury_recurrence = survival_df_time_varying_game_level_processed[['player_name', 'recurrence']].drop_duplicates()

    rows = []
    for row in tqdm.tqdm(pitcher_injury_recurrence.iterrows()):
        # check if there exists a column where all the games are null
        time_varying_entry_for_instance = survival_df_time_varying_game_level_processed[
            (survival_df_time_varying_game_level_processed['player_name'] == row[1].player_name)
            & (survival_df_time_varying_game_level_processed['recurrence'] == row[1].recurrence)
        ]

        event = time_varying_entry_for_instance['event'].iloc[-1]

        num_games = time_varying_entry_for_instance.shape[0]

        # average pitcher characteristics
        avg_pitch_release_spin_rate = time_varying_entry_for_instance['avg_release_spin_rate'].mean()
        avg_pitch_release_speed = time_varying_entry_for_instance['avg_release_speed'].mean()
        avg_effective_speed = time_varying_entry_for_instance['avg_effective_speed'].mean()
        avg_vx0 = time_varying_entry_for_instance['avg_vx0'].mean()
        avg_vy0 = time_varying_entry_for_instance['avg_vy0'].mean()
        avg_vz0 = time_varying_entry_for_instance['avg_vz0'].mean()

        # average age
        avg_age = time_varying_entry_for_instance['age'].mean()
        avg_weight = time_varying_entry_for_instance['weight'].mean()
        avg_height = time_varying_entry_for_instance['height'].mean()
        avg_bats = time_varying_entry_for_instance['bats'].mean()
        avg_throws = time_varying_entry_for_instance['throws'].mean()

        row_dict = {"player_name": row[1].player_name,
                    "avg_release_spin_rate": avg_pitch_release_spin_rate,
                    "avg_release_speed": avg_pitch_release_speed,
                    "avg_effective_speed": avg_effective_speed,
                    "avg_vx0": avg_vx0, "avg_vy0": avg_vy0,
                    "avg_vz0": avg_vz0, "age": avg_age,
                    "height": avg_height, "weight": avg_weight,
                    "bats": avg_bats, "throws": avg_throws,
                    "recurrence": row[1].recurrence, "event": event,
                    "num_games": num_games}

        rows.append(row_dict)

    output_file_processed = f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{season_year}_level_processed.csv"

    survival_df_time_invariant_game_level = pd.DataFrame(rows)

    print(survival_df_time_invariant_game_level.shape)
    survival_df_time_invariant_game_level.dropna(inplace=True)

    survival_df_time_invariant_game_level.to_csv(output_file_processed, index=False)

if not os.path.exists(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{2021}_level_processed.csv"):
    create_time_invariant_survival_dataframe("2021", min_max=False, more_features=False, pitch_type=False)

if not os.path.exists(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{2022}_level_processed.csv"):
    create_time_invariant_survival_dataframe("2022", min_max=False, more_features=False, pitch_type=False)

if not os.path.exists(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{2023}_level_processed.csv"):
    create_time_invariant_survival_dataframe("2023", min_max=False, more_features=False, pitch_type=False)

if not os.path.exists(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{2024}_level_processed.csv"):
    create_time_invariant_survival_dataframe("2024", min_max=False, more_features=False, pitch_type=False)

survival_df = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_2024_level_processed.csv")
print(survival_df.shape)