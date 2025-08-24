import os

import numpy as np
import pandas as pd
import tqdm
from Survival_Dataframe_Construction.demographics import get_demographic_information

def create_time_invariant_survival_dataframe(season_year):
    """
        Constructs the time invariant survival dataframe for the season
        :param season_year: the followup season year
    """
    # read in the time-varying survival  for the season
    survival_df_time_varying_game_level_processed = pd.read_csv(f"../Survival-Dataframes/Time-Varying/"
                                                                f"survival_df_time_varying_game_level_{season_year}_processed.csv")

    # pitcher injury recurrence instances
    pitcher_injury_recurrence = survival_df_time_varying_game_level_processed[['player_name', 'recurrence']].drop_duplicates()

    rows = []
    for row in tqdm.tqdm(pitcher_injury_recurrence.iterrows()):
        # get the time-varying entries for the instance
        time_varying_entry_for_instance = survival_df_time_varying_game_level_processed[
            (survival_df_time_varying_game_level_processed['player_name'] == row[1].player_name)
            & (survival_df_time_varying_game_level_processed['recurrence'] == row[1].recurrence)
        ]
        # get the event for the instance
        event = time_varying_entry_for_instance['event'].iloc[-1]

        # get the number of games played
        num_games = time_varying_entry_for_instance.shape[0]

        # average pitcher workload characteristics
        avg_pitch_release_spin_rate = time_varying_entry_for_instance['avg_release_spin_rate'].mean()
        avg_pitch_release_speed = time_varying_entry_for_instance['avg_release_speed'].mean()
        avg_effective_speed = time_varying_entry_for_instance['avg_effective_speed'].mean()
        avg_vx0 = time_varying_entry_for_instance['avg_vx0'].mean()
        avg_vy0 = time_varying_entry_for_instance['avg_vy0'].mean()
        avg_vz0 = time_varying_entry_for_instance['avg_vz0'].mean()

        # average demographic features such as age, weight, height, batting hand, and throwing hand
        # they should not have changed
        avg_age = time_varying_entry_for_instance['age'].mean()
        avg_weight = time_varying_entry_for_instance['weight'].mean()
        avg_height = time_varying_entry_for_instance['height'].mean()
        avg_bats = time_varying_entry_for_instance['bats'].mean()
        avg_throws = time_varying_entry_for_instance['throws'].mean()

        # average number of pitches
        avg_pitches = time_varying_entry_for_instance['num_pitches'].mean()

        # row in the time-invariant survival dataframe for the instance
        row_dict = {"player_name": row[1].player_name,
                    "avg_release_spin_rate": avg_pitch_release_spin_rate,
                    "avg_release_speed": avg_pitch_release_speed,
                    "avg_effective_speed": avg_effective_speed,
                    "avg_vx0": avg_vx0, "avg_vy0": avg_vy0,
                    "avg_vz0": avg_vz0, "age": avg_age,
                    "avg_pitches": avg_pitches,
                    "height": avg_height, "weight": avg_weight,
                    "bats": avg_bats, "throws": avg_throws,
                    "recurrence": row[1].recurrence, "event": event,
                    "num_games": num_games}
        # add that row to the time-invariant survival dataframe for season
        rows.append(row_dict)

    # output file for the time-invariant survival dataframe
    output_file_processed = (f"../Survival-Dataframes/Time-Invariant/"
                             f"survival_df_time_invariant_game_{season_year}_level_processed.csv")

    survival_df_time_invariant_game_level = pd.DataFrame(rows)

    # remove the nan values
    survival_df_time_invariant_game_level.dropna(inplace=True)

    # save the dataframe to csv file
    survival_df_time_invariant_game_level.to_csv(output_file_processed, index=False)

if __name__=="__main__":
    if not os.path.exists(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{2021}_level_processed.csv"):
        create_time_invariant_survival_dataframe("2021")

    if not os.path.exists(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{2022}_level_processed.csv"):
        create_time_invariant_survival_dataframe("2022")

    if not os.path.exists(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{2023}_level_processed.csv"):
        create_time_invariant_survival_dataframe("2023")

    if not os.path.exists(f"../Survival-Dataframes/Time-Invariant/survival_df_time_invariant_game_{2024}_level_processed.csv"):
        create_time_invariant_survival_dataframe("2024")
