# for each of the seasons, read in the survival analysis dataframes
# get the number of instances (both injured and healthy)
import numpy as np

season_year = "2024"
import tqdm

import pandas as pd
survival_df_time_varying_game_level_season = pd.read_csv(f"../Survival-Dataframes/Time-Varying/survival_df_time_varying_game_level_{season_year}_processed.csv")
num_instances = len(survival_df_time_varying_game_level_season[['player_name', 'recurrence']].drop_duplicates())
pitcher_injury_combo_time_varying = survival_df_time_varying_game_level_season[['player_name', 'recurrence']].drop_duplicates()

print(f"There are {num_instances} instances for {season_year} for time-varying game-level survival dataframe")

# read in the time-invariant game-level survival dataframe for the season
survival_df_time_invariant_game_level_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/"
                                                           f"survival_df_time_invariant_game_{season_year}_level_processed.csv")
pitcher_injury_combo_time_invariant = survival_df_time_invariant_game_level_season[['player_name', 'recurrence']].drop_duplicates()
num_instances = len(pitcher_injury_combo_time_invariant)
print(f"There are {num_instances} instances for {season_year} for the time-invariant game-level survival dataframe")

event_list = []
time_list = []
for row in tqdm.tqdm(pitcher_injury_combo_time_invariant.iterrows()):
    # check if there exists a column where all the games are null
    time_invariant_entry_for_instance = survival_df_time_invariant_game_level_season[
        (survival_df_time_invariant_game_level_season['player_name'] == row[1].player_name)
        & (survival_df_time_invariant_game_level_season['recurrence'] == row[1].recurrence)]

    event = time_invariant_entry_for_instance['event']
    event_list.append(event)

    num_games = time_invariant_entry_for_instance['num_games']
    time_list.append(num_games)


pitcher_injury_combo_time_invariant['num_games'] = np.array(time_list)
pitcher_injury_combo_time_invariant['event'] = np.array(event_list)
print(pitcher_injury_combo_time_invariant['event'].value_counts())

pitcher_injury_combo_time_invariant.to_csv(f"pitcher_injury_combo_{season_year}_time_invariant.csv", index=False)
