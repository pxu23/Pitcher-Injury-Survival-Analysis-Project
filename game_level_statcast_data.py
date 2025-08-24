"""
    Get the game-level pitch features for the statcast data for the season.
"""
import os

import pandas as pd

def get_game_level_avg_pitch_features_statcast_by_player(season_year):
    """
        For each of the players, get the average pitch characteristics across all pitches in each game they played for the season.
        :param season_year: The season year
    """
    # read in the statcast pitch data fro the season
    statcast_df_season = pd.read_csv(f"Statcast_Data/statcast_data_{season_year}.csv")

    # groupby by player_name and game_date
    game_level_avg_pitch_features_season = statcast_df_season.groupby(['player_name', 'game_date']).agg(
                                                num_pitches=('game_date', 'count'),
                                                avg_release_spin_rate=('release_spin_rate', 'mean'),
                                                release_spin_rate_90_percentile = ('release_spin_rate', lambda x: x.quantile(0.9)),
                                                release_spin_rate_10_percentile = ('release_spin_rate',
                                                                                 lambda x: x.quantile(0.1)),
                                              avg_release_speed=('release_speed', 'mean'),
                                              release_speed_90_percentile = ('release_speed',
                                                                             lambda x: x.quantile(0.9)),
                                              release_speed_10_percentile = ('release_speed',
                                                                             lambda x: x.quantile(0.1)),
                                              avg_effective_speed=('effective_speed', 'mean'),
                                              effective_speed_90_percentile=('effective_speed',
                                                                             lambda x: x.quantile(0.9)),
                                              effective_speed_10_percentile=('effective_speed',
                                                                             lambda x: x.quantile(0.1)),

                                              avg_vx0=('vx0', 'mean'),
                                              vx0_90_percentile=('vx0', lambda x: x.quantile(0.9)),
                                              vx0_10_percentile=('vx0', lambda x: x.quantile(0.1)),

                                              avg_vy0=('vy0', 'mean'),
                                              vy0_90_percentile = ('vy0', lambda x: x.quantile(0.9)),
                                              vy0_10_percentile = ('vy0', lambda x: x.quantile(0.1)),

                                              avg_vz0=('vz0', 'mean'),
                                              vz0_90_percentile = ('vz0', lambda x: x.quantile(0.9)),
                                              vz0_10_percentile  = ('vz0', lambda x: x.quantile(0.1))

                                              # can add additional features such as pitch type, and more
                                              )


    # sort the values by player name and then game_date for the player
    game_level_avg_pitch_features_season.sort_values(by=['player_name', 'game_date'], inplace=True)

    # resets the index for the game-level pitch characteristics
    game_level_avg_pitch_features_season.reset_index(inplace=True)

    # saves the game-level pitch characteristics for csv file
    game_level_avg_pitch_features_season.to_csv(f"Game-Level-Pitch-Characteristics-Statcast/"
                                                f"avg_pitch_characteristics_game_level_{season_year}.csv", index=False)

def get_game_level_avg_pitch_features_statcast_by_player_multiple_season(start_season_year, end_season_year):
    """
        For each of the players, get the average pitch characteristics across all pitches in each game they played for the season.
        :param season_year: The season year
    """
    # read in the statcast pitch data fro the season
    statcast_df_multiple_season = pd.read_csv(f"Statcast_Data/statcast_data_{start_season_year}_{end_season_year}.csv")

    # groupby by player_name and game_date
    game_level_avg_pitch_features_multiple_season = statcast_df_multiple_season.groupby(['player_name', 'game_date']).agg(
                                                num_pitches=('game_date', 'count'),
                                                avg_release_spin_rate=('release_spin_rate', 'mean'),
                                                release_spin_rate_90_percentile = ('release_spin_rate', lambda x: x.quantile(0.9)),
                                                release_spin_rate_10_percentile = ('release_spin_rate',
                                                                                 lambda x: x.quantile(0.1)),
                                              avg_release_speed=('release_speed', 'mean'),
                                              release_speed_90_percentile = ('release_speed',
                                                                             lambda x: x.quantile(0.9)),
                                              release_speed_10_percentile = ('release_speed',
                                                                             lambda x: x.quantile(0.1)),
                                              avg_effective_speed=('effective_speed', 'mean'),
                                              effective_speed_90_percentile=('effective_speed',
                                                                             lambda x: x.quantile(0.9)),
                                              effective_speed_10_percentile=('effective_speed',
                                                                             lambda x: x.quantile(0.1)),

                                              avg_vx0=('vx0', 'mean'),
                                              vx0_90_percentile=('vx0', lambda x: x.quantile(0.9)),
                                              vx0_10_percentile=('vx0', lambda x: x.quantile(0.1)),

                                              avg_vy0=('vy0', 'mean'),
                                              vy0_90_percentile = ('vy0', lambda x: x.quantile(0.9)),
                                              vy0_10_percentile = ('vy0', lambda x: x.quantile(0.1)),

                                              avg_vz0=('vz0', 'mean'),
                                              vz0_90_percentile = ('vz0', lambda x: x.quantile(0.9)),
                                              vz0_10_percentile  = ('vz0', lambda x: x.quantile(0.1))

                                              # can add additional features such as pitch type, and more
                                              )


    # sort the values by player name and then game_date for the player
    game_level_avg_pitch_features_multiple_season.sort_values(by=['player_name', 'game_date'], inplace=True)

    # resets the index for the game-level pitch characteristics
    game_level_avg_pitch_features_multiple_season.reset_index(inplace=True)

    # saves the game-level pitch characteristics for csv file
    game_level_avg_pitch_features_multiple_season.to_csv(f"Game-Level-Pitch-Characteristics-Statcast/"
                                                f"avg_pitch_characteristics_game_level_{start_season_year}_{end_season_year}.csv", index=False)
if __name__=="__main__":
    # uncomment to get the game level statcast features for season if not already present
    #for season_year in [2021, 2022, 2023, 2024, 2025]:
    #    if not os.path.exists(f'Game-Level-Pitch-Characteristics-Statcast_{season_year}.csv'):
    #        get_game_level_avg_pitch_features_statcast_by_player(season_year)

    #get_game_level_avg_pitch_features_statcast_by_player(2018)
    #get_game_level_avg_pitch_features_statcast_by_player(2019)
    #get_game_level_avg_pitch_features_statcast_by_player(2020)

    game_level_features_2018 = pd.read_csv(f"Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2018.csv")
    game_level_features_2019 = pd.read_csv(f"Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2019.csv")
    game_level_features_2020 = pd.read_csv(f"Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2020.csv")
    game_level_features_2021 = pd.read_csv(f"Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2021.csv")
    game_level_features_2022 = pd.read_csv(f"Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2022.csv")
    game_level_features_2023 = pd.read_csv(f"Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2023.csv")

    game_level_features_2018_2023 = pd.concat([game_level_features_2018, game_level_features_2019,
                                               game_level_features_2020, game_level_features_2021, game_level_features_2022, game_level_features_2023],
                                              axis=0)
    game_level_features_2018_2023.to_csv(f"Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2018_2023.csv",
                                         index=False)

    game_level_features_2022_2023 = pd.concat([game_level_features_2022, game_level_features_2023],axis=0)
    game_level_features_2022_2023.to_csv(f"Game-Level-Pitch-Characteristics-Statcast/avg_pitch_characteristics_game_level_2022_2023.csv",
                                         index=False)
    #get_game_level_avg_pitch_features_statcast_by_player_multiple_season(2018, 2023)
    #get_game_level_avg_pitch_features_statcast_by_player_multiple_season(2022, 2023)