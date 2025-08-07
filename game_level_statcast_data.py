"""
    Get the game-level pitch features for the statcast data for the season.
"""
import pandas as pd

def get_game_level_avg_pitch_features_statcast_by_player(season_year):
    """
        For each of the players, get the average pitch characteristics across all
        pitches in each game for the season.
        :param season_year: The season year
    """
    # statcast pitches for the season
    statcast_df_season = pd.read_csv(f"Statcast_Data/statcast_data_{season_year}.csv")

    # groupby by player_name and game_date
    game_level_avg_pitch_features_season = statcast_df_season.groupby(
        ['player_name', 'game_date']).agg(num_pitches=('game_date', 'count'),
                                          avg_release_spin_rate=('release_spin_rate', 'mean'),
                                          release_spin_rate_90_percentile = ('release_spin_rate', lambda x: x.quantile(0.9)),
                                          release_spin_rate_10_percentile = ('release_spin_rate', lambda x: x.quantile(0.1)),

                                          avg_release_speed=('release_speed', 'mean'),
                                          release_speed_90_percentile = ('release_speed', lambda x: x.quantile(0.9)),
                                          release_speed_10_percentile = ('release_speed', lambda x: x.quantile(0.1)),

                                          avg_effective_speed=('effective_speed', 'mean'),
                                          effective_speed_90_percentile=('effective_speed', lambda x: x.quantile(0.9)),
                                          effective_speed_10_percentile=('effective_speed', lambda x: x.quantile(0.1)),

                                          avg_vx0=('vx0', 'mean'),
                                          vx0_90_percentile=('vx0', lambda x: x.quantile(0.9)),
                                          vx0_10_percentile=('vx0', lambda x: x.quantile(0.1)),

                                          avg_vy0=('vy0', 'mean'),
                                          vy0_90_percentile = ('vy0', lambda x: x.quantile(0.9)),
                                          vy0_10_percentile = ('vy0', lambda x: x.quantile(0.1)),

                                          avg_vz0=('vz0', 'mean'),
                                          vz0_90_percentile = ('vz0', lambda x: x.quantile(0.9)),
                                          vz0_10_percentile  = ('vz0', lambda x: x.quantile(0.1)))

                                          # age of the pitcher (age_pit not available for every Statcast year)
                                          #age = ('age_pit', 'mean'))

                                          # whether the throwing is left or right-handed
                                          #pitching_hand = ('p_throws', lambda x: x.mode()[0] if not x.mode().empty else None))

                                          # number of days since last game
                                          #num_games_since_last_game = ('pitcher_days_since_prev_game', 'mean'))


    # sort the values by player name and then game_date
    game_level_avg_pitch_features_season.sort_values(by=['player_name', 'game_date'], inplace=True)

    game_level_avg_pitch_features_season.reset_index(inplace=True)

    game_level_avg_pitch_features_season.to_csv(f"Game-Level-Pitch-Characteristics-Statcast/"
                                                f"avg_pitch_characteristics_game_level_{season_year}.csv", index=False)

def get_game_level_avg_pitch_features_statcast_by_player_multiple_seasons(start_season_year,
                                                                          end_season_year):
    """
        For each of the players, get the average pitch characteristics across all
        pitches in each game for multiple seasons.
        :param start_season_year: The start season year
        :param end_season_year: The end season year
    """
    # statcast pitches for the multiple season
    statcast_df_multiple_season = pd.read_csv(f"Statcast_Data/statcast_data_{start_season_year}_{end_season_year}.csv")

    # groupby by player_name and game_date
    game_level_avg_pitch_features_multiple_season = statcast_df_multiple_season.groupby(['player_name', 'game_date']).agg(num_pitches=('game_date', 'count'),
                                          avg_release_spin_rate=('release_spin_rate', 'mean'),
                                          release_spin_rate_90_percentile = ('release_spin_rate', lambda x: x.quantile(0.9)),
                                          release_spin_rate_10_percentile = ('release_spin_rate', lambda x: x.quantile(0.1)),

                                          avg_release_speed=('release_speed', 'mean'),
                                          release_speed_90_percentile = ('release_speed', lambda x: x.quantile(0.9)),
                                          release_speed_10_percentile = ('release_speed', lambda x: x.quantile(0.1)),

                                          avg_effective_speed=('effective_speed', 'mean'),
                                          effective_speed_90_percentile=('effective_speed', lambda x: x.quantile(0.9)),
                                          effective_speed_10_percentile=('effective_speed', lambda x: x.quantile(0.1)),

                                          avg_vx0=('vx0', 'mean'),
                                          vx0_90_percentile=('vx0', lambda x: x.quantile(0.9)),
                                          vx0_10_percentile=('vx0', lambda x: x.quantile(0.1)),

                                          avg_vy0=('vy0', 'mean'),
                                          vy0_90_percentile = ('vy0', lambda x: x.quantile(0.9)),
                                          vy0_10_percentile = ('vy0', lambda x: x.quantile(0.1)),

                                          avg_vz0=('vz0', 'mean'),
                                          vz0_90_percentile = ('vz0', lambda x: x.quantile(0.9)),
                                          vz0_10_percentile  = ('vz0', lambda x: x.quantile(0.1)))

    # sort the values by player name and then game_date
    game_level_avg_pitch_features_multiple_season.sort_values(by=['player_name', 'game_date'], inplace=True)

    game_level_avg_pitch_features_multiple_season.reset_index(inplace=True)

    game_level_avg_pitch_features_multiple_season.to_csv(f"Game-Level-Pitch-Characteristics-Statcast/"
                                                f"avg_pitch_characteristics_game_level_{start_season_year}_{end_season_year}.csv", index=False)


#get_game_level_avg_pitch_features_statcast_by_player(2020)
#get_game_level_avg_pitch_features_statcast_by_player(2021)
#get_game_level_avg_pitch_features_statcast_by_player(2022)
#get_game_level_avg_pitch_features_statcast_by_player(2023)
#get_game_level_avg_pitch_features_statcast_by_player(2024)
#get_game_level_avg_pitch_features_statcast_by_player(2025)

get_game_level_avg_pitch_features_statcast_by_player_multiple_seasons(2018, 2023)
get_game_level_avg_pitch_features_statcast_by_player_multiple_seasons(2020, 2023)

get_game_level_avg_pitch_features_statcast_by_player_multiple_seasons(2021, 2023)