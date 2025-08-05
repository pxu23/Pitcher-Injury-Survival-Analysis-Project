"""
    Get the game-level pitch features for the statcast data for the season.
"""
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
def get_game_level_pitch_features_for_player(player_name):
    """
        For each of the players, get the average pitch characteristics across all
        pitches in each game for the season. in addition to the 90th percentile and the 10th percentile
        :param season_year: The season year
    """
    # statcast pitches for the season
    if not os.path.exists(f"Pitch_Data_2025_Pitchers/{player_name}_pitches.csv"):
        return

    statcast_pitches_player = pd.read_csv(f"Pitch_Data_2025_Pitchers/{player_name}_pitches.csv")

    # groupby by player_name and game_date
    game_level_avg_pitch_features_season = statcast_pitches_player.groupby(
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

    # sort the values by player name and then game_date
    game_level_avg_pitch_features_season.sort_values(by=['player_name', 'game_date'], inplace=True)

    game_level_avg_pitch_features_season.reset_index(inplace=True)

    game_level_avg_pitch_features_season.to_csv(f"Game_Level_Pitch_Data_2025_Pitchers/"
                                                f"avg_pitch_characteristics_game_level_{player_name}.csv", index=False)

if __name__ == "__main__":
    pitcher_list = [("Blake", "Snell"), ("Cade", "Povich"), ("Chris", "Sale"), ("Cole", "Ragans"),
                    ("Corbin", "Burnes"), ("Framber", "Valdez"), ("Gerrit", "Cole"), ("Jack", "Flaherty"),
                    ("Jacob", "deGrom"), ("Logan", "Gilbert"), ("Logan", "Webb"),
                    ("Max", "Fried"), ("Spencer", "Strider"), ("Tarik", "Skubal"),
                    ("Yoshinobu", "Yamamoto"), ("Zack", "Wheeler"), ("Paul", "Skenes"),
                    ("Kevin", "Gausman"), ("Shane", "McClanahan")]

    for first_name, last_name in pitcher_list:
        player_name = first_name.lower() + "_" + last_name.lower()
        get_game_level_pitch_features_for_player(player_name)
