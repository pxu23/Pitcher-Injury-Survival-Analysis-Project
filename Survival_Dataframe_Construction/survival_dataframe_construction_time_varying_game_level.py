import warnings
import os

import numpy as np

# the time varying nature of the games between the two injury dates for a pitcher in the followup period
from injury_data_utils import *
warnings.filterwarnings('ignore')

from demographics import get_demographic_information, get_players_with_demographic_information
import tqdm


def create_time_varying_survival_dataframe_seasonal_game_level(season_year, include_min_max,
                                                               more_features):
    # survival dataframe for all injured players in season across all their injury dates
    # read in the statcast data for the season
    if include_min_max:
        game_level_statcast_data_season = pd.read_csv(f"../Game-Level-Pitch-Characteristics-Statcast/"
                                                      f"avg_pitch_characteristics_game_level_{season_year}_min_max.csv")
    elif more_features:
        game_level_statcast_data_season = pd.read_csv(f"../Game-Level-Pitch-Characteristics-Statcast/"
                                                      f"avg_pitch_characteristics_game_level_{season_year}_more_features.csv")
    else:
        game_level_statcast_data_season = pd.read_csv(f"../Game-Level-Pitch-Characteristics-Statcast/"
                                               f"avg_pitch_characteristics_game_level_{season_year}.csv")
        game_level_statcast_data_season.drop(columns=["release_spin_rate_90_percentile",
                                                      "release_spin_rate_10_percentile",
                                                      "release_speed_90_percentile",
                                                      "release_speed_10_percentile",
                                                      "effective_speed_90_percentile",
                                                      "effective_speed_10_percentile",
                                                      "vx0_90_percentile",
                                                      "vx0_10_percentile",
                                                      "vy0_90_percentile",
                                                      "vy0_10_percentile",
                                                      "vz0_90_percentile",
                                                      "vz0_10_percentile"], inplace=True)
    game_level_statcast_data_season['game_date'] = game_level_statcast_data_season['game_date'].apply(convert_date_string_to_datetime)

    # get the injured pitchers for the season
    # dataframe for the injured players from 2020
    injured_players_df_season = get_injury_dataframe(season_year)

    # list of pitchers for the season (both healthy and injured)

    injured_pitchers_season_with_demographic_info = np.genfromtxt(f"../Pitchers_Season/"
                                                                  f"injured_pitchers_{season_year}_with_demographic_info.txt",
                                                                  dtype=str, delimiter="\n")
    healthy_pitchers_season_with_demographic_info = np.genfromtxt(f"../Pitchers_Season/"
                                                                  f"healthy_pitchers_{season_year}_with_demographic_info.txt",
                                                                  dtype=str, delimiter="\n")

    players_season_with_demographic_info = np.concatenate((healthy_pitchers_season_with_demographic_info,injured_pitchers_season_with_demographic_info))

    if include_min_max:
        output_file = f"../Survival-Dataframes/Time-Varying/survival_df_time_varying_game_level_{season_year}_min_max_processed.csv"
    elif more_features:
        output_file = f"../Survival-Dataframes/Time-Varying/survival_df_time_varying_game_level_{season_year}_more_features_processed.csv"
    else:
        output_file = f"../Survival-Dataframes/Time-Varying/survival_df_time_varying_game_level_{season_year}_processed.csv"

    if not os.path.exists(output_file):
        survival_df_list = list()

        for player_name in tqdm.tqdm(players_season_with_demographic_info):
            injury_dates_player = get_all_injury_dates(injured_players_df_season, player_name)
            augmented_injury_dates_player = np.insert(injury_dates_player, 0, datetime.date(int(season_year), 1, 1))
            augmented_injury_dates_player = np.append(augmented_injury_dates_player, datetime.date(int(season_year), 12, 31))
            game_level_time_varying_survival_df_player_list = list()

            num_previous_injury = 0
            for i in range(len(augmented_injury_dates_player) - 1):
                cur_injury_date = augmented_injury_dates_player[i]
                next_injury_date = augmented_injury_dates_player[i + 1]

                # get the games between the consecutive injury
                games_between_injury_player = game_level_statcast_data_season[
                    (game_level_statcast_data_season['game_date'] > cur_injury_date) &
                    (game_level_statcast_data_season['game_date'] < next_injury_date) &
                    (game_level_statcast_data_season['player_name'] == player_name)]

                num_games_between_injury = games_between_injury_player.shape[0]

                if num_games_between_injury == 0:
                    # only consider recurrences where the number of games played is more than zero
                    num_previous_injury += 1
                    continue

                # Compute the mean of each numerical column for missing data imputation
                mean_values = games_between_injury_player.select_dtypes(include='number').mean()

                # Fill missing values in numerical columns with the mean
                games_between_injury_player.fillna(mean_values, inplace=True)


                game_level_time_varying_survival_df_player_between_injury = games_between_injury_player
                game_level_time_varying_survival_df_player_between_injury['recurrence'] = num_previous_injury

                # the column for the start and stop
                game_level_time_varying_survival_df_player_between_injury['start'] = np.arange(num_games_between_injury)
                game_level_time_varying_survival_df_player_between_injury['stop'] = np.arange(1,num_games_between_injury + 1)

                # event status (whether injury occured after the last game)
                status = np.zeros((num_games_between_injury, 1))
                if num_previous_injury < len(injury_dates_player):
                    status[-1] = 1
                game_level_time_varying_survival_df_player_between_injury['event'] = status

                game_level_time_varying_survival_df_player_list.append(game_level_time_varying_survival_df_player_between_injury)
                num_previous_injury += 1

            if len(game_level_time_varying_survival_df_player_list) == 0:
                continue

            game_level_time_varying_survival_df_player = pd.concat(game_level_time_varying_survival_df_player_list, axis=0)
            #print(game_level_time_varying_survival_df_player)
            # get the demographic information of the pitcher
            demographic_info_player = get_demographic_information(player_name)

            # add the demographic information
            game_level_time_varying_survival_df_player = pd.merge(game_level_time_varying_survival_df_player,demographic_info_player, on='player_name')

            # convert the covariates for the bats and throws to categorical codes
            game_level_time_varying_survival_df_player['bats'] = pd.Categorical(
                game_level_time_varying_survival_df_player['bats']).codes
            game_level_time_varying_survival_df_player['throws'] = pd.Categorical(
                game_level_time_varying_survival_df_player['throws']).codes

            #print(game_level_time_varying_survival_df_player)
            game_level_time_varying_survival_df_player['birth_date'] = pd.to_datetime(
                {'year': game_level_time_varying_survival_df_player['birthYear'],
                 'month': game_level_time_varying_survival_df_player['birthMonth'],
                 'day': game_level_time_varying_survival_df_player['birthDay']})

            game_level_time_varying_survival_df_player['age'] = (pd.to_datetime(
                game_level_time_varying_survival_df_player['game_date']) - game_level_time_varying_survival_df_player['birth_date']).dt.days / 365.25
            game_level_time_varying_survival_df_player.drop(
                columns=['birthYear', 'birthMonth', 'birthDay', 'birth_date'], inplace=True)
            survival_df_list.append(game_level_time_varying_survival_df_player)

        survival_df_season_time_varying_game_level = pd.concat(survival_df_list, axis=0)
        survival_df_season_time_varying_game_level.dropna(inplace=True)
        survival_df_season_time_varying_game_level.to_csv(output_file,index=False)

        pitcher_injury_combo = survival_df_season_time_varying_game_level[['player_name', 'recurrence']].drop_duplicates()
        num_instances = pitcher_injury_combo.shape[0]

        print(f"For time-varying case, there are {num_instances} instances")

if __name__=="__main__":
    # create the survival dataframe for each season from 2020 to 2024
    create_time_varying_survival_dataframe_seasonal_game_level("2021", False, False)

    create_time_varying_survival_dataframe_seasonal_game_level("2022", False, False)

    create_time_varying_survival_dataframe_seasonal_game_level("2023", False, False)

    create_time_varying_survival_dataframe_seasonal_game_level("2024", False, False)

