import warnings
import os

from injury_data_utils import *
warnings.filterwarnings('ignore')

from demographics import get_demographic_information
import tqdm

def create_time_varying_survival_dataframe_seasonal_game_level(season_year):
    """
        Constructs time-varying survival dataframe for all healthy and injured players in season across all their injury dates
        :param season_year: year of the followup season
    """
    # read in the game-level statcast data for the season
    game_level_statcast_data_season = pd.read_csv(f"../Game-Level-Pitch-Characteristics-Statcast/"
                                               f"avg_pitch_characteristics_game_level_{season_year}.csv")

    # since we are not considering the 10th and 90th percentiles, we drop them for now
    game_level_statcast_data_season.drop(columns=["release_spin_rate_90_percentile","release_spin_rate_10_percentile",
                                                  "release_speed_90_percentile","release_speed_10_percentile",
                                                  "effective_speed_90_percentile", "effective_speed_10_percentile",
                                                  "vx0_90_percentile", "vx0_10_percentile",
                                                  "vy0_90_percentile", "vy0_10_percentile",
                                                  "vz0_90_percentile", "vz0_10_percentile"], inplace=True)

    # convert the game date into datetime object
    game_level_statcast_data_season['game_date'] = game_level_statcast_data_season['game_date'].apply(convert_date_string_to_datetime)

    # get the dataframe for the injured players in the season that contains the injury dates
    injured_players_df_season = get_injury_dataframe(season_year)

    # read in the list of injured pitchers with demographic information
    injured_pitchers_season_with_demographic_info = np.genfromtxt(f"../Pitchers_Season/"
                                                                  f"injured_pitchers_{season_year}_with_demographic_info.txt",
                                                                  dtype=str, delimiter="\n")
    # read in the list of healthy pitchers with demographic information
    healthy_pitchers_season_with_demographic_info = np.genfromtxt(f"../Pitchers_Season/"
                                                                  f"healthy_pitchers_{season_year}_with_demographic_info.txt",
                                                                  dtype=str, delimiter="\n")
    # concatenate to get all the players in the season with demographic information
    players_season_with_demographic_info = np.concatenate((healthy_pitchers_season_with_demographic_info,
                                                           injured_pitchers_season_with_demographic_info))

    # output file for the time-varying survival dataframe
    output_file = f"../Survival-Dataframes/Time-Varying/survival_df_time_varying_game_level_{season_year}_processed.csv"

    if not os.path.exists(output_file):
        survival_df_list = list()

        # loop through the list of pitchers with demographic information
        for player_name in tqdm.tqdm(players_season_with_demographic_info):
            # get the injury dates for the player from the dataframe for season
            injury_dates_player = get_all_injury_dates(injured_players_df_season, player_name)

            season = int(season_year)
            # add the start date 1/1 of the followup season
            augmented_injury_dates_player = np.insert(injury_dates_player, 0,
                                                      datetime.date(season,1,1))
            # add the end date 12/31 of the followup season
            augmented_injury_dates_player = np.append(augmented_injury_dates_player,
                                                      datetime.date(season,12,31))

            # survival dataframe for player across all its injures
            game_level_time_varying_survival_df_player_list = list()

            # number of previous injury to account for dependence among multiple injures for player
            num_previous_injury = 0
            for i in range(len(augmented_injury_dates_player) - 1):
                # two subsequence injury dates
                cur_injury_date = augmented_injury_dates_player[i]
                next_injury_date = augmented_injury_dates_player[i + 1]

                # get the games between the consecutive injury
                games_between_injury_player = game_level_statcast_data_season[
                    (game_level_statcast_data_season['game_date'] > cur_injury_date) &
                    (game_level_statcast_data_season['game_date'] < next_injury_date) &
                    (game_level_statcast_data_season['player_name'] == player_name)]

                # get the number of games between the two consecutive injuries
                num_games_between_injury = games_between_injury_player.shape[0]

                if num_games_between_injury == 0:
                    # only consider recurrences where the number of games played is more than zero
                    num_previous_injury += 1
                    continue

                # Compute the mean of each numerical column for missing data imputation
                mean_values = games_between_injury_player.select_dtypes(include='number').mean()

                # Fill missing values in numerical columns with the mean
                games_between_injury_player.fillna(mean_values, inplace=True)

                # add the recurrence to the game-level time-varying survival dataframe for player betwen injury
                game_level_time_varying_survival_df_player_between_injury = games_between_injury_player
                game_level_time_varying_survival_df_player_between_injury['recurrence'] = num_previous_injury

                # the column for the start and stop game
                game_level_time_varying_survival_df_player_between_injury['start'] = np.arange(num_games_between_injury)
                game_level_time_varying_survival_df_player_between_injury['stop'] = np.arange(1,num_games_between_injury + 1)

                # event status (whether event occurred after the last game before possible event)
                status = np.zeros((num_games_between_injury, 1))
                if num_previous_injury < len(injury_dates_player):
                    status[-1] = 1
                game_level_time_varying_survival_df_player_between_injury['event'] = status

                # add the game-level time-varying survival dataframe between injury
                game_level_time_varying_survival_df_player_list.append(game_level_time_varying_survival_df_player_between_injury)
                num_previous_injury += 1

            # if no games are played by the player between any two consecutive injures, skip
            if len(game_level_time_varying_survival_df_player_list) == 0:
                continue

            # get the game-level time-varying survival dataframe for the player
            game_level_time_varying_survival_df_player = pd.concat(game_level_time_varying_survival_df_player_list, axis=0)

            # get the demographic information of the pitcher
            demographic_info_player = get_demographic_information(player_name)

            # add the demographic information to the game-level time-varying survival dataframe for player
            game_level_time_varying_survival_df_player = pd.merge(game_level_time_varying_survival_df_player,
                                                                  demographic_info_player, on='player_name')

            # convert the covariates for the bats and throws to categorical codes
            game_level_time_varying_survival_df_player['bats'] = pd.Categorical(
                game_level_time_varying_survival_df_player['bats']).codes
            game_level_time_varying_survival_df_player['throws'] = pd.Categorical(
                game_level_time_varying_survival_df_player['throws']).codes

            # add a new column for the birth_date in order to compute the age
            game_level_time_varying_survival_df_player['birth_date'] = pd.to_datetime(
                {'year': game_level_time_varying_survival_df_player['birthYear'],
                 'month': game_level_time_varying_survival_df_player['birthMonth'],
                 'day': game_level_time_varying_survival_df_player['birthDay']})

            # computes the age of the player
            game_level_time_varying_survival_df_player['age'] = (pd.to_datetime(
                game_level_time_varying_survival_df_player['game_date']) -
                game_level_time_varying_survival_df_player['birth_date']).dt.days / 365.25

            # drop the birthYear, birthMonth, birthDay, birth_date columns since they are not needed after age is computed
            game_level_time_varying_survival_df_player.drop(
                columns=['birthYear', 'birthMonth', 'birthDay', 'birth_date'], inplace=True)

            # add the time-varying survival dataframe for player to the time-varying survival dataframe for season
            survival_df_list.append(game_level_time_varying_survival_df_player)

        survival_df_season_time_varying_game_level = pd.concat(survival_df_list, axis=0)

        # removes the null values
        survival_df_season_time_varying_game_level.dropna(inplace=True)

        # save the time-varying survival dataframes to csv file
        survival_df_season_time_varying_game_level.to_csv(output_file,index=False)

        # get the unique instances (pitcher and injury combo) for the time-varying game-level survival dataframe
        pitcher_injury_combo = survival_df_season_time_varying_game_level[['player_name', 'recurrence']].drop_duplicates()
        num_instances = pitcher_injury_combo.shape[0]

        print(f"For time-varying case, there are {num_instances} instances")

if __name__=="__main__":
    # create the survival dataframe for each season from 2021 to 2024 seasons
    create_time_varying_survival_dataframe_seasonal_game_level("2021")
    create_time_varying_survival_dataframe_seasonal_game_level("2022")
    create_time_varying_survival_dataframe_seasonal_game_level("2023")
    create_time_varying_survival_dataframe_seasonal_game_level("2024")

