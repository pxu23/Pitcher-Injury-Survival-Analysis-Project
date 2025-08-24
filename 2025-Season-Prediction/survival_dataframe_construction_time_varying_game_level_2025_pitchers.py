import warnings
import os
import datetime
import numpy as np
import pandas as pd
# the time varying nature of the games between the two injury dates for a pitcher in the followup period
warnings.filterwarnings('ignore')
from demographics_2025_pitchers import get_demographic_information_2025_pitcher

def convert_injury_date_string_to_datetime(injury_date):
    splitted_date = injury_date.split('/')
    month = int(splitted_date[0])
    day = int(splitted_date[1])
    year = int(splitted_date[2])

    return datetime.date(year, month, day)

def create_time_varying_survival_dataframe_2025_game_level_for_pitcher(first_name, last_name):
    lower_first_name = first_name.lower()
    lower_last_name = last_name.lower()

    # read in the Statcast data corresponding ot the player
    game_level_statcast_data = pd.read_csv(f"Game_Level_Pitch_Data_2025_Pitchers/"
                                           f"avg_pitch_characteristics_game_level_{lower_first_name}_{lower_last_name}.csv")


    game_level_statcast_data_2025 = game_level_statcast_data[game_level_statcast_data['game_date'] >= '2025-01-01']

    game_level_statcast_data_2025['game_date'] = game_level_statcast_data_2025['game_date'].apply(convert_injury_date_string_to_datetime)

    # get the injury dataframe corresponding to the 2025 pitcher
    injury_df_file = f"Injury_Data_2025_Pitchers/injury_data_{lower_first_name}_{lower_last_name}_preprocessed.xlsx"
    if not os.path.exists(injury_df_file):
        injury_dates_player_2025 = []
        injury_dates_player_2025_augmented = np.array([datetime.date(2025, 1, 1), datetime.date(2025, 12, 31)])
    else:
        injury_df_player = pd.read_excel(f"Injury_Data_2025_Pitchers/"
                                         f"injury_data_{lower_first_name}_{lower_last_name}_preprocessed.xlsx")
        injury_df_player["Date"] = injury_df_player["Date"].apply(convert_injury_date_string_to_datetime)

        # get the injury dates in 2025 for the pitcher
        injury_dates_player_2025 = injury_df_player[injury_df_player['Date'] >= datetime.date(2025, 1, 1)]['Date'].to_numpy()

        injury_dates_player_2025_augmented = np.insert(injury_dates_player_2025,0, datetime.date(2025, 1, 1))
        injury_dates_player_2025_augmented = np.append(injury_dates_player_2025_augmented,datetime.date(2025, 12, 31))

    output_file = (f"Survival_Dataframe_Time_Varying_2025_Pitchers/"
                   f"survival_df_time_varying_game_level_2025_{lower_first_name}_{lower_last_name}_processed.csv")

    if not os.path.exists(output_file):
        survival_df_time_varying_pitcher_list = list()

        num_previous_injury = 0
        for i in range(len(injury_dates_player_2025_augmented) - 1):
                first_injury_date = (injury_dates_player_2025_augmented[i])
                second_injury_date = (injury_dates_player_2025_augmented[i + 1])

                # get the games between two injury dates
                games_between_injury_player = game_level_statcast_data_2025[
                    (game_level_statcast_data_2025['game_date'] > first_injury_date) &
                    (game_level_statcast_data_2025['game_date'] < second_injury_date)]

                # Compute the mean of each numerical column
                mean_values = games_between_injury_player.select_dtypes(include='number').mean()

                # Fill missing values in numerical columns with the mean
                games_between_injury_player.fillna(mean_values, inplace=True)

                # get the number of games between two injury dates
                num_games_between_injury = games_between_injury_player.shape[0]
                if num_games_between_injury == 0:
                    num_previous_injury += 1
                    continue

                game_level_time_varying_survival_df_between_injury = games_between_injury_player
                game_level_time_varying_survival_df_between_injury["recurrence"] = num_previous_injury

                game_level_time_varying_survival_df_between_injury["start"] = np.arange(0, num_games_between_injury)
                game_level_time_varying_survival_df_between_injury["stop"] = np.arange(1, num_games_between_injury + 1)
                status = np.zeros(num_games_between_injury)

                if num_previous_injury < len(injury_dates_player_2025):
                    status[-1] = 1
                # did the pitcher experience the event after each game
                game_level_time_varying_survival_df_between_injury["event"] = status

                survival_df_time_varying_pitcher_list.append(game_level_time_varying_survival_df_between_injury)
                num_previous_injury += 1


        survival_df_player_time_varying_game_level = pd.concat(survival_df_time_varying_pitcher_list, axis=0)
        # get the demographic information for the player
        demographic_information_player = get_demographic_information_2025_pitcher(first_name, last_name)

        survival_df_player_time_varying_game_level = pd.merge(survival_df_player_time_varying_game_level,demographic_information_player,on="player_name")

        # remove all instances where at least one covariate is NaN (now let's not preprocess it)
        survival_df_player_time_varying_game_level.dropna(inplace=True)

        # convert the covariates for the bats and throws to categorical codes
        survival_df_player_time_varying_game_level['Batting Hand'] = pd.Categorical(survival_df_player_time_varying_game_level['Batting Hand']).codes
        survival_df_player_time_varying_game_level['Throwing Hand'] = pd.Categorical(survival_df_player_time_varying_game_level['Throwing Hand']).codes

        survival_df_player_time_varying_game_level.to_csv(output_file,index=False)



if __name__=="__main__":
    pitcher_list = [("Blake", "Snell"), ("Cade", "Povich"), ("Chris", "Sale"), ("Cole", "Ragans"),
                    ("Corbin", "Burnes"), ("Framber", "Valdez"), ("Gerrit", "Cole"), ("Jack", "Flaherty"),
                    ("Jacob", "deGrom"), ("Logan", "Gilbert"), ("Logan", "Webb"),
                    ("Max", "Fried"), ("Spencer", "Strider"), ("Tarik", "Skubal"),
                    ("Yoshinobu", "Yamamoto"), ("Zack", "Wheeler"), ("Paul", "Skenes"),
                    ("Kevin", "Gausman"), ("Shane", "McClanahan"), ("Pablo", "Lopez")]
    for first_name, last_name in pitcher_list:
        create_time_varying_survival_dataframe_2025_game_level_for_pitcher(first_name, last_name)
