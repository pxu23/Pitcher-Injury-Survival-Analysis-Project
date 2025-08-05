import datetime

import pandas as pd

def get_demographic_information_2025_pitcher(first_name, last_name):
    individual_2025_pitchers = pd.read_excel("Individual_Pitchers_2025.xlsx")
    player_name = last_name + ", " +  first_name
    # entry in the demographic database for the player
    demographic_feature_player = individual_2025_pitchers[(individual_2025_pitchers['nameFirst'] == first_name) &
                                                          (individual_2025_pitchers['nameLast'] == last_name)]
    demographic_feature_player_df = pd.DataFrame.from_dict({
                                                  'player_name':[player_name],
                                                  'Weight': [demographic_feature_player['Weight'].item()],
                                                  'Height': [demographic_feature_player['Height'].item()],
                                                  'Batting Hand': [demographic_feature_player['Batting Hand'].item()],
                                                  'Throwing Hand': [demographic_feature_player['Throwing Hand'].item()],
                                                  "Age": [demographic_feature_player["Age"].item()]})
    return demographic_feature_player_df

if __name__ == '__main__':
    first_name = "blake"
    last_name = "snell"

    print(get_demographic_information_2025_pitcher(first_name, last_name))