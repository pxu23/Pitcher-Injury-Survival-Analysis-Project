import pandas as pd

def get_demographic_information_2025_pitcher(first_name, last_name):
    """
        Gets the demographic information for the top 2025 Major League pitchers
        :param first_name: The first name of the person
        :param last_name: The last name of the person
    """

    # read in the file corresponding to the demographic information for the 2025 pitchers
    individual_2025_pitchers = pd.read_excel("Individual_Pitchers_2025.xlsx")

    player_name = last_name + ", " +  first_name

    # entry in the demographic database for the player
    demographic_feature_player = individual_2025_pitchers[(individual_2025_pitchers['nameFirst'] == first_name) &
                                                          (individual_2025_pitchers['nameLast'] == last_name)]

    # the demographic features for the person as a Pandas dataframe
    demographic_feature_player_df = pd.DataFrame.from_dict({
                                                  'player_name':[player_name],
                                                  'Weight': [demographic_feature_player['Weight'].item()],
                                                  'Height': [demographic_feature_player['Height'].item()],
                                                  'Batting Hand': [demographic_feature_player['Batting Hand'].item()],
                                                  'Throwing Hand': [demographic_feature_player['Throwing Hand'].item()],
                                                  "Age": [demographic_feature_player["Age"].item()]})
    return demographic_feature_player_df