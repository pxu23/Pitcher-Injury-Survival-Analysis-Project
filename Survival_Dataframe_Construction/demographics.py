import pandas as pd
from tqdm import tqdm

def get_demographic_information(player):
    """ 
        Acquires the demographic information for the player. These includes weight, height, as well as batting and throwing hands.
        :param player: the name of the player to get demographic information
        :return demographic_feature_player_dict: the demographic features of the player
    """
    # read in the Lahman database for the player
    people_df = pd.read_csv("../Lahman_Demographic_Data/People.csv",
                            encoding="latin-1")

    # get the first and last name for the player
    splitted_name = player.split(",")
    last_name = splitted_name[0].strip()
    first_name = splitted_name[1].strip()

    # entry in the demographic database for the player
    demographic_feature_player = people_df.loc[(people_df['nameFirst'] == first_name) &
                                           (people_df['nameLast'] == last_name)]

    # gets the weight, height, batting hand, throwing hand, birth year, birth month, and birth date
    # for the pitcher
    demographic_feature_player_dict = demographic_feature_player[['weight', 'height', 'bats',
                                                                  'throws', 'birthYear',
                                                                  'birthMonth', 'birthDay']]
    # if not unique record (no record or multiple records), skip
    if demographic_feature_player_dict.shape[0] != 1:
        return None

    # get the player name column for later joins with pitch data
    demographic_feature_player_dict["player_name"] = player

    return demographic_feature_player_dict

def get_players_with_demographic_information(player_list):
    """
        Get the list of injured players with demographic information in the Sean Lahman database
        :param injured_player_list:the list of injured players
    """
    player_with_demographic_information_list = []

    # loop through the list of player
    for player in tqdm(player_list):
        # gets the demographic features for the player
        demographic_feature_player = get_demographic_information(player)

        # unique record in the Lahman databaes
        if demographic_feature_player is not None:
            # add that player to the list of players with demographic information
            player_with_demographic_information_list.append(player)

    return player_with_demographic_information_list

if __name__ == "__main__":
    # TEST: get the demographic information of Snell, Blake
    player = "Snell, Blake"
    print(get_demographic_information(player))
