"""
    File to acquire and preprocess the injury data from Prosports, include getting the dates of relinquishment
    for each of the pitchers
"""
import datetime
import pandas as pd
import numpy as np

def preprocess_name(name):
    """
        Preprocess the name of the injured pitcher which includes removing the items in the () and
        rewritten it in the form of Last Name, First Name
        :param name: the name of the injured pitcher to process
        :return full_name: the full preprocessed name
    """
    splitted_name = name.split()

    # sometimes only the first or last name is shown
    if len(splitted_name) == 1:
        return name

    # get the first and last name
    first_name = splitted_name[0]
    last_name = splitted_name[1]

    # get the full name
    full_name = last_name + ", " + first_name

    # remove the ( and ) in full name
    full_name.replace("(", "")
    full_name.replace(")", "")
    return full_name


def preprocess_name_column(injury_file, output_file):
    """
        Preprocess the name column of the injury dataframe (i.e. relinquished column)
        :param injury_file: the name of the injury file to preprocess
        :param output_file: the name of the output file in which the name column is preprocessed
    """
    # read in the injury data from prosports baseball
    injury_data_prosports = pd.read_csv(injury_file)

    # drop the Team, Acquired, and Notes columns since we focus on the relinquished players
    injury_data_prosports.drop(['Team', 'Acquired', 'Notes'], axis=1, inplace=True)

    # remove all the rows with nan in Relinquished column
    injury_data_prosports.dropna(inplace=True)

    # preprocess the name to match the format in the statcast data (Last Name, Firest Name)
    injury_data_prosports['Relinquished'] = injury_data_prosports['Relinquished'].apply(preprocess_name)

    # remove all the rows with nan in Relinquished column
    injury_data_prosports.dropna(inplace=True)

    # save the preprocessed injury dataframe to csv file
    injury_data_prosports.to_csv(output_file, index=False)

def get_injury_dataframe(year):
    """
        Get the dataframe for the pitcher injuries for a specific year
        :param year: the year of the followup period
        :return: the dataframe corresponding to the pitcher injuries in the followup period
    """
    # read in the preprocessed injury data from prosports baseball
    injury_data_prosports = pd.read_csv(f"../Injury_Data/prosports_transactions_injuries_{year}_preprocessed.csv")

    # remove all the rows with nan in Relinquished column
    injury_data_prosports.dropna(inplace=True)

    # get all the players injuried (i.e. relinquished) in the season
    injuries_df = injury_data_prosports[
        (injury_data_prosports['Date'] >= f"{year}-01-01")
        & (injury_data_prosports["Date"] <= f"{year}-12-31")]

    # convert injury dates to datetime
    injuries_df['Date'] = injuries_df['Date'].apply(convert_date_string_to_datetime)

    # rename the Relinquished column to player_name to help when joining with the statcast dataframe
    injuries_df.rename(columns={"Relinquished": "player_name"}, inplace=True)

    return injuries_df

def get_injured_players(season_year, statcast_players):
    """
        Get the injured players from the injury dataframe for the specific followup period
        :param season_year: the season for which injured pitchers are obtained
        :return: the list of injured players
    """
    # read in the injury data from prosports baseball
    #injury_df = get_injury_dataframe(season_year)
    injury_df = pd.read_csv(f"../Injury_Data/prosports_transactions_injuries_{season_year}_preprocessed.csv")

    # the players who are injured in prosports (note that some entries has NaN in the Relinquished, these correspond to
    # acquired)
    injured_players_prosports = injury_df['Relinquished'].dropna().unique()

    # get the injured players in Statcast (name in both Statcast and prosports)
    injured_players = np.intersect1d(injured_players_prosports, statcast_players)

    return injured_players

def get_all_pitchers(season_year):
    """
        Get all the pitchers for the season according to the season_year
        :param season_year: the season year
    """
    # get the statcast data for the season
    statcast_df_season = pd.read_csv(f"../../Statcast_Data/statcast_data_{season_year}.csv")

    # get all the pitchers from player_name in the Statcast date
    players = statcast_df_season["player_name"].dropna().unique()

    return players

def get_healthy_pitchers(season_year, all_pitchers, injured_pitchers):
    """
        Get the healthy pitchers for the season (those are censored)
        :param season_year: the season year
        :return: the list of healthy pitchers
    """
    # healthy pitchers are not in the injured pitcher list
    healthy_pitchers = np.setdiff1d(all_pitchers, injured_pitchers)
    return healthy_pitchers

def get_all_injury_dates(injury_df, player_name):
    """
        Get all the injury dates of a pitcher in the injury dataframe

        :param injury_df: the injury dataframe for the followup period
        :param player_name: the name of the player
        :return: the injury dates for the player in ascending order
    """
    # the rows in the injury dataframe corresponding to player
    injury_df_player = injury_df[injury_df['player_name'] == player_name]

    # sort by ascending date for the relinquishment
    injury_df_player.sort_values(by="Date")
    return injury_df_player['Date'].to_numpy()

def convert_date_string_to_datetime(date_string):
    """
        Convert the date from string object to a datetime object
        :param date_string: the date string
        :return: the date as a datetime object
    """
    splitted_date = date_string.split("-")
    year = int(splitted_date[0])
    month = int(splitted_date[1])
    day = int(splitted_date[2])

    new_date = datetime.date(year, month, day)
    return new_date
