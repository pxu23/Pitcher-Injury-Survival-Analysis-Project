"""
    File to acquire and preprocess the injury data from Prosports, include getting the dates of relinquishment
    for each of the pitchers
"""
import datetime
import pandas as pd
import numpy as np

def get_injury_dataframe(year):
    """
        Get the dataframe for the pitcher injuries for a specific year
        :param year: the year of the followup period
        :return: the dataframe corresponding to the pitcher injuries in the followup period
    """
    # read in the preprocessed injury data from prosports baseball
    injury_data_prosports = pd.read_csv(f"../Injury_Data/"
                                        f"prosports_transactions_injuries_{year}_preprocessed.csv")

    # remove all the rows with nan in Relinquished column
    injury_data_prosports.dropna(inplace=True)

    # get all the players injured (i.e. relinquished) in the season
    injuries_df = injury_data_prosports[
        (injury_data_prosports['Date'] >= f"{year}-01-01")
        & (injury_data_prosports["Date"] <= f"{year}-12-31")]

    # convert injury dates to datetime
    injuries_df['Date'] = injuries_df['Date'].apply(convert_date_string_to_datetime)

    # rename the Relinquished column to player_name to help when joining with the statcast dataframe
    injuries_df.rename(columns={"Relinquished": "player_name"}, inplace=True)

    return injuries_df

def get_injury_df_multiple_season(start_season_year, end_season_year):
    """
        Get the dataframe for the pitcher injuries for multiple seasons
        :param start_season_year: the start year of the followup period
        :param end_season_year: the end year of the followup period
        :return: the dataframe corresponding to the pitcher injuries in the followup period
    """
    # read in the preprocessed injury data from prosports baseball
    injury_data_prosports = pd.read_csv(f"../Injury_Data/"
                                        f"prosports_transactions_injuries_{start_season_year}_{end_season_year}_preprocessed.csv")

    # remove all the rows with nan in Relinquished column
    injury_data_prosports.dropna(inplace=True)

    # get all the players injured (i.e. relinquished) in the season
    injuries_df = injury_data_prosports[
        (injury_data_prosports['Date'] >= f"{start_season_year}-01-01")
        & (injury_data_prosports["Date"] <= f"{end_season_year}-12-31")]

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
    injury_df = pd.read_csv(f"../Injury_Data/"
                            f"prosports_transactions_injuries_{season_year}_preprocessed.csv")

    # the players who are injured in prosports (note that some entries has NaN in the Relinquished, these correspond to
    # acquired)
    injured_players_prosports = injury_df['Relinquished'].dropna().unique()

    # get the injured players in Statcast (name in both Statcast and prosports)
    injured_players = np.intersect1d(injured_players_prosports, statcast_players)

    return injured_players

def get_injured_players_multiple_season(start_season_year, end_season_year, statcast_players):
    """
            Get the injured players from the injury dataframe for the multiple season
            :param start_season_year: the start season for which injured pitchers are obtained
            :param end_season_year: the end season for which injured pitchers are obtained
            :return: the list of injured players
        """
    # read in the injury data from prosports baseball
    # injury_df = get_injury_dataframe(season_year)
    injury_df = pd.read_csv(f"../Injury_Data/"
                            f"prosports_transactions_injuries_{start_season_year}_{end_season_year}_preprocessed.csv")

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

def get_all_pitchers_multiple_season(start_season_year, end_season_year):
    """
        Get all the pitchers who played for multiple seasons according to the season year
        :param start_season_year: the start season
        :param end_season_year: the end season
    """
    # get the Statcast data for multiple seasons
    statcast_df_multiple_season = pd.read_csv(f"../../Statcast_Data/statcast_data_{start_season_year}_{end_season_year}.csv")

    # get all pitchers from player_name in Statcast data
    players = statcast_df_multiple_season["player_name"].dropna().unique()

    return players

def get_healthy_pitchers(season_year, all_pitchers, injured_pitchers):
    """
        Get the healthy pitchers for the season (those are censored)
        :param season_year: the season year
        :return: the list of healthy pitchers
    """
    # healthy pitchers are not in the injured pitcher list but in Statcast
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
    # split the date by -
    splitted_date = date_string.split("-")
    # get the year, month, and day
    year = int(splitted_date[0])
    month = int(splitted_date[1])
    day = int(splitted_date[2])
    
    new_date = datetime.date(year, month, day)
    return new_date
