from pybaseball import statcast
import os
import warnings

""" 
    This file is used to acquire the Statcast data for each of the recent seasons
    we used in our work. This can also be used to acquire the Statcast data for multiple seasons.
"""
def acquire_statcast_data(season_year):
    """
        Acquires the statcast pitching data for the season.
        @param season_year: the season year for which the Statcast pitching data is acquired.
    """
    warnings.filterwarnings("ignore")
    statcast_pitching_data = statcast(start_dt=f"{season_year}-01-01", end_dt= f"{season_year}-12-31", verbose=False)
    statcast_pitching_data.to_csv(f"Statcast_Data/statcast_data_{season_year}.csv",index=False)

def acquire_statcast_data_in_range(start_season_year, end_season_year):
    """
        Acquire Statcast data for multiple seasons.
        @param: start_season_year the start year to acquire data
        @param: end_season_year the end year to acquire data
    """
    warnings.filterwarnings("ignore")
    statcast_pitching_data = statcast(start_dt=f"{start_season_year}-01-01", end_dt= f"{end_season_year}-12-31", verbose=False)
    statcast_pitching_data.to_csv(f"Statcast_Data/statcast_data_{start_season_year}_{end_season_year}.csv",index=False)

if __name__=="__main__":
    # if the pitching data is not already present, acquire them
    #if not os.path.exists("Statcast_Data/statcast_data_2024.csv"):
        # 2024 data
    #    acquire_statcast_data(2024)

    #if not os.path.exists("Statcast_Data/statcast_data_2023.csv"):
        # 2023 data
    #    acquire_statcast_data(2023)

    #if not os.path.exists("Statcast_Data/statcast_data_2022.csv"):
        # 2022 data
    #    acquire_statcast_data(2022)

    #if not os.path.exists("Statcast_Data/statcast_data_2021.csv"):
        # 2021 data
    #    acquire_statcast_data(2021)

    if not os.path.exists("Statcast_Data/statcast_data_2020.csv"):
        # 2020 data
        acquire_statcast_data(2020)

    if not os.path.exists("Statcast_Data/statcast_data_2019.csv"):
        # 2019 data
        acquire_statcast_data(2019)

    if not os.path.exists("Statcast_Data/statcast_data_2018.csv"):
        # 2018 data
        acquire_statcast_data(2018)

    #if not os.path.exists("Statcast_Data/statcast_data_2018_2023.csv"):
        # 2018 - 2023 data
    #    acquire_statcast_data_in_range(2018, 2023)

    #if not os.path.exists("Statcast_Data/statcast_data_2022_2023.csv"):
        # 2022 - 2023 Statcast data
    #    acquire_statcast_data_in_range(2022, 2023)

    
    