from pybaseball import statcast
import os
import warnings

""" This file is used to acquire the Statcast data for each of the years
    Run this file to acquire the statcast data for each of the 2020-2024 season,
    the 2018-2023 season, and the 2022-2023 season used in our work. 
"""
def acquire_statcast_data(year):
    """
        Acquires the statcast pitching data for the pitchers starting and ending at two different years.
        @param: start
    """
    warnings.filterwarnings("ignore")
    statcast_pitching_data = statcast(start_dt=f"{year}-01-01", end_dt= f"{year}-12-31", verbose=False)
    statcast_pitching_data.to_csv(f"Statcast_Data/statcast_data_{year}.csv",index=False)

def acquire_statcast_data_in_range(start_year, end_year):
    """
        Acquire Statcast data in range from start year to end year.
        @param: start_year the start year to acquire data
        @param: end_year the end year to acquire data
    """
    warnings.filterwarnings("ignore")
    statcast_pitching_data = statcast(start_dt=f"{start_year}-01-01", end_dt= f"{end_year}-12-31", verbose=False)
    statcast_pitching_data.to_csv(f"Statcast_Data/statcast_data_{start_year}_{end_year}.csv",index=False)

if __name__=="__main__":
    # if the file is not present, acquire the data

    if not os.path.exists("Statcast_Data/statcast_data_2024.csv"):
        # 2024 data
        acquire_statcast_data(2024)

    if not os.path.exists("Statcast_Data/statcast_data_2023.csv"):
        # 2023 data
        acquire_statcast_data(2023)

    if not os.path.exists("Statcast_Data/statcast_data_2022.csv"):
        # 2022 data
        acquire_statcast_data(2022)

    if not os.path.exists("Statcast_Data/statcast_data_2021.csv"):
        # 2021 data
        acquire_statcast_data(2021)

    if not os.path.exists("Statcast_Data/statcast_data_2020.csv"):
        # 2020 data
        acquire_statcast_data(2020)

    if not os.path.exists("Statcast_Data/statcast_data_2018_2023.csv"):
        # 2018 - 2023 data
        acquire_statcast_data_in_range(2018, 2023)

    if not os.path.exists("Statcast_Data/statcast_data_2022_2023.csv"):
        # 2022 - 2023 Statcast data
        acquire_statcast_data_in_range(2022, 2023)

    
    