# get the injured and healthy pitchers for the season
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from injury_data_utils import get_all_pitchers, get_healthy_pitchers, get_injured_players
from demographics import get_players_with_demographic_information

for season in [2021, 2022, 2023, 2024, 2025]:
    # get all the Statcast pitchers for the season
    all_pitchers_season = get_all_pitchers(season)

    # get all the injured pitchers for the season
    injured_pitchers_season = get_injured_players(season, all_pitchers_season)

    # get all the healthy pitchers for the season
    healthy_pitchers_season = get_healthy_pitchers(season, all_pitchers_season, injured_pitchers_season)

    print(f"There are {len(injured_pitchers_season)} injured players in {season}")
    print(f"There are {len(healthy_pitchers_season)} healthy players in {season}")
    print(f"There are {len(all_pitchers_season)} players in {season}")

    if not os.path.exists(f"../Pitchers_Season/healthy_pitchers_{season}_with_demographic_info.txt"):
        # get the healthy pitchers in season with demographic information
        healthy_pitchers_season_with_demographic_info = get_players_with_demographic_information(healthy_pitchers_season)
        # save the results to txt file
        np.savetxt(f"../Pitchers_Season/healthy_pitchers_{season}_with_demographic_info.txt",
                   healthy_pitchers_season_with_demographic_info,
                   fmt="%s")

    if not os.path.exists(f"../Pitchers_Season/injured_pitchers_{season}_with_demographic_info.txt"):
        # get the injured pitchers in season with demographic information
        injured_pitchers_season_with_demographic_info = get_players_with_demographic_information(injured_pitchers_season)
        # save the results to txt file
        np.savetxt(f"../Pitchers_Season/injured_pitchers_{season}_with_demographic_info.txt",
                   injured_pitchers_season_with_demographic_info,
                   fmt="%s")

    # get the injured pitchers for season with demographic information from reading txt file
    injured_pitchers_season_with_demographic_info = np.genfromtxt(f"../Pitchers_Season/injured_pitchers_{season}_with_demographic_info.txt",
                                                                  dtype=str,
                                                                  delimiter="\n")
    print(f"There are {len(injured_pitchers_season_with_demographic_info)} injured pitchers with demographic info in {season}")

    # get the healthy pitchers for season with demographic information from reading txt file
    healthy_pitchers_season_with_demographic_info = np.genfromtxt(f"../Pitchers_Season/healthy_pitchers_{season}_with_demographic_info.txt",
                                                                  dtype=str,
                                                                  delimiter="\n")
    print(f"There are {len(healthy_pitchers_season_with_demographic_info)} healthy pitchers with demographic info in {season}")