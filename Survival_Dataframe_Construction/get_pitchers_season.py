# get the injured and healthy pitchers for the season
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
import numpy as np

from injury_data_utils import get_all_pitchers, get_healthy_pitchers, get_injured_players
from demographics import get_players_with_demographic_information

season = "2024"
all_pitchers_season = get_all_pitchers(season)
injured_pitchers_season = get_injured_players(season, all_pitchers_season)
healthy_pitchers_season = get_healthy_pitchers(season, all_pitchers_season, injured_pitchers_season)

print(f"There are {len(injured_pitchers_season)} injuried players in {season}")
print(f"There are {len(healthy_pitchers_season)} healthy players in {season}")
print(f"There are {len(all_pitchers_season)} players in {season}")

if not os.path.exists(f"../Pitchers_Season/healthy_pitchers_{season}_with_demographic_info.txt"):
    healthy_pitchers_season_with_demographic_info = get_players_with_demographic_information(healthy_pitchers_season)
    np.savetxt(f"../Pitchers_Season/healthy_pitchers_{season}_with_demographic_info.txt", healthy_pitchers_season_with_demographic_info,
               fmt="%s")

if not os.path.exists(f"../Pitchers_Season/injured_pitchers_{season}_with_demographic_info.txt"):
    injured_pitchers_season_with_demographic_info = get_players_with_demographic_information(injured_pitchers_season)
    np.savetxt(f"../Pitchers_Season/injured_pitchers_{season}_with_demographic_info.txt", injured_pitchers_season_with_demographic_info,
               fmt="%s")

injured_pitchers_season_with_demographic_info = np.genfromtxt(f"../Pitchers_Season/injured_pitchers_{season}_with_demographic_info.txt",dtype=str,
                                                              delimiter="\n")
print(f"There are {len(injured_pitchers_season_with_demographic_info)} injured pitchers with demographic info in {season}")
healthy_pitchers_season_with_demographic_info = np.genfromtxt(f"../Pitchers_Season/healthy_pitchers_{season}_with_demographic_info.txt", dtype=str,
                                                              delimiter="\n")
print(f"There are {len(healthy_pitchers_season_with_demographic_info)} healthy pitchers with demographic info in {season}")

lahman_data = pd.read_csv("../Lahman_Demographic_Data/People.csv", encoding="latin-1")
lahman_data['player_name'] = lahman_data['nameLast'] + ", " + lahman_data['nameFirst']
lahman_data_statcast_pitchers = lahman_data[lahman_data["player_name"].isin(healthy_pitchers_season) |
                                            lahman_data["player_name"].isin(injured_pitchers_season)]
print(lahman_data_statcast_pitchers.isnull().sum())
lahman_data_statcast_pitchers_alive = lahman_data_statcast_pitchers[lahman_data_statcast_pitchers['deathYear'].isnull()
& lahman_data_statcast_pitchers['deathMonth'].isnull() & lahman_data_statcast_pitchers['deathDay'].isnull()
& lahman_data_statcast_pitchers['deathCity'].isnull() & lahman_data_statcast_pitchers['deathState'].isnull()
& lahman_data_statcast_pitchers['deathCountry'].isnull()]
print(lahman_data_statcast_pitchers)
print(lahman_data_statcast_pitchers_alive)

alive_healthy_pitchers_with_demographic_info = lahman_data_statcast_pitchers_alive[lahman_data_statcast_pitchers_alive['player_name'].isin(
    healthy_pitchers_season
)]
print(alive_healthy_pitchers_with_demographic_info)