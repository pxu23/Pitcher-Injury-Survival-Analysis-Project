from pybaseball import statcast, playerid_lookup, statcast_pitcher
import warnings
warnings.filterwarnings("ignore")

lookup = playerid_lookup("lopez", "pablo")
print(lookup)

# get the player id
id = lookup["key_mlbam"].item()

# get the pitches for the player on Statcast
pitcher_df = statcast_pitcher('2013-01-01', '2025-06-30', id)

print(pitcher_df)