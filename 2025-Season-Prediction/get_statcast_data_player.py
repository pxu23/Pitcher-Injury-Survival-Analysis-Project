import pandas as pd
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
import os

# top MLB pitchers for 2025 season
player_names = [('skubal', 'tarik'),('wheeler', 'zack'), ("sale","chris"),
                ("skenes","paul"), ('fried', 'max'), ('ragans', 'cole'),
                ('webb', 'logan'), ('snell', 'blake'), ('valdez', 'framber'),
                ('strider', 'spencer'), ('degrom', 'jacob'), ('yamamoto', 'yoshinobu'),
                ('cole', 'gerrit'), ('burnes', 'corbin'),
                ('gilbert','logan'), ('flaherty', 'jack'),
                ('gausman', 'kevin'),  ('povich', 'cade'), ('alcantara', 'sandy'),
                ('lopez', 'pablo'), ('mcclanahan', 'shane')]

pitcher_num_games_pitches = []

for (last_name, first_name) in player_names:
    output_file = f"Pitch_Data_2025_Pitchers/{first_name}_{last_name}_pitches.csv"
    if not os.path.exists(output_file):
        lookup = playerid_lookup(last_name, first_name)

        # no Statcast data can be found for the player
        if len(lookup) == 0:
            print(f"{last_name}:{first_name} not found")
            continue
        # get the player id
        id = lookup["key_mlbam"].item()
        # get the pitches for the player on Statcast
        pitcher_df = statcast_pitcher('2013-01-01', '2025-06-30', id)
        pitcher_df.to_csv(output_file, index=False)

    pitcher_df = pd.read_csv(output_file)
    num_pitches = pitcher_df.shape[0]
    num_games = len(pitcher_df['game_date'].unique())


    # Add data for each row as dictionaries
    pitcher_num_games_pitches.append({'Name': f"{last_name}, {first_name}", 'num_games': num_games, 'num_pitches': num_pitches})

    print(f"For {first_name} {last_name} there are {num_pitches} pitches in total.")
    print(f"For {first_name} {last_name} there are {num_games} games in total.")

# Create the DataFrame from the list of dictionaries
pitcher_num_games_pitches_df = pd.DataFrame(pitcher_num_games_pitches)
pitcher_num_games_pitches_df.to_csv(f"top_mlb_pitchers_2025_num_pitches_games.csv", index=False)