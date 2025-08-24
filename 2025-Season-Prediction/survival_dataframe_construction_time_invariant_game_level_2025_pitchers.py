import pandas as pd
import tqdm

def create_time_invariant_survival_dataframe(first_name, last_name):
    lower_first_name = first_name.lower()
    lower_last_name = last_name.lower()
    survival_df_time_varying_game_level_processed = pd.read_csv(f"Survival_Dataframe_Time_Varying_2025_Pitchers/"
                                                                f"survival_df_time_varying_game_level_2025_{lower_first_name}_{lower_last_name}_processed.csv")

    # pitcher injury recurrence
    pitcher_injury_recurrence = survival_df_time_varying_game_level_processed[['player_name', 'recurrence']].drop_duplicates()

    rows = []
    for row in tqdm.tqdm(pitcher_injury_recurrence.iterrows()):

        # check if there exists a column where all the games are null
        time_varying_entry_for_instance = survival_df_time_varying_game_level_processed[
            survival_df_time_varying_game_level_processed['recurrence'] == row[1].recurrence
        ]

        event = time_varying_entry_for_instance['event'].iloc[-1]

        num_games = time_varying_entry_for_instance.shape[0]

        # average pitcher characteristics
        avg_pitch_release_spin_rate = time_varying_entry_for_instance['avg_release_spin_rate'].mean()
        avg_pitch_release_speed = time_varying_entry_for_instance['avg_release_speed'].mean()
        avg_effective_speed = time_varying_entry_for_instance['avg_effective_speed'].mean()
        avg_vx0 = time_varying_entry_for_instance['avg_vx0'].mean()
        avg_vy0 = time_varying_entry_for_instance['avg_vy0'].mean()
        avg_vz0 = time_varying_entry_for_instance['avg_vz0'].mean()

        # average age
        avg_age = time_varying_entry_for_instance['Age'].mean()
        # average height
        avg_height = time_varying_entry_for_instance['Height'].mean()
        # weight
        avg_weight = time_varying_entry_for_instance['Weight'].mean()
        # batting hand
        avg_bats = time_varying_entry_for_instance['Batting Hand'].mean()
        # throwing hand
        avg_throws = time_varying_entry_for_instance['Throwing Hand'].mean()

        # average number of pitches
        avg_pitches = time_varying_entry_for_instance['num_pitches'].mean()

        row_dict = {"player_name": row[1].player_name,
                    "avg_release_spin_rate": avg_pitch_release_spin_rate,
                    "avg_release_speed": avg_pitch_release_speed,
                    "avg_effective_speed": avg_effective_speed,
                    "avg_vx0": avg_vx0, "avg_vy0": avg_vy0,
                    "avg_vz0": avg_vz0, "avg_pitches": avg_pitches,
                    "age": avg_age, "Height": avg_height, "Weight": avg_weight,
                    "Batting Hand": avg_bats, "Throwing Hand": avg_throws,
                    "recurrence": row[1].recurrence,
                    "event": event,
                    "num_games": num_games}

        rows.append(row_dict)

    output_file_processed = (f"Survival_Dataframe_Time_Invariant_2025_Pitchers/"
                             f"survival_df_time_invariant_game_level_2025_{lower_first_name}_{lower_last_name}_processed.csv")

    survival_df_time_invariant_game_level = pd.DataFrame(rows)

    survival_df_time_invariant_game_level.dropna(inplace=True)

    survival_df_time_invariant_game_level.to_csv(output_file_processed, index=False)

if __name__ == '__main__':
    pitcher_list = [("Blake", "Snell"), ("Cade", "Povich"), ("Chris", "Sale"), ("Cole", "Ragans"),
                    ("Corbin", "Burnes"), ("Framber", "Valdez"), ("Gerrit", "Cole"), ("Jack", "Flaherty"),
                    ("Jacob", "deGrom"), ("Logan", "Gilbert"), ("Logan", "Webb"),
                    ("Max", "Fried"), ("Spencer", "Strider"), ("Tarik", "Skubal"),
                    ("Yoshinobu", "Yamamoto"), ("Zack", "Wheeler"), ("Paul", "Skenes"),
                    ("Kevin", "Gausman"), ("Shane", "McClanahan")]
    pitcher_list = [("Pablo", "Lopez")]
    for first_name, last_name in pitcher_list:
        create_time_invariant_survival_dataframe(first_name, last_name)
