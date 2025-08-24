import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

def prepare_time_varying_pitcher_input_for_rnn(pitcher_name):
        """
            Prepare the inputs (features, times, and events) for the RNN for pitcher
            Creates the features (X), Event Times (T), and Event Indicators (E)
            :param pitcher_name: the player name that the RNN model is making a prediction for
            :return: the features, times, and events in 2025 season for the RNN model for that pitcher
        """

        # STEP 1: Read in the time varying game level survival dataframe for the 2025 season for the pitcher
        survival_df_time_varying_game_level_pitcher_2025 = pd.read_csv(f"Survival_Dataframe_Time_Varying_2025_Pitchers/"
                                                                       f"survival_df_time_varying_game_level_2025_{pitcher_name}_processed.csv")

        # X - features (workload and pitcher demographics) before each injury recurrence
        # T - survival times (number of games) before each injury recurrence
        # E - events (whether an injury occurred)
        X, T, E = [], [], []

        # STEP 2: get the injury occurrences for the pitcher
        injury_recurrence_pitcher = survival_df_time_varying_game_level_pitcher_2025[['recurrence']].drop_duplicates()

        for idx, row in injury_recurrence_pitcher.iterrows():
            recurrence = row['recurrence']

            # STEP 3: get the corresponding game rows in the survival dataframe for the pitcher for that injury occurrence
            survival_df_time_varying_recurrence_pitcher = survival_df_time_varying_game_level_pitcher_2025[
                survival_df_time_varying_game_level_pitcher_2025['recurrence'] == recurrence
                ]

            # STEP 4: Get the corresponding workload and demographic features for these rows
            # drop the 10th and 90th percentiles since we are not focusing on them at least for now
            features = survival_df_time_varying_recurrence_pitcher.drop(columns=['player_name', 'game_date',
                                                                                 'start', 'stop', 'event',
                                                                                 'Batting Hand', 'Throwing Hand',
                                                                                 'release_spin_rate_10_percentile',
                                                                                 'release_spin_rate_90_percentile',
                                                                                 'release_speed_10_percentile',
                                                                                 'release_speed_90_percentile',
                                                                                 'effective_speed_10_percentile',
                                                                                 'effective_speed_90_percentile',
                                                                                 'vx0_10_percentile',
                                                                                 'vx0_90_percentile',
                                                                                 'vy0_10_percentile',
                                                                                 'vy0_90_percentile',
                                                                                 'vz0_10_percentile',
                                                                                 'vz0_90_percentile']).to_numpy()
            # STEP 5: Get the number of games played before injury occurrence
            num_games_player_injury = survival_df_time_varying_recurrence_pitcher.shape[0]

            X.append(features)
            T.append(num_games_player_injury)

            # STEP 6: Get whether the injury actually occurred
            event = int(survival_df_time_varying_recurrence_pitcher['event'].iloc[-1])
            E.append(event)

        # STEP 7: Pad the sequence of variable length sequence of features with zeros
        X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
        X_padded = pad_sequence(X, batch_first=True, padding_value=0.0)

        # STEP 8: Convert the event and games array to tensor
        E = torch.tensor(E, dtype=torch.float32)
        T = torch.tensor(T)

        return X_padded, T, E