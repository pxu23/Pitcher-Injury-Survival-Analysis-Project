import numpy as np
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def prepare_time_varying_input_for_rnn(season_year):
    """
        Prepare the inputs (features, times, and events) for the RNN for the season
        :param season_year: the season to train the RNN model
        :return: the features, times, and events for the RNN model
    """
    # read in the time-varying survival dataframes game-level

    survival_df_time_varying_game_level_season = pd.read_csv(f"../Survival-Dataframes/Time-Varying/"
                                                             f"survival_df_time_varying_game_level_{season_year}_processed.csv")
    survival_df_time_varying_game_level_season.drop(columns=[], inplace=True)

    survival_df_time_varying_game_level_season.dropna(inplace=True)

    # survival times
    T = []

    # events
    E = []

    # features
    X = []

    pitcher_injury_combo = survival_df_time_varying_game_level_season[['player_name', 'recurrence']].drop_duplicates()

    for idx, row in pitcher_injury_combo.iterrows():
        player_name = row['player_name']
        recurrence = row['recurrence']

        # get the corresponding game rows in the survival dataframe
        games_for_player_injury = survival_df_time_varying_game_level_season[
            (survival_df_time_varying_game_level_season['player_name'] == player_name) &
            (survival_df_time_varying_game_level_season['recurrence'] == recurrence)
            ]

        # features

        # likely multicollinearity
        games_features = games_for_player_injury.drop(columns=['player_name', 'game_date',
                                                               'start', 'stop', 'event', 'bats', 'throws']).to_numpy()

        num_games_player_injury = games_for_player_injury.shape[0]

        X.append(games_features)
        T.append(num_games_player_injury)

        # event
        event = int(games_for_player_injury['event'].iloc[-1])
        E.append(event)

    X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
    X_padded = pad_sequence(X, batch_first=True, padding_value=0.0)

    E = torch.tensor(E, dtype=torch.float32)
    T = torch.tensor(T)

    return X_padded, T, E

def prepare_time_invariant_input_for_rnn(season_year):
    """
        Prepare the inputs (features, times, and events) for the RNN for the season
        :param season_year: the season year to train the RNN model
        :return: the features, times, and events for the RNN model
    """
    # read in the time-varying survival dataframes game-level
    # survival_df_time_varying_game_level_season = pd.read_csv(f"survival_df_time_varying_game_level_{season_year}_processed.csv")

    # maybe read in the time-invariant survival dataframes game-level for season (to be consistent with RNN-Surv, and the baseline survival
    # models and to prevent padding which can cause data leakage)

    survival_df_time_invariant_game_level_season = pd.read_csv(f"../Survival-Analysis-Dataframes/Time-Invariant/survival_df_time_invariant_game_{season_year}_level_processed.csv")
    # get the pitcher injury recurrence combinations
    pitcher_injury_combo = survival_df_time_invariant_game_level_season[['player_name', 'recurrence']]
    pitcher_injury_combo.drop_duplicates(inplace=True)

    # get the maximum number of games (T_max)
    max_games_before_injury = survival_df_time_invariant_game_level_season['num_games'].max()

    # survival times
    T = []

    # events
    E = []

    # features
    X = []

    for idx, row in pitcher_injury_combo.iterrows():
        player_name = row['player_name']
        recurrence = row['recurrence']

        # get the corresponding game rows in the survival dataframe
        games_for_player_injury = survival_df_time_invariant_game_level_season[
            (survival_df_time_invariant_game_level_season['player_name'] == player_name) &
            (survival_df_time_invariant_game_level_season['recurrence'] == recurrence)
            ]

        # features
        # likely multicollinearity
        games_features = games_for_player_injury.drop(columns=['player_name', 'num_games', 'event']).to_numpy()

        num_games_player_injury = games_for_player_injury['num_games'].item()

        # repeat the game_games for T_max
        repeated_game_feature_instance = []
        for i in range(max_games_before_injury):
            # get the game features with game identifier
            games_features_with_identifier = np.append(games_features, i+1)
            repeated_game_feature_instance.append(games_features_with_identifier)

        X.append(repeated_game_feature_instance)

        T.append(num_games_player_injury)

        # event
        event = int(games_for_player_injury['event'])
        E.append(event)

    X = torch.tensor(X, dtype=torch.float32).squeeze()
    E = torch.tensor(E, dtype=torch.float32)
    T = torch.tensor(T)

    return X, T, E
