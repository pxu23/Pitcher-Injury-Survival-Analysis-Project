from lifelines import CoxPHFitter
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def fit_game_level_cox_ph_model_for_multiple_season(start_season_year, end_season_year,
                                                    recurrence):
    """
        This function fits a Cox PH model for multiple seasons.
        :param start_season_year: The start year of the season.
        :param end_season_year: The end year of the season.
        :param recurrence: whether injury recurrence is considered
        :return: Cox PH model fitted for multiple seasons.
    """
    # read in the survival dataframe for multiple season
    survival_df_time_invariant_game_level_multiple_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/survival_df_{start_season_year}_{end_season_year}_time_invariant_game_level.csv")

    # drop the player_name, previous_injury_date, and next_injury_date columns
    survival_df_time_invariant_game_level_multiple_season_to_fit = survival_df_time_invariant_game_level_multiple_season.drop(
        columns=['player_name','previous_injury_date','next_injury_date'])

    # if do not consider recurrence, drop the recurrence column
    if not recurrence:
        survival_df_time_invariant_game_level_multiple_season_to_fit.drop(columns=['recurrence'],inplace=True)

    # initialize and fit the Cox PH model on multiple seasons
    cph = CoxPHFitter()
    cph.fit(survival_df_time_invariant_game_level_multiple_season_to_fit,event_col="EVENT", duration_col="num_games")

    return cph

