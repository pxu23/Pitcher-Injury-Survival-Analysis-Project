## for each of the season, fit various survival analysis models
import pandas as pd
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt


def fit_game_level_cox_ph_model_for_season(season_year, recurrence):
    # read in the survival dataframe for the season
    survival_df_time_invariant_game_level_season = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/survival_df_{season_year}_time_invariant_game_level.csv")

    survival_df_time_invariant_game_level_season_to_fit = survival_df_time_invariant_game_level_season.drop(
        columns=['player_name','previous_injury_date','next_injury_date'])

    if not recurrence:
        survival_df_time_invariant_game_level_season_to_fit.drop(columns=['recurrence'],inplace=True)

    cph = CoxPHFitter()
    cph.fit(survival_df_time_invariant_game_level_season_to_fit,event_col="EVENT", duration_col="num_games")

    return cph

## evaluate the performance of the game-level Cox PH model on the held-out season
def evaluate_cox_ph_model_game_level_held_out_season(cox_ph_model, held_out_season_year, recurrence):

    # survival dataframe for the held out season
    survival_df_held_out_season_game_level_time_invariant = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/survival_df_{held_out_season_year}_time_invariant_game_level.csv")




    survival_df_held_out_season_game_level_time_invariant.drop(columns=['player_name','previous_injury_date','next_injury_date'],inplace=True)
    if not recurrence:
        survival_df_held_out_season_game_level_time_invariant.drop(columns=['recurrence'],inplace=True)

    concordance_score = cox_ph_model.score(survival_df_held_out_season_game_level_time_invariant, scoring_method="concordance_index")

    return concordance_score

# survival dataframe for the held out season
survival_df_held_out_season_game_level_time_invariant = pd.read_csv(f"../Survival-Dataframes/Time-Invariant/survival_df_{2024}_time_invariant_game_level.csv")

survival_df_held_out_season_game_level_time_invariant.drop(columns=['player_name','previous_injury_date','next_injury_date'],inplace=True)

cph_season_game_level = fit_game_level_cox_ph_model_for_season("2023", recurrence=True)

cox_survival = cph_season_game_level.predict_survival_function(survival_df_held_out_season_game_level_time_invariant).mean(axis=1)

# Cox Proportional Hazards survival function
plt.plot(cox_survival.index, cox_survival.values, label="Cox PH Model", linestyle="-.", color="green")

# Labels and Legend
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.title("Cox Proportional Hazards Survival Functions for Season 2023")
plt.legend()
plt.show()