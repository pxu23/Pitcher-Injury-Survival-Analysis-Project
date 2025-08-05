This is the repo for Survival Analysis to Predict Pitcher Injury. 

The ``statcast_data_utils.py`` file acquires the statcast data for each of the 2020-2024 seasons as well as the 2018-2023 
and 2022-2023 seasons.

The ``demographics.py`` file contains the functions to get the demographic features of a pitcher, as well as the injured 
pitchers that have demographic information. 

The ``baseball_injury_scrapper.py`` scrapes the Prosports Injury data for the pitcher injury

The ``injury_data_utils.py`` contains the functions to process the injury dateframe, including
getting the relinquishment dates for each player.

The ``Time-Varying-Survival-Analysis`` folder creates the time-varying survival dataframes and performs
the statistical survival analysis for the 2022-2023 season, 2018-2023 season and each of the 2020-2023 season at both the
game-level and pitch-level.

The ``Time-Invariant-Survival-Analysis`` folder creates the time-invariant survival dataframes and performs
the statistical survival analysis for the 2022-2023 season, 2018-2023 season and each of the 2020-2023 season at both the
game-level and pitch-level. Here, it does not model the dependence on the previous injury in the followup period.

The ``Time-Invariant-Survival-Analysis-Recurrent`` folder creates the time-invariant survival dataframes and performs
the statistical survival analysis for the 2022-2023 season, 2018-2023 season and each of the 2020-2023 season at both the
game-level and pitch-level. Here, it does model the dependence on the previous injury in the followup period.

The ``ML_Survival_Analysis`` folder contains the implementation of the ML models to perform survival analysis for the 2018-2023,
2022-2023, and each of 2020-2023 seasons at both the pitch and game level. Here, the analysis is time-invariant and does not
consider the dependence of previous injury in the followup period.

The ``ML_Survival_Analysis_Recurrent`` folder contains the implementation of the ML models to perform survival analysis for the 2018-2023,
2022-2023, and each of 2020-2023 seasons at both the pitch and game level. Here, the analysis is time-invariant and does
consider the dependence of previous injury in the followup period.

The ``Statcast_Data`` folder contains the statcast data used to perform survival analysis.

The ``Lahman_Demographic_Data`` folder contains the demographic information of baseball pitchers that we use for survival analysis.

## How to Run
Run the code according to the following steps. Note that Steps 1-4 must be completed before running Step 5 in order to have the survival dataframes ready.
Note that Steps 2-4 can be completed in any order. 
1. Make sure that you have the Statcast data available for survival analysis. Run the ``statcast_data_utils.py`` file to generate the Statcast data
2. Run the following notebooks in ``Time-Invariant-Survival-Analysis`` in any order to both create the survival dataframes and reproduce the results for the
   statistical survival analysis models for time-invariant pitch and game characteristics.
   - ``survival_analysis_game_level_time_invariant.ipynb`` for game-level results for the 2018-2023 and 2022-2023 followup period.
   - ``survival_analysis_game_level_time_invariant_seasonal.ipynb`` for game-level results for each of the 2020-2023 seasons.
   - ``survival_analysis_pitch_level_time_invariant.ipynb`` for pitch-level results for the 2018-2023 and 2022-2023 followup period.
   - ``survival_analysis_pitch_level_time_invariant_seasonal.ipynb`` for pitch-level results for each of the 2020-2023 seasons.
3. Run the following notebooks in ``Time-Varying-Survival-Analysis`` in any order to both create the survival dataframes and reproduce the results for the
   statistical survival analysis models for time-varying pitch and game characteristics.
   - ``survival_analysis_game_level_time_varying.ipynb`` for game-level results for the 2018-2023 and 2022-2023 followup period.
   - ``survival_analysis_game_level_time_varying_seasonal.ipynb`` for game-level results for each of the 2020-2023 seasons.
   - ``survival_analysis_pitch_level_time_varying.ipynb`` for pitch-level results for the 2018-2023 and 2022-2023 followup period.
   - ``survival_analysis_pitch_level_time_varying_seasonal.ipynb`` for pitch-level results for each of the 2020-2023 seasons.
4. Run the following notebooks in ``Time-Invariant-Recurrent-Survival-Analysis`` in any order to both create the survival dataframes and reproduce the results for the
   statistical survival analysis models for time-invariant pitch and game characteristics that considers injury recurrence.
    - ``survival_analysis_game_level_time_invariant_recurrent.ipynb`` for game-level results for the 2018-2023 and 2022-2023 followup period.
    - ``survival_analysis_game_level_time_invariant_seasonal_recurrent.ipynb`` for game-level results for each of the 2020-2023 seasons.
    - ``survival_analysis_pitch_level_time_invariant_recurrent.ipynb`` for pitch-level results for the 2018-2023 and 2022-2023 followup period.
    - ``survival_analysis_pitch_level_time_invariant_seasonal_recurrent.ipynb`` for pitch-level results for each of the 2020-2023 seasons.
5. Run the following notebooks in ``ML_Survival_Analysis`` in any order to both create the survival dataframes and reproduce the results for the
   Machine Learning survival analysis models for time-invariant pitch and game characteristics.
    - ``gradient_boosting_models.ipynb`` for Gradient Boosting model results for the 2018-2023 and 2022-2023 followup period at both pitch and game level.
    - ``random_survival_forest.ipynb`` for Random Survival Forest model results for the 2018-2023 and 2022-2023 followup period at both pitch and game level.
    - ``gradient_boosting_models_seasonal.ipynb`` for Gradient Boosting model results for each of 2020-2023 seasons at both pitch and game level.
    - ``random_survival_forest_models_seasonal.ipynb`` for Random Survival Forest model results for each of 2020-2023 seasons at both pitch and game level.
6. Run the following notebooks in ``ML_Survival_Analysis_Recurrent`` in any order to both create the survival dataframes and reproduce the results for the
   Machine Learning survival analysis models for time-invariant pitch and game characteristics.
   - ``gradient_boosting_models_seasonal.ipynb`` for Gradient Boosting model results for the 2020-2023 seasons at both pitch and game level while considering recurrence.
   - ``random_survival_forest_seasonal.ipynb`` for Random Survival Forest model results for the 2020-2023 seasons at both pitch and game level while considering recurrence.

