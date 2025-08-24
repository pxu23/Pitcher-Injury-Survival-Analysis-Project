This is the repo for Survival Analysis to Predict Pitcher Injury. 

The ``statcast_data_utils.py`` file acquires the statcast data for the followup seasons in our work.

The ``game_level_statcast_data.py`` file gets the average, 10th percentile, and 90th perentile pitch characteristics for each 
game played by a pitcher in a followup period.

The ``Statcast_Data`` folder contains the statcast data used to perform survival analysis.

The ``Lahman_Demographic_Data`` folder contains the demographic information of baseball pitchers that we use for survival analysis.

The ``Game-Level-Pitch-Characteristics-Statcast`` contains the game-level statcast data for each played game during each followup season.

The ``Injury_Data`` folder contains the injury dataframe for each of the followup season.

The ``Pitchers_Season`` folder contains the list of pitchers for each of the followup season, including those with demographic information.

The ``Survival_Dataframe_Construction`` folder contains the code to create the survival analysis formulation of the
pitcher injury prediction problem including both the time-invariant and time-varying formulations.

The ``Survival-Dataframes`` folder contains the time-invariant and time-varying survival dataframes for each of the followup
seasons that we considered.

The ``Survival_Dataframe_Analysis`` folder performs the feature analysis for the survival dataframe for different followup seasons.

The ``Baselines-Survival-Models`` contains the scripts to train and evaluate the baseline statistical and 
Machine Learning survival models.

The ``SurvivalRNN`` folder contains the implementation of the SurvivalRNN model along with the scripts
to train and evaluate the RNN. 

The ``Survival_Curves`` folder contains the overall survival curves for the SurvivalRNN and other survival models for
different training and evaluation seasons.

The ``Survival_Probabilities`` folder contains the survival probabilities predictions for each instances during a 
followup season for each of the survival models.

The ``2025-Season-Predicton`` folder contains the pitch, injury, and demographic data for top 2025 Major League pitchers,
as well as the predicted survival curve for each of them. 

## How to Run
Run the code according to the following steps.
1. Make sure that you have the Statcast pitch data available for survival analysis. Run the ``statcast_data_utils.py`` file to generate the Statcast data
2. Run the ``game_level_statcast_data.py`` to acquire the game-level Statcast data for each of the followup seasons.
3. Run the ``get_pitchers_season.py`` to acquire the list of healthy and injured pitchers for each season.
4. Run the ``survival_dataframe_construction_time_varying_game_level.py`` to construct the time-varying survival dataframes for each of the seasons.
4. Run the ``survival_dataframe_construction_time_invariant_game_level.py`` to construct the time-invariant survival dataframes for each season.
5. Run the following files in ``Baseline-Survival-Models`` to reproduce the results in Table for the Concordance Index and Integrated Brier Score.
   - `statistical_model_seasonal.py` to reproduce the results for the Cox Proportional Hazard and Integrated Brier Scores
   - `ml_survival_models_seasonal.py` to reproduce the Concordance Index and Integrated Brier Scores for the Random Survival Forest and Gradient Boosting models
6. Run the `rnn_discrete_sports_injury_single_season.py` file to reproduce the results for the Concordance Index and Integrated Brier Scores in Table .
7. Run the `plot_survival_curves_rnn.py` to generate the overall survival curves for the RNN model along with the baseline survival models in Table .
8. Run the following files in ``2025-Season-Prediction`` to generate the individual predictions for the 2025 season.
   1. Run `get_statcast_data_player.py` to acquire the Statcast pitch data for the player.
   2. Run `get_game_level_statcast_data_player.py` to acquire the game-level pitch characteristics for the player.
   3. Run `survival_dataframe_construction_time_invariant_game_level_2025_pitchers.py` to get the time-invariant survival dataframe for the 2025 season for player.
   4. Run `survival_dataframe_construction_time_varying_game_level_2025_pitchers.py` to get the time-varying survival dataframe for the 2025 season for player
   5. Run `individual_pitcher_survival_rnn.py` to get the predicted RNN curve for the pitcher along with that for baseline survival models.

