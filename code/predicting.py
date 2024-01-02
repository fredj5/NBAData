from clustering import clean_dicts, read_advanced_stats, read_season_stats, merge_dataframes
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from xgboost import XGBRegressor
from bs4 import BeautifulSoup
import requests

# Function to scrape a tables from Basketball Reference
def scrape_table(url, table_id):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': table_id})
    df = pd.read_html(str(table))[0]
    return df

# Custom cleaning function
def clean_dataframes(df):
    
    df.dropna()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.loc[:, ~df.columns.str.contains('Rk_y')]
    df = df.rename(columns={'Rk_x': 'Rk'})
    df = df.loc[:, ~df.columns.str.contains('Arena')]
    df = df.loc[:, ~df.columns.str.contains('Attend.')]
    
    return df


# Predicting NBA Champ function (training models)
def predict_champ(train_set, test_set):
    # Final dropping of any NaN values
    train_set = train_set.dropna()
    test_set = test_set.dropna()


    # Train set and target variable
    X_train = train_set.drop(columns = ['Team', 'Champion', 'Conference Finals', 'Season_Year', 'MP', 'Age'], axis=1)
    y_train = train_set['Champion']

    # Test set
    X_test = test_set.drop(columns = ['Team', 'Season_Year', 'MP', 'Age'])

    # Four different models
    model = LinearRegression()
    model_xgboost = XGBRegressor(eta=0.01, max_depth=4)
    model_lasso = Lasso()
    model_ridge = Ridge()


    # Training
    model.fit(X_train, y_train)
    model_xgboost.fit(X_train, y_train)
    model_lasso.fit(X_train, y_train)
    model_ridge.fit(X_train, y_train)

    # Predict confidence values for each team becoming the next NBA champion
    predictions = model.predict(X_test)
    xgg_predicitons = model_xgboost.predict(X_test)
    lasso_preds = model_lasso.predict(X_test)
    ridge_preds = model_ridge.predict(X_test)

    # Display Predictions
    results = pd.DataFrame({'Team': test_set['Team'], 'Predicted_Champion': predictions})
    results = results.sort_values(by='Predicted_Champion', ascending=False)

    results_xg = pd.DataFrame({'Team': test_set['Team'], 'Predicted_Champion': xgg_predicitons})
    results_xg = results_xg.sort_values(by='Predicted_Champion', ascending=False)

    results_lasso = pd.DataFrame({'Team': test_set['Team'], 'Predicted_Champion': lasso_preds})
    results_lasso = results_lasso.sort_values(by='Predicted_Champion', ascending=False)

    results_ridge = pd.DataFrame({'Team': test_set['Team'], 'Predicted_Champion': ridge_preds})
    results_ridge = results_ridge.sort_values(by='Predicted_Champion', ascending=False)

    # Print predictions for each model
    print('Linear Regression Predictions:')
    print()
    print(results)
    print('XGBoost Predictions:')
    print()
    print(results_xg)
    print('Lasso Regression Predictions:')
    print()
    print(results_lasso)
    print('Ridge Regression Predictions:')
    print()
    print(results_ridge)

# Predicting conference finalists
def predict_conference_finals(train_set, test_set):
    train_set = train_set.dropna()
    test_set = test_set.dropna()


    # Train set and target variable
    X_train = train_set.drop(columns = ['Team', 'Champion', 'Conference Finals', 'Season_Year', 'MP', 'Age'], axis=1)
    y_train = train_set['Conference Finals']

    # Test set
    X_test = test_set.drop(columns = ['Team', 'Season_Year', 'MP', 'Age'])

    
    # Four different models
    model = LinearRegression()
    model_xgboost = XGBRegressor(eta=0.01, max_depth=4)
    model_lasso = Lasso()
    model_ridge = Ridge()


    # Training
    model.fit(X_train, y_train)
    model_xgboost.fit(X_train, y_train)
    model_lasso.fit(X_train, y_train)
    model_ridge.fit(X_train, y_train)

    # Predict conference finals appearance confidence value
    predictions = model.predict(X_test)
    xgg_predicitons = model_xgboost.predict(X_test)
    lasso_preds = model_lasso.predict(X_test)
    ridge_preds = model_ridge.predict(X_test)

    # Display Predictions
    results = pd.DataFrame({'Team': test_set['Team'], 'Predicted Conference Finals Appearance': predictions})
    results = results.sort_values(by='Predicted Conference Finals Appearance', ascending=False)

    results_xg = pd.DataFrame({'Team': test_set['Team'], 'Predicted Conference Finals Appearance': xgg_predicitons})
    results_xg = results_xg.sort_values(by='Predicted Conference Finals Appearance', ascending=False)

    results_lasso = pd.DataFrame({'Team': test_set['Team'], 'Predicted Conference Finals Appearance': lasso_preds})
    results_lasso = results_lasso.sort_values(by='Predicted Conference Finals Appearance', ascending=False)

    results_ridge = pd.DataFrame({'Team': test_set['Team'], 'Predicted Conference Finals Appearance': ridge_preds})
    results_ridge = results_ridge.sort_values(by='Predicted Conference Finals Appearance', ascending=False)

    # Print predictions for each model
    print()
    print('Linear Regression Predictions:')
    print()
    print(results)
    print()
    print('XGBoost Predictions:')
    print()
    print(results_xg)
    print()
    print('Lasso Regression Predictions:')
    print()
    print(results_lasso)
    print()
    print('Ridge Regression Predictions:')
    print()
    print(results_ridge)


# Main driver
def main():

    # URLs for current season data
    per_game_url = "https://www.basketball-reference.com/leagues/NBA_2024.html#per_game-team"
    advanced_url = "https://www.basketball-reference.com/leagues/NBA_2024.html#advanced-team"

    # Scrape team statistics
    per_game_data = scrape_table(per_game_url, 'per_game-team')
    advanced_data = scrape_table(advanced_url, 'advanced-team')
    advanced_data.columns = advanced_data.columns.droplevel(0)
    clean_per_game = clean_dataframes(per_game_data)
    clean_advanced = clean_dataframes(advanced_data)

    # Test df
    test_df = pd.merge(clean_per_game, clean_advanced, left_on='Team', right_on='Team')
    test_df['Season_Year'] = int(2024)
    test_df = clean_dataframes(test_df)
    test_df['Team'] = test_df['Season_Year'].astype(str) + ' ' + test_df['Team']
    test_df.columns.values[47] = 'FT/FGA.1'
    test_df.columns.values[45] = 'TOV%.1'
    test_df.columns.values[44] = 'eFG%.1'

    # Paths to CSV data
    folder_path = "/Users/freddiejones/Desktop/CIS563/project/szn-stats"
    adv_folder_path = "/Users/freddiejones/Desktop/CIS563/project/advanced-stats"
    

    # Read in CSVs, clean data
    all_seasonal_dataframes = read_season_stats(folder_path)
    clean_seasonal_dataframes = clean_dicts(all_seasonal_dataframes)
    
    all_advanced_dataframes = read_advanced_stats(adv_folder_path)
    clean_advanced_dataframes = clean_dicts(all_advanced_dataframes)


    # Dictionary with dataframe of stats for every season
    merged_data = merge_dataframes(clean_seasonal_dataframes, clean_advanced_dataframes)
    clean_merged_data = clean_dicts(merged_data)

    # Concatenate all dataframes into a single dataframe for clustering
    every_szn_df = pd.concat(clean_merged_data.values(), ignore_index=True)
    every_szn_df['Team'] = every_szn_df['Season_Year'].astype(str) + ' ' + every_szn_df['Team']

    predict_champ(every_szn_df, test_df)
    predict_conference_finals(every_szn_df, test_df)


if __name__ == "__main__":
    main()
