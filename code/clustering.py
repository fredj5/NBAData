# Packages
import pandas as pd
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2 as cv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup
from pyts.approximation import SymbolicAggregateApproximation

def scrape_current_season(url):
    # URL to current szn stats
    response = requests.get(url)
     # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table containing team statistics
        table = soup.find('table', {'id': 'per_game-team'})

        # Convert the HTML table to a Pandas DataFrame
        df = pd.read_html(str(table))[0]

        # Remove unnecessary rows and columns
        df = df.dropna()

        return df

    else:
        print(f"Failed to retrieve data from {url}. Status Code: {response.status_code}")
        return None

def scrape_advanced_team_stats(url):
    """
    Scrape advanced team statistics from the provided URL.

    Parameters:
        url (str): URL of the NBA advanced team statistics page.

    Returns:
        DataFrame: DataFrame containing advanced team statistics.
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table containing advanced team statistics
        table = soup.find('table', {'id': 'advanced-team'})

        # Convert the HTML table to a Pandas DataFrame
        df = pd.read_html(str(table))[0]

        # Remove unnecessary rows and columns
        df = df.dropna()

        return df

    else:
        print(f"Failed to retrieve data from {url}. Status Code: {response.status_code}")
        return None


# Read in CSV files for season averages by team
def read_season_stats(folder_path):
    
    # Error checking
    if not os.path.isdir(folder_path):
        print(f"The path '{folder_path}' is not a directory.")
        return
    
    # List of files in seasonal stats folder, dictionary to store each seasons data
    file_list = os.listdir(folder_path)
    dataframes = {}

    # Loop through list of CSVs
    for file in file_list:
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)

            # Dataframing
            dataframe_name = os.path.splitext(file)[0]  # Use the file name as the DataFrame name
            dataframes[dataframe_name] = pd.read_csv(file_path)

    return dataframes

def read_advanced_stats(adv_folder_path):
    
    # Error checking
    if not os.path.isdir(adv_folder_path):
        print(f"The path '{adv_folder_path}' is not a directory.")
        return
    
    # List of files in seasonal stats folder, dictionary to store each seasons data
    file_list = os.listdir(adv_folder_path)
    dataframes = {}

    # Loop through list of CSVs
    for file in file_list:
        if file.endswith(".csv"):
            file_path = os.path.join(adv_folder_path, file)

            # Dataframing
            dataframe_name = os.path.splitext(file)[0]  # Use the file name as the DataFrame name
            dataframes[dataframe_name] = pd.read_csv(file_path)

    return dataframes

def merge_dataframes(dict1, dict2):
    """
    Merge DataFrames from two dictionaries I made

    Parameters:
        - dict1 (dict): Seasonal stats dictionary.
        - dict2 (dict): Advanced stats dictionary.
        - common_column (str): The common column to use for merging, in this case it will be TEAM.

    Returns:
        dict: A new dictionary with merged DataFrames containing seasonal and advanced stats from the same season.
    """
    # Merged dictionaries
    merged_dict = {}

    # Merge DataFrames based on the common column

    # Iterate through the keys in the first dictionary
    for key1, dataframe1 in dict1.items():
        # Construct a key for the second dictionary based on the common pattern
        key2 = key1.replace('szn-stats', 'advanced-stats')
        
        # Check if the key exists in the second dictionary
        if key2 in dict2:
            # Merge DataFrames based on the common column
            merged_df = pd.merge(dataframe1, dict2[key2], left_on=dataframe1['Team'], right_on=dict2[key2]['Team'])
            merged_df['Season_Year'] = int(key1[5:9])
            merged_dict[key1] = merged_df

    return merged_dict

def clean_dataframes(dict):
    
    cleaned_dict = {}

    # Various specific cleaning techniques for different columns and strings
    for key, df in dict.items():
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]    # Remove unnamed columns
        
        df = df.loc[:, ~df.columns.str.contains('Arena')]   # Remove unwanted columns
        df = df.loc[:, ~df.columns.str.contains('Attend.')]
        df = df.loc[:, ~df.columns.str.contains('key_0')]
        df = df.loc[:, ~df.columns.str.contains('Team_y')]
        df = df.loc[:, ~df.columns.str.contains('Rk_y')]        
        df = df.rename(columns={'Rk_x': 'Rk', 'Team_x': 'Team'})
        
        # Convert "team name" column to lowercase
        df['Team'] = df['Team'].str.lower()
        # df['Team'] = df['Team'].str.replace('*', '')    # Remove * for indicating a playoff team for uniformity between seasons


        # Drop empty columns
        df = df.dropna(how='all')

        cleaned_dict[key] = df

    return cleaned_dict


# def logistic_regression(dataframes, curr_szn_key):



def main():

    # URLs for current season data
    per_game_url = "https://www.basketball-reference.com/leagues/NBA_2024.html#per_game-team"
    advanced_url = "https://www.basketball-reference.com/leagues/NBA_2024.html#advanced-team"

    # Scrape team statistics
    per_game_data = scrape_current_season(per_game_url)
    advanced_data = scrape_advanced_team_stats(advanced_url)


    # Paths to CSV data
    folder_path = "/Users/freddiejones/Desktop/CIS563/project/szn-stats"
    adv_folder_path = "/Users/freddiejones/Desktop/CIS563/project/advanced-stats"

    # Read in CSVs, clean data
    all_seasonal_dataframes = read_season_stats(folder_path)
    clean_seasonal_dataframes = clean_dataframes(all_seasonal_dataframes)
    
    all_advanced_dataframes = read_advanced_stats(adv_folder_path)
    clean_advanced_dataframes = clean_dataframes(all_advanced_dataframes)

    # Dictionary with dataframe of stats for every season
    merged_data = merge_dataframes(clean_seasonal_dataframes, clean_advanced_dataframes)
    clean_merged_data = clean_dataframes(merged_data)
    
    # Concatenate all dataframes into a single dataframe for clustering
    every_szn_df = pd.concat(clean_merged_data.values(), ignore_index=True)
    every_szn_df['Team'] = every_szn_df['Season_Year'].astype(str) + ' ' + every_szn_df['Team']
    print(every_szn_df)

    # Mapping teams to logos
    logo_folder = '/Users/freddiejones/Desktop/CIS563/project/logos/'
    team_image_mapping = {
        'boston celtics': 'celtics.png',
        'los angeles clippers': 'clippers.png',
        'houston rockets': 'rockets.png',
        'minnesota timberwolves': 'timberwolves.png',
        'portland trail blazers': 'blazers.png',
        'oklahoma city thunder': 'thunder.png',
        'san antonio spurs': 'spurs.png',
        'phoenix suns': 'suns.png',
        'dallas mavericks': 'mavericks.png',
        'denver nuggets': 'nuggets.png',
        'golden state warriors': 'warriors.png',
        'los angeles lakers': 'lakers.png',
        'miami heat': 'heat.png',
        'toronto raptors': 'raptors.png',
        'atlanta hawks': 'hawks.png',
        'detroit pistons': 'pistons.png',
        'washington wizards': 'wizards.png',
        'sacramento kings': 'kings.png',
        'new orleans pelicans': 'pelicans.png',
        'philadelphia 76ers': 'sixers.png',
        'new york knicks': 'knicks.png',
        'brooklyn nets': 'nets.png',
        'cleveland cavaliers': 'cavaliers.png',
        'charlotte bobcats': 'hornets.png',
        'indiana pacers': 'pacers.png',
        'orlando magic': 'magic.png',
        'memphis grizzlies': 'grizzlies.png',
        'milwaukee bucks': 'bucks.png',
        'utah jazz': 'jazz.png',
        'chicago bulls': 'bulls.png'
    }
    
    # Clustering.....
    columns_for_clustering = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P',
                              '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
                              'BLK', 'TOV', 'PF', 'PTS', 'Age', 'W', 'L', 'PW',
                              'PL', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr',
                              '3PAr', 'TS%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'eFG%.1', 'TOV%.1',
                              'DRB%', 'FT/FGA.1']
    
    # for key, df in clean_merged_data.items():
    # Drop any remaining NaN values
    every_szn_df = every_szn_df.dropna()

    # Apply KMeans clustering
    scaler = StandardScaler()
    clustering_data = scaler.fit_transform(every_szn_df[['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P',
                              '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
                              'BLK', 'TOV', 'PF', 'PTS', 'Age', 'W', 'L', 'PW',
                              'PL', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr',
                              '3PAr', 'TS%', 'eFG%', 'TOV%', 'ORB%']])

    
    # KMeans
    kmeans = KMeans(n_clusters=6, random_state=26)
    every_szn_df['KMeans_Cluster'] = kmeans.fit_predict(clustering_data)

    # Apply Agglomerative clustering
    agg_clustering = AgglomerativeClustering(n_clusters=8)
    every_szn_df['Agg_Cluster'] = agg_clustering.fit_predict(clustering_data)

    # Print the results
    print(f"Results for large DataFrame:")
    print(every_szn_df[['Team', 'KMeans_Cluster', 'Agg_Cluster']])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(every_szn_df['ORtg'], every_szn_df['DRtg'], c='black', alpha=0)  # Use a constant color and make the points transparent

    # Visualize the clusters
    # plt.scatter(df['ORtg'], df['DRtg'], c=df['KMeans_Cluster'], cmap='viridis', label='KMeans Clusters')
    plt.scatter(every_szn_df['ORtg'], every_szn_df['DRtg'], c=every_szn_df['Agg_Cluster'], cmap='plasma', label='Agg Clusters')
        
    # Annotate points with team names
    for i, team_name in enumerate(every_szn_df['Team']):
        plt.annotate(team_name, (every_szn_df['ORtg'].iloc[i], every_szn_df['DRtg'].iloc[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6)
        
    # Make the quadrants and the diagonal
    ax.plot([.5, .5], [0, 1], transform=ax.transAxes, alpha=0.15)
    ax.plot([0, 1], [.5, .5], transform=ax.transAxes, alpha=0.15)
    ax.plot([.1, .9], [.9, .1], transform=ax.transAxes, alpha=0.15)


    plt.xlabel('ORtg')
    plt.ylabel('DRtg')
    plt.title(f'Clustering like teams since 1985')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()



