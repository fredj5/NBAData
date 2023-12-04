import pandas as pd
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2 as cv




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
            merged_dict[key1] = pd.merge(dataframe1, dict2[key2], left_on=dataframe1['Team'], right_on=dict2[key2]['Team'])

    return merged_dict

def clean_dataframes(dict):
    
    cleaned_dict = {}

    # Various specific cleaning techniques for different columns and strings
    for key, df in dict.items():
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.loc[:, ~df.columns.str.contains('Arena')]
        df = df.loc[:, ~df.columns.str.contains('Attend.')]
        df = df.loc[:, ~df.columns.str.contains('key_0')]
        df = df.loc[:, ~df.columns.str.contains('Team_y')]
        df = df.loc[:, ~df.columns.str.contains('Rk_y')]        
        df = df.rename(columns={'Rk_x': 'Rk', 'Team_x': 'Team'})
        
        # Convert both "team name" columns to lowercase
        df['Team'] = df['Team'].str.lower()
        df['Team'] = df['Team'].str.replace('*', '')


        # Drop empty columns
        df = df.dropna(how='all')

        cleaned_dict[key] = df

    return cleaned_dict

# Simple resizing function
def resize_image(img_path, size):
    
    image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    # Convert the image from BGR to RGB
    imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Convert the image to PIL format
    img = Image.fromarray(imageRGB)
    # Resize the image
    img = img.resize(size, Image.ANTIALIAS)  # Use Image.ANTIALIAS for high-quality downsampling
    return img


def main():

    # Paths to CSV data
    folder_path = "/Users/freddiejones/Desktop/CIS563/project/szn-stats"
    adv_folder_path = "/Users/freddiejones/Desktop/CIS563/project/advanced-stats"

    # Read in CSVs, clean data
    all_seasonal_dataframes = read_season_stats(folder_path)
    clean_seasonal_dataframes = clean_dataframes(all_seasonal_dataframes)
    
    all_advanced_dataframes = read_advanced_stats(adv_folder_path)
    clean_advanced_dataframes = clean_dataframes(all_advanced_dataframes)

    merged_data = merge_dataframes(clean_seasonal_dataframes, clean_advanced_dataframes)
    clean_merged_data = clean_dataframes(merged_data)

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
    
    for key, df in clean_merged_data.items():
        # Drop any remaining NaN values
        df = df.dropna()

        # Apply KMeans clustering
        scaler = StandardScaler()
        clustering_data = scaler.fit_transform(df[['ORtg', 'DRtg']])
        kmeans = KMeans(n_clusters=6, random_state=26)
        df['KMeans_Cluster'] = kmeans.fit_predict(clustering_data)

        # Apply Agglomerative clustering
        agg_clustering = AgglomerativeClustering(n_clusters=8)
        df['Agg_Cluster'] = agg_clustering.fit_predict(clustering_data)

        # Print the results
        print(f"Results for {key} DataFrame:")
        print(df[['Team', 'KMeans_Cluster', 'Agg_Cluster']])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['ORtg'], df['DRtg'], c='black', alpha=0)  # Use a constant color and make the points transparent

        # Visualize the clusters
        # plt.scatter(df['ORtg'], df['DRtg'], c=df['KMeans_Cluster'], cmap='viridis', label='KMeans Clusters')
        plt.scatter(df['ORtg'], df['DRtg'], c=df['Agg_Cluster'], cmap='plasma', label='Agg Clusters')
        
        # Annotate points with team names
        for i, team_name in enumerate(df['Team']):
            plt.annotate(team_name, (df['ORtg'].iloc[i], df['DRtg'].iloc[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6)
        
        ax.plot([.5, .5], [0, 1], transform=ax.transAxes, alpha=0.15)
        ax.plot([0, 1], [.5, .5], transform=ax.transAxes, alpha=0.15)
        ax.plot([.1, .9], [.9, .1], transform=ax.transAxes, alpha=0.15)


        plt.xlabel('ORtg')
        plt.ylabel('DRtg')
        plt.title(f'Clusters for {key}')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    main()



