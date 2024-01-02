# Packages
import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

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
            dataframe_name = os.path.splitext(file)[0]
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
            dataframe_name = os.path.splitext(file)[0] 
            dataframes[dataframe_name] = pd.read_csv(file_path)

    return dataframes

def merge_dataframes(dict1, dict2):
    # Merged dictionaries
    merged_dict = {}

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

def clean_dicts(dict):
    
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
        
        # Convert team name column to lowercase
        df['Team'] = df['Team'].str.lower()

        # Drop empty columns
        df = df.dropna(how='all')

        cleaned_dict[key] = df

    return cleaned_dict


def main():

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
    print(every_szn_df)
    
    # Every column for reference
    columns_for_clustering = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P',
                              '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
                              'BLK', 'TOV', 'PF', 'PTS', 'Age', 'W', 'L', 'PW',
                              'PL', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr',
                              '3PAr', 'TS%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'eFG%.1', 'TOV%.1',
                              'DRB%', 'FT/FGA.1']
    
    # Drop any remaining NaN values
    every_szn_df = every_szn_df.dropna()

    # Group by season and calculate average values for each season
    time_series_data = every_szn_df.groupby('Season_Year').mean()
    percentage_columns_to_plot = ['FG%', '3P%', 'FT%', 'TS%', 'eFG%']
    other_columns_to_plot = ['3P', '3PA', '2P', '2PA', 'PTS', 'Pace', 'FTA']

    # Plotting the time series
    plt.figure(figsize=(12, 6))

    for column in percentage_columns_to_plot:
        plt.plot(time_series_data.index, time_series_data[column], marker='o', linestyle='-', label=column)

    plt.title('League Average Percentage Per Season')
    plt.xlabel('Year')
    plt.ylabel('Percentages')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the time series
    plt.figure(figsize=(12, 6))

    for column in other_columns_to_plot:
        plt.plot(time_series_data.index, time_series_data[column], marker='o', linestyle='-', label=column)

    plt.title('League Average Shooting Volume/Pace')
    plt.xlabel('Year')
    plt.ylabel('Average')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Standard
    minmax = MinMaxScaler()
    scaler = StandardScaler()
    clustering_data = every_szn_df[['FG%', '3P%', 'FT%', 'ORB', 'DRB', 'AST', 'STL',
                                   'BLK', 'TOV', 'PTS', 'W', 'L', 'NRtg', 'Champion',
                                   'TS%', 'eFG%', 'eFG%.1']]
    
    clustering_data = minmax.fit_transform(clustering_data)

    
    # PCA
    n_components = 2
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(clustering_data)

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=26)
    every_szn_df['KMeans_Cluster'] = kmeans.fit_predict(clustering_data)

    # Apply Agglomerative clustering
    agg_clustering = AgglomerativeClustering(n_clusters=6, linkage='ward')
    every_szn_df['Agg_Cluster'] = agg_clustering.fit_predict(clustering_data)

    # Print the results
    print(f"Results for large DataFrame:")
    print(every_szn_df[['Team', 'KMeans_Cluster', 'Agg_Cluster']])

    # Create plot for cultsering
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c='black', alpha=0)

    # Visualize the clusters (agglomerative or KMeans)
    # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=every_szn_df['KMeans_Cluster'], cmap='plasma', label='KMeans Clusters')
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=every_szn_df['Agg_Cluster'], cmap='plasma', label='Agg Clusters')
    
    # Annotate points with team names
    for i, team_name in enumerate(every_szn_df['Team']):
        plt.annotate(team_name, (reduced_data[i, 0], reduced_data[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6)
        
    # Make the quadrants and the diagonal
    ax.plot([.5, .5], [0, 1], transform=ax.transAxes, alpha=0.15)
    ax.plot([0, 1], [.5, .5], transform=ax.transAxes, alpha=0.15)
    ax.plot([.1, .9], [.9, .1], transform=ax.transAxes, alpha=0.15)

    # Label and display clusters
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Clustering like teams since 1985')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()



