"""
Spotify Song Recommender System (Console Version)

This program implements a song recommendation system using machine learning techniques.
It processes Spotify track data to cluster similar songs and recommend tracks based on user preferences.
The application provides a console-based interface with visualizations.

Author: [Student Name]
Date: April 6, 2025
Course: ELEC825 - Machine Learning Applications
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class SpotifySongRecommender:
    """A class that implements a song recommendation system using Spotify track data."""
    
    def __init__(self, data_path="spotify_data"):
        """Initialize the song recommender with path to data file."""
        self.data_path = data_path
        self.data = None
        self.features = None
        self.cluster_pipeline = None
        self.song_cluster_labels = None
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"Created directory: {data_path}")
            print("Please place tracks.csv file in this directory.")
    
    def load_data(self, tracks_file="tracks.csv"):
        """
        Load Spotify tracks dataset file.
        
        Args:
            tracks_file (str): Filename for the tracks data
            
        Returns:
            tuple: (bool, str) Success status and message
        """
        try:
            # Construct full file path
            tracks_path = os.path.join(self.data_path, tracks_file)
            
            # Load data file
            self.data = pd.read_csv(tracks_path)
            
            return True, "Data loaded successfully!"
            
        except FileNotFoundError as e:
            return False, f"Error loading data: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    def get_data_info(self):
        """
        Get basic information about the dataset.
        
        Returns:
            str: Information about the dataset
        """
        if self.data is None:
            return "Data not loaded. Please load data first."
        
        info = []
        info.append(f"Tracks dataset shape: {self.data.shape}")
        
        # Add sample data
        info.append("\nSample tracks data:")
        info.append(self.data.head().to_string())
        
        return "\n".join(info)
    
    def prepare_features_for_clustering(self):
        """
        Prepare audio features for clustering by selecting and scaling relevant features.
        
        Returns:
            tuple: (bool, str) Success status and message
        """
        if self.data is None:
            return False, "Data not loaded. Please load data first."
        
        # Select features for clustering
        self.features = ['danceability', 'energy', 'loudness', 'speechiness', 
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        # Create a subset with only the selected features
        feature_subset = self.data[self.features]
        
        # Create a pipeline for preprocessing and clustering
        self.cluster_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('kmeans', KMeans(n_clusters=10, random_state=42))  # K-means clustering
        ])
        
        # Fit the pipeline to the data
        self.cluster_pipeline.fit(feature_subset)
        
        # Get cluster labels for each song
        self.song_cluster_labels = self.cluster_pipeline.predict(feature_subset)
        
        # Add cluster labels to the original data
        self.data['cluster_label'] = self.song_cluster_labels
        
        # Get cluster information
        cluster_counts = self.data['cluster_label'].value_counts().sort_index()
        
        message = "Features prepared for clustering.\n"
        message += f"Number of clusters: {len(np.unique(self.song_cluster_labels))}\n"
        message += f"Features used: {', '.join(self.features)}\n\n"
        message += "Cluster distribution:\n"
        
        for cluster, count in cluster_counts.items():
            message += f"Cluster {cluster}: {count} songs ({count/len(self.data)*100:.1f}%)\n"
        
        return True, message
    
    def get_cluster_visualization(self):
        """
        Create a visualization of song clusters using PCA.
        
        Returns:
            matplotlib.figure.Figure: The cluster visualization figure
        """
        if self.song_cluster_labels is None:
            return None
        
        # Select features for visualization
        feature_subset = self.data[self.features]
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(self.cluster_pipeline.named_steps['scaler'].transform(feature_subset))
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'PC1': reduced_features[:, 0],
            'PC2': reduced_features[:, 1],
            'Cluster': self.song_cluster_labels
        })
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Use a colormap with distinct colors
        cmap = plt.cm.get_cmap('viridis', len(np.unique(self.song_cluster_labels)))
        
        # Plot each cluster with a different color
        for cluster in np.unique(self.song_cluster_labels):
            cluster_data = plot_df[plot_df['Cluster'] == cluster]
            plt.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                       c=[cmap(cluster)], label=f'Cluster {cluster}',
                       alpha=0.7, edgecolors='none', s=30)
        
        # Add labels and legend
        plt.title('Song Clusters Visualization (PCA)', fontsize=16)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_tsne_visualization(self):
        """
        Create a visualization of song clusters using t-SNE.
        
        Returns:
            matplotlib.figure.Figure: The t-SNE visualization figure
        """
        if self.song_cluster_labels is None:
            return None
        
        # Select features for visualization
        feature_subset = self.data[self.features]
        
        # Standardize features
        scaled_features = self.cluster_pipeline.named_steps['scaler'].transform(feature_subset)
        
        # Apply t-SNE for dimensionality reduction
        print("Computing t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        reduced_features = tsne.fit_transform(scaled_features)
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'TSNE1': reduced_features[:, 0],
            'TSNE2': reduced_features[:, 1],
            'Cluster': self.song_cluster_labels
        })
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Use a colormap with distinct colors
        cmap = plt.cm.get_cmap('viridis', len(np.unique(self.song_cluster_labels)))
        
        # Plot each cluster with a different color
        for cluster in np.unique(self.song_cluster_labels):
            cluster_data = plot_df[plot_df['Cluster'] == cluster]
            plt.scatter(cluster_data['TSNE1'], cluster_data['TSNE2'], 
                       c=[cmap(cluster)], label=f'Cluster {cluster}',
                       alpha=0.7, edgecolors='none', s=30)
        
        # Add labels and legend
        plt.title('Song Clusters Visualization (t-SNE)', fontsize=16)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Add note about t-SNE
        #plt.figtext(0.5, 0.01, 
                   #"Note: t-SNE focuses on preserving local structure and cluster separation", 
                   #ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
        return plt.gcf()
    
    def get_cluster_characteristics_plot(self):
        """
        Create a visualization of cluster characteristics.
        
        Returns:
            matplotlib.figure.Figure: The cluster characteristics figure
        """
        if self.song_cluster_labels is None:
            return None
        
        # Get mean feature values for each cluster
        cluster_means = self.data.groupby('cluster_label')[self.features].mean()
        
        # Standardize the means for better visualization
        scaler = StandardScaler()
        cluster_means_scaled = pd.DataFrame(
            scaler.fit_transform(cluster_means),
            index=cluster_means.index,
            columns=cluster_means.columns
        )
        
        # Create a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_means_scaled, cmap="viridis", annot=True, fmt=".2f", linewidths=.5)
        plt.title('Cluster Characteristics (Standardized Feature Values)', fontsize=16)
        plt.xlabel('Audio Features')
        plt.ylabel('Cluster')
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_radar_chart(self, song_data, recommendations=None):
        """
        Create a radar chart comparing a song with its recommendations.
        
        Args:
            song_data (pd.Series): The input song data
            recommendations (pd.DataFrame, optional): Recommended songs data
            
        Returns:
            matplotlib.figure.Figure: The radar chart figure
        """
        # Select features for the radar chart
        radar_features = ['danceability', 'energy', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        # Filter features that exist in the dataset
        radar_features = [f for f in radar_features if f in song_data.index]
        
        if not radar_features:
            return None
        
        # Number of features
        N = len(radar_features)
        
        # Create angles for each feature
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add feature labels
        plt.xticks(angles[:-1], [f.capitalize() for f in radar_features], size=12)
        
        # Draw y-axis labels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot input song
        input_values = [song_data[feature] for feature in radar_features]
        input_values += input_values[:1]  # Close the loop
        ax.plot(angles, input_values, linewidth=2, linestyle='solid', label=f"{song_data['track_name']}")
        ax.fill(angles, input_values, alpha=0.1)
        
        # Plot recommendations if provided
        if recommendations is not None:
            # Use a colormap for recommendations
            cmap = plt.cm.get_cmap('viridis', len(recommendations) + 1)
            
            for i, (_, rec) in enumerate(recommendations.iterrows()):
                rec_values = [rec[feature] for feature in radar_features]
                rec_values += rec_values[:1]  # Close the loop
                ax.plot(angles, rec_values, linewidth=1, linestyle='solid', 
                       label=f"{rec['track_name']}", color=cmap(i+1))
                ax.fill(angles, rec_values, alpha=0.1, color=cmap(i+1))
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title(f"Audio Features Comparison", size=16)
        
        return fig
    
    def get_cluster_analysis(self):
        """
        Get detailed analysis of clusters.
        
        Returns:
            tuple: (str, dict) Cluster analysis text and cluster data dictionary
        """
        if self.song_cluster_labels is None:
            return "Clusters not created. Please prepare features for clustering first.", None
        
        analysis = []
        analysis.append("=== CLUSTER ANALYSIS ===\n")
        
        # Get mean feature values for each cluster
        cluster_means = self.data.groupby('cluster_label')[self.features].mean()
        
        # Get most popular songs in each cluster
        top_songs_by_cluster = {}
        for cluster in range(len(cluster_means)):
            cluster_songs = self.data[self.data['cluster_label'] == cluster].sort_values('popularity', ascending=False).head(3)
            top_songs_by_cluster[cluster] = cluster_songs[['track_name', 'artists', 'popularity']]
        
        # Store cluster data for visualization
        cluster_data = {
            'means': cluster_means,
            'top_songs': top_songs_by_cluster
        }
        
        # Generate analysis for each cluster
        for cluster in range(len(cluster_means)):
            analysis.append(f"CLUSTER {cluster}:")
            
            # Get dominant features
            cluster_profile = cluster_means.loc[cluster]
            dominant_features = cluster_profile[cluster_profile > cluster_profile.mean() + 0.5*cluster_profile.std()].index.tolist()
            weak_features = cluster_profile[cluster_profile < cluster_profile.mean() - 0.5*cluster_profile.std()].index.tolist()
            
            if dominant_features:
                analysis.append(f"  Dominant features: {', '.join(dominant_features)}")
            if weak_features:
                analysis.append(f"  Weak features: {', '.join(weak_features)}")
            
            # Describe the cluster
            description = "  Cluster character: "
            if 'energy' in dominant_features and 'danceability' in dominant_features:
                description += "Energetic and danceable. "
            elif 'energy' in dominant_features:
                description += "Energetic. "
            elif 'danceability' in dominant_features:
                description += "Danceable. "
                
            if 'acousticness' in dominant_features:
                description += "Acoustic. "
            if 'instrumentalness' in dominant_features:
                description += "Instrumental. "
            if 'valence' in dominant_features:
                description += "Positive and upbeat. "
            elif 'valence' in weak_features:
                description += "Negative or sad. "
                
            if 'speechiness' in dominant_features:
                description += "Speech-like (possibly rap or spoken word). "
                
            analysis.append(description)
            
            # Top songs in cluster
            analysis.append("  Top songs in this cluster:")
            for _, row in top_songs_by_cluster[cluster].iterrows():
                analysis.append(f"    - {row['track_name']} by {row['artists']} (Popularity: {row['popularity']})")
            
            analysis.append("")  # Empty line between clusters
        
        return "\n".join(analysis), cluster_data
    
    def find_similar_songs(self, song_name, artist_name=None, n_recommendations=5):
        """
        Find songs similar to a given song based on audio features.
        
        Args:
            song_name (str): Name of the song to find recommendations for
            artist_name (str, optional): Artist name to disambiguate songs with the same title
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            tuple: (pd.DataFrame, str) DataFrame containing recommended songs and message
        """
        if self.song_cluster_labels is None:
            return None, "Clusters not created. Please prepare features for clustering first."
        
        # Find the song in the dataset
        if artist_name:
            song_data = self.data[(self.data['track_name'].str.lower() == song_name.lower()) & 
                                 (self.data['artists'].str.lower().str.contains(artist_name.lower()))]
        else:
            song_data = self.data[self.data['track_name'].str.lower() == song_name.lower()]
        
        if song_data.empty:
            return None, f"Song '{song_name}' not found in the dataset."
        
        if len(song_data) > 1:
            if artist_name is None:
                artists_list = [f"- {row['artists']}" for _, row in song_data.iterrows()]
                return None, f"Multiple songs found with title '{song_name}'. Please specify an artist.\nAvailable artists for this song:\n" + "\n".join(artists_list)
            else:
                message = f"Multiple songs found with title '{song_name}' by artist containing '{artist_name}'.\nUsing the first match."
        else:
            message = ""
        
        # Get the first matching song
        song_row = song_data.iloc[0]
        
        # Get the cluster label for this song
        cluster_label = song_row['cluster_label']
        
        # Find songs in the same cluster
        same_cluster_songs = self.data[self.data['cluster_label'] == cluster_label]
        
        # Remove the input song from recommendations
        same_cluster_songs = same_cluster_songs[same_cluster_songs.index != song_row.name]
        
        # Calculate feature vector for the input song
        song_features = song_row[self.features].values.reshape(1, -1)
        
        # Calculate feature vectors for all songs in the same cluster
        cluster_features = same_cluster_songs[self.features].values
        
        # Standardize features
        scaler = self.cluster_pipeline.named_steps['scaler']
        song_features_scaled = scaler.transform(song_features)
        cluster_features_scaled = scaler.transform(cluster_features)
        
        # Calculate distances
        distances = euclidean_distances(song_features_scaled, cluster_features_scaled)[0]
        
        # Add distances to the dataframe
        same_cluster_songs = same_cluster_songs.copy()
        same_cluster_songs['distance'] = distances
        
        # Sort by distance
        recommendations = same_cluster_songs.sort_values('distance').head(n_recommendations)
        
        # Select columns for output
        output_columns = ['track_name', 'artists', 'album_name', 'popularity', 'distance'] + self.features
        available_columns = [col for col in output_columns if col in recommendations.columns]
        
        return recommendations[available_columns], message


def display_recommendations(recommendations, song_name, input_song_data=None):
    """
    Display song recommendations in a formatted way and create visualizations.
    
    Args:
        recommendations (pd.DataFrame): DataFrame containing recommended songs
        song_name (str): Name of the song recommendations are based on
        input_song_data (pd.Series, optional): Data for the input song
        
    Returns:
        matplotlib.figure.Figure: Radar chart comparing input song with recommendations
    """
    print(f"\nFound {len(recommendations)} recommendations for '{song_name}'")
    
    # Create radar chart if input song data is provided
    if input_song_data is not None:
        radar_chart = SpotifySongRecommender().get_radar_chart(input_song_data, recommendations.head(5))
        return radar_chart
    
    return None


def main():
    """Main function to run the Spotify Song Recommender console application with visualizations."""
    print("Loading Spotify dataset...")
    
    # Initialize the recommender
    recommender = SpotifySongRecommender()
    
    # Step 1: Load data
    success, message = recommender.load_data()
    
    if not success:
        print(f"Error: {message}")
        return
    
    print("Data loaded successfully!")
    
    # Step 2: Prepare features for clustering
    print("\nPreparing features for clustering...")
    success, message = recommender.prepare_features_for_clustering()
    
    if not success:
        print(f"Error: {message}")
        return
    
    # Step 3: Show cluster visualization with PCA
    print("\nGenerating cluster visualization (PCA)...")
    cluster_fig = recommender.get_cluster_visualization()
    plt.show()
    
    # Step 4: Show cluster visualization with t-SNE
    print("\nGenerating cluster visualization (t-SNE)...")
    tsne_fig = recommender.get_tsne_visualization()
    plt.show()
    
    # Step 5: Show cluster characteristics
    print("\nGenerating cluster characteristics visualization...")
    char_fig = recommender.get_cluster_characteristics_plot()
    plt.show()
    
    # Step 5: Song recommendation loop
    while True:
        print("\n" + "="*60)
        print("SONG RECOMMENDATIONS".center(60))
        print("="*60)
        
        # Get song name
        song_name = input("\nEnter a song name (or 'quit' to exit): ")
        if song_name.lower() == 'quit':
            break
        
        # Get artist name (optional)
        artist_name = input("Enter artist name (optional, press Enter to skip): ")
        if not artist_name:
            artist_name = None
        
        # Get number of recommendations
        try:
            n_recommendations = int(input("Number of recommendations (default: 5): ") or "5")
        except ValueError:
            n_recommendations = 5
        
        # Find similar songs
        print(f"\nFinding recommendations for '{song_name}'...")
        recommendations, message = recommender.find_similar_songs(song_name, artist_name, n_recommendations)
        
        if message:
            print(message)
        
        if recommendations is not None and not recommendations.empty:
            # Get the input song data for visualization
            if artist_name:
                input_song_data = recommender.data[(recommender.data['track_name'].str.lower() == song_name.lower()) & 
                                                 (recommender.data['artists'].str.lower().str.contains(artist_name.lower()))].iloc[0]
            else:
                input_song_data = recommender.data[recommender.data['track_name'].str.lower() == song_name.lower()].iloc[0]
            
            # Display recommendations with visualization
            radar_fig = display_recommendations(recommendations, song_name, input_song_data)
            
            # Show the radar chart
            if radar_fig:
                plt.show()
                
            # Print detailed recommendations
            print("\nDetailed recommendations:")
            for idx, row in recommendations.iterrows():
                print(f"{idx+1}. {row['track_name']} by {row['artists']}")
                print(f"   Album: {row['album_name'] if 'album_name' in row else 'N/A'}")
                print(f"   Popularity: {row['popularity']}/100")
                print(f"   Similarity score: {1/(1+row['distance']):.2f}")
                print()
        else:
            if not message:
                print("No recommendations found.")
        
        # Ask if user wants to continue
        continue_choice = input("\nDo you want to get more recommendations? (y/n): ")
        if continue_choice.lower() != 'y':
            break
    
    print("\nThank you for using the Spotify Song Recommender!")


if __name__ == "__main__":
    main()
