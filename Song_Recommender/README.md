# Spotify Song Recommender

A machine learning-based song recommendation system that uses clustering techniques to find similar songs based on audio features.

## Overview

This application analyzes Spotify track data to group similar songs together using K-means clustering. It then provides personalized song recommendations based on a user's input song, along with visualizations to help understand the recommendations.

## Features

- **Cluster-Based Recommendations**: Groups similar songs together based on audio features
- **Interactive Console Interface**: Simple text-based interface for getting recommendations
- **Rich Visualizations**:
  - Cluster visualization using PCA
  - Cluster characteristics heatmap
  - Radar charts comparing recommended songs with the input song

## How It Works

### 1. Data Processing

The system uses the Spotify tracks dataset (`tracks.csv`) which contains information about songs including:
- Basic metadata (track name, artist, album, popularity)
- Audio features (danceability, energy, tempo, etc.)

### 2. Feature Selection

Nine key audio features are selected for clustering:
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence (musical positiveness)
- Tempo

### 3. Clustering Algorithm

The system uses K-means clustering with 10 clusters:

1. **Standardization**: Features are standardized using `StandardScaler()` to ensure all features contribute equally regardless of their original scale.

2. **K-means Clustering**: The algorithm:
   - Randomly initializes 10 cluster centers in the feature space
   - Assigns each song to the nearest cluster center
   - Recalculates the cluster centers based on the mean of all songs in each cluster
   - Repeats until convergence (when the cluster assignments stop changing)

3. **Why 10 Clusters?**
   - Provides a balance between granularity and interpretability
   - Creates meaningful groups without becoming too fragmented
   - Ensures enough songs per cluster for good recommendations

### 4. Recommendation Process

When a user requests recommendations:

1. The system finds the input song in the dataset
2. Identifies which cluster the song belongs to
3. Finds other songs in the same cluster
4. Calculates similarity based on Euclidean distance in the feature space
5. Returns the most similar songs as recommendations

### 5. Visualizations

The system provides four types of visualizations:

1. **Cluster Visualization (PCA)**
   - Reduces the 9-dimensional feature space to 2 dimensions using Principal Component Analysis
   - Shows how songs are grouped in the feature space
   - Different colors represent different clusters
   - Focuses on preserving global structure and variance

2. **Cluster Visualization (t-SNE)**
   - Alternative dimensionality reduction technique using t-Distributed Stochastic Neighbor Embedding
   - Better at preserving local structure and cluster separation
   - Often shows more distinct cluster boundaries than PCA
   - Particularly useful for visualizing complex, non-linear relationships in the data

3. **Cluster Characteristics**
   - Heatmap showing the standardized feature values for each cluster
   - Helps understand what makes each cluster unique

4. **Radar Chart**
   - Compares the audio features of the input song with recommended songs
   - Shows strengths and weaknesses across different audio features

## Example

```
Enter a song name: Shape of You
Enter artist name (optional): Ed Sheeran
Number of recommendations (default: 5): 5

Found 5 recommendations for 'Shape of You'

Detailed recommendations:
1. Castle on the Hill by Ed Sheeran
   Album: ÷ (Divide)
   Popularity: 84/100
   Similarity score: 0.87

2. Something Just Like This by The Chainsmokers
   Album: Memories...Do Not Open
   Popularity: 86/100
   Similarity score: 0.82
