import opendatasets as od
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                   activation='relu',
                   solver='adam',
                   alpha=0.001,
                   learning_rate='adaptive',
                   max_iter=1000,
                   early_stopping=True,
                   validation_fraction=0.1,
                   random_state=42)

# Download Spotify Tracks Dataset
dataset_url = 'https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data'
od.download(dataset_url)

# Get the data into pandas dataframe
data_dir = './-spotify-tracks-dataset'
os.listdir(data_dir)
spotify_csv = data_dir + '/dataset.csv'
df = pd.read_csv(spotify_csv, encoding='latin1')

# Remove unnecessary information
df = df.drop(columns=['Unnamed: 0', 'track_id', 'album_name', 'track_name'])

# Convert Features to Numbers
df['explicit'] = df['explicit'].astype(int)
df['mode'] = df['mode'].astype(int)
label_encoder = LabelEncoder()
df['track_genre'] = label_encoder.fit_transform(df['track_genre'])
label_encoder_artists = LabelEncoder()
df['artists'] = label_encoder_artists.fit_transform(df['artists'])

numeric_features = ['duration_ms', 'danceability', 'energy', 'key', 'loudness',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                    'valence', 'tempo', 'time_signature']
#

# Normalize numerical features
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Save the dataset
df.to_csv("processed_spotify_data.csv", index=False)

# Train-test split
numeric_df = df.select_dtypes(include=[np.number])
X = numeric_df.drop(columns=['popularity'])
y = numeric_df['popularity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
mlp.fit(X_train, y_train.ravel())

# Testing
y_pred = mlp.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MLP RMSE (original scale): {rmse}")