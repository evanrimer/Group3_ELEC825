from xgboost import XGBRegressor
import opendatasets as od
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

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
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Train-test split
numeric_df = df.select_dtypes(include=[np.number])
X = numeric_df.drop(columns=['popularity'])
scaler = MinMaxScaler()
y = scaler.fit_transform(numeric_df[['popularity']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and testing
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_test_original = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_original = scaler.inverse_transform(y_pred.reshape(-1,1))
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"RMSE (original scale): {rmse}")