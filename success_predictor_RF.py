import opendatasets as od
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import LabelEncoder

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

# Plot Distribution of Popularity Scores
plt.figure()
sns.histplot(df['popularity'], bins=50, color='green')
plt.title('Distribution of Track Popularity')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()
numeric_df = df.select_dtypes(include=[np.number])

# Train-test split
X = numeric_df.drop(columns=['popularity'])
y = numeric_df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
#model = RandomForestRegressor(n_estimators=100, random_state=42)
#model.fit(X_train, y_train)
#joblib.dump(model, 'spotify_rf_model.pkl')

# Load saved model
model = joblib.load('spotify_rf_model.pkl')
print("Model loaded from file.")

# Test Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)

test_bins = pd.qcut(y_test, q=5, labels=False)  # 5 popularity groups
errors = np.abs(y_test - y_pred)

# Determine most important features
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
plt.title('Top 15 Most Influential Features')
plt.tight_layout()
plt.show()

# Find most successful and least successful genres
base_input = X_train.mean().to_frame().T  # Inputs are average of train data

# Predict popularity per genre
genre_preds = []
for genre_code in np.unique(X['track_genre']):
    temp_input = base_input.copy()
    temp_input['track_genre'] = genre_code
    pred_popularity = model.predict(temp_input)[0]
    genre_name = label_encoder.inverse_transform([genre_code])[0]
    genre_preds.append((genre_name, pred_popularity))

# Sort by predicted popularity
genre_preds_sorted = sorted(genre_preds, key=lambda x: x[1], reverse=True)

# Extract top 3 and bottom 3
top_3 = genre_preds_sorted[:3]
bottom_3 = genre_preds_sorted[-3:]
highlight_genres = top_3 + bottom_3
genres, preds = zip(*highlight_genres)

plot_df = pd.DataFrame({'Genre': genres, 'Predicted Popularity': preds})
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, x='Predicted Popularity', y='Genre', palette='viridis')
plt.xlabel("Predicted Popularity")
plt.title("Top 3 vs Bottom 3 Genres by Model-Predicted Popularity")
plt.tight_layout()
plt.show()
