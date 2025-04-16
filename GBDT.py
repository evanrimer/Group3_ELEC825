import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Set random seed for reproducibility
RANDOM_STATE = 65
np.random.seed(RANDOM_STATE)

# Load the dataset
# Note: Replace 'your_dataset.csv' with your actual file path
df = pd.read_csv('spotify_data/dataset.csv')

# Drop the unnamed index column
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics for numerical features
print("\nBasic statistics:")
print(df.describe())

# Drop songs with zero popularity
zero_popularity_count = len(df[df['popularity'] == 5])
df = df[df['popularity'] > 5]
print(f"Removed {zero_popularity_count} songs with zero popularity")
print(f"Remaining dataset size: {len(df)}")

# Remove outliers
def remove_outliers(df, columns, threshold=3):
    """Remove outliers based on z-score."""
    df_clean = df.copy()
    for col in columns:
        if df_clean[col].dtype in [np.float64, np.int64]:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    print(f"Removed {len(df) - len(df_clean)} outliers")
    return df_clean

# Apply to numerical features
numerical_cols_to_check = ['duration_ms', 'danceability', 'energy', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness',
                           'liveness', 'valence', 'tempo']
df = remove_outliers(df, numerical_cols_to_check)

# Log-transform highly skewed features
skewed_features = ['duration_ms', 'instrumentalness', 'acousticness']
for feature in skewed_features:
    # Add small constant to handle zeros
    df[f'{feature}_log'] = np.log1p(df[feature] + 1e-5)

# Feature interactions
df['energy_valence'] = df['energy'] * df['valence']
df['danceability_tempo'] = df['danceability'] * df['tempo'] / 100
df['loudness_energy_ratio'] = df['loudness'] / (df['energy'] + 1e-5)

# Bin popularity into categories for stratified sampling
df['popularity_bins'] = pd.qcut(df['popularity'], q=5, labels=False)

# Data preprocessing
# Drop irrelevant columns
df = df.drop(['track_id', 'track_name', 'album_name'], axis=1)

# Check the distribution of the target variable after filtering
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], kde=True)
plt.title('Distribution of Song Popularity (After Removing Zeros)')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

# Explore correlations between features and target
plt.figure(figsize=(12, 10))
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
correlation = df[numerical_features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Prepare features and target
# Remove popularity_bins before modeling as it's derived from the target
X = df.drop(['popularity', 'popularity_bins'], axis=1)
y = df['popularity']

# Identify categorical columns - CatBoost will handle these directly
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Convert categorical columns to string type to ensure CatBoost recognizes them
for col in categorical_cols:
    X[col] = X[col].astype(str)

# Get categorical indices for CatBoost
cat_indices = [X.columns.get_loc(col) for col in categorical_cols]

# Split the data into train, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE)

# Create CatBoost Pools - much more efficient for CatBoost
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_indices)
val_pool = Pool(data=X_val, label=y_val, cat_features=cat_indices)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_indices)

# Base CatBoost model
base_catboost = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    random_seed=RANDOM_STATE,
    verbose=50
)

# Define parameters for hyperparameter tuning
param_dist = {
    'iterations': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'depth': randint(4, 10),
    'l2_leaf_reg': uniform(1, 10),
    'bagging_temperature': uniform(0, 1),
    'random_strength': uniform(0, 1),
    'one_hot_max_size': randint(10, 25),
    'leaf_estimation_iterations': randint(1, 10)
}

# Set up RandomizedSearchCV for CatBoost
print("Performing hyperparameter tuning...")
catboost_random_search = RandomizedSearchCV(
    estimator=base_catboost,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Fit the hyperparameter search (CatBoost will handle categorical variables automatically)
catboost_random_search.fit(X_train, y_train, cat_features=cat_indices)

# Get best model
best_model = catboost_random_search.best_estimator_
print(f"Best parameters: {catboost_random_search.best_params_}")
print(f"Best RMSE: {-catboost_random_search.best_score_:.4f}")

# Print all tested parameters and their scores for reference
print("\nAll hyperparameter combinations tried:")
for i, params in enumerate(catboost_random_search.cv_results_['params']):
    score = -catboost_random_search.cv_results_['mean_test_score'][i]
    print(f"Combination {i+1}: RMSE={score:.4f}, Parameters: {params}")

# Evaluate on validation set
val_pred = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# Make predictions on test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs Predicted Popularity')
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='--')
plt.xlabel('Predicted Popularity')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Add error analysis
error_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'error': np.abs(y_test - y_pred)
})

# Identify high error regions
high_error = error_df[error_df['error'] > error_df['error'].quantile(0.75)]
low_error = error_df[error_df['error'] < error_df['error'].quantile(0.25)]

print("\nError Analysis:")
print(f"Average error on worst 25% predictions: {high_error['error'].mean():.2f}")
print(f"Average error on best 25% predictions: {low_error['error'].mean():.2f}")

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(error_df['error'], bins=20)
plt.axvline(error_df['error'].mean(), color='r', linestyle='--', label=f'Mean Error: {error_df["error"].mean():.2f}')
plt.title('Error Distribution')
plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.legend()
plt.show()

# Error by popularity range
plt.figure(figsize=(10, 6))
error_df['popularity_bin'] = pd.cut(error_df['actual'], bins=10)
error_by_pop = error_df.groupby('popularity_bin')['error'].mean().reset_index()
plt.bar(error_by_pop['popularity_bin'].astype(str), error_by_pop['error'])
plt.title('Average Error by Popularity Range')
plt.xlabel('Popularity Range')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature importance for CatBoost
feature_importances = best_model.get_feature_importance()
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)[-15:]
plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances (CatBoost)')
plt.tight_layout()
plt.show()

# Learning curves - CatBoost specific visualization if available
if hasattr(best_model, 'evals_result_'):
    train_curve = best_model.evals_result_['learn']['RMSE']
    plt.figure(figsize=(10, 6))
    plt.plot(train_curve, label='Training')
    if 'validation' in best_model.evals_result_:
        val_curve = best_model.evals_result_['validation']['RMSE']
        plt.plot(val_curve, label='Validation')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# Train with decreasing learning rate - FIXED PARAMETERS
model_with_schedule = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=best_model.get_params()['depth'],
    loss_function='RMSE',
    random_seed=RANDOM_STATE,
    verbose=50,
    od_type='Iter',
    od_wait=50,
    boost_from_average=True,
    boosting_type='Ordered',
    auto_class_weights='Balanced'
)

# Use this for learning rate decay
lr_callback = lambda iter_num, lr: lr * 0.95 if iter_num % 50 == 0 else lr

# Train with evaluation set
model_with_schedule.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_indices,
    use_best_model=True,
    callbacks=[{'type': 'learning_rate_scheduler', 'schedule': lr_callback}]
)

# Evaluate scheduled model
schedule_pred = model_with_schedule.predict(X_test)
schedule_rmse = np.sqrt(mean_squared_error(y_test, schedule_pred))
schedule_r2 = r2_score(y_test, schedule_pred)
print(f"\nScheduled Learning Rate Model:")
print(f"RMSE: {schedule_rmse:.4f}")
print(f"R² Score: {schedule_r2:.4f}")

# Compare with original best model
print(f"\nOriginal Best Model:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


