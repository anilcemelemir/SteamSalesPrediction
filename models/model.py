import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\anil_\Documents\GitHub\SteamSalesPrediction\data\steam.csv"
data = pd.read_csv(file_path)

# Convert the 'owners' column to numerical by averaging the range
data['owners_mean'] = data['owners'].str.split('-').apply(
    lambda x: (int(x[0]) + int(x[1])) // 2 if len(x) == 2 else 0
)

# Select relevant features for training
features = ['price', 'positive_ratings', 'negative_ratings', 'average_playtime', 'english',
            'categories', 'genres', 'steamspy_tags', 'platforms', 'achievements']
target = 'owners_mean'

# Fill missing data in new features (if any)
data.fillna('', inplace=True)

# Encode categorical features with simple counts
data['categories_count'] = data['categories'].apply(lambda x: len(x.split(';')))
data['genres_count'] = data['genres'].apply(lambda x: len(x.split(';')))
data['tags_count'] = data['steamspy_tags'].apply(lambda x: len(x.split(';')))
data['platforms_count'] = data['platforms'].apply(lambda x: len(x.split(';')))

# Add encoded features
X = data[['price', 'positive_ratings', 'negative_ratings', 'average_playtime', 'english',
          'categories_count', 'genres_count', 'tags_count', 'platforms_count', 'achievements']]
y = data[target]

# Split the dataset into training and testing sets (70%-30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Results summary
model_metrics = {
    "Mean Absolute Error (MAE)": mae,
    "Mean Squared Error (MSE)": mse,
    "R² Score": r2
}

# Print model accuracy metrics
print("Model Accuracy Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Display feature importance
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Actual vs Predicted Owners")
plt.xlabel("Actual Owners")
plt.ylabel("Predicted Owners")
plt.grid(True)
plt.show()

# Plot residuals (errors)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Distribution of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Function to predict game sales based on user input
def predict_sales_pre_release(price, platforms, genres, categories, tags, achievements):
    """
    Predict sales based on pre-release information.
    """
    # Preprocess input features for the model (e.g., encode platforms, genres)
    platform_score = len(platforms.split(";"))  # Count the number of platforms
    genre_score = len(genres.split(";"))  # Count the number of genres
    categories_score = len(categories.split(";"))  # Count the number of categories
    tags_score = len(tags.split(";"))  # Count the number of tags

    # Use simplified feature set for prediction
    input_data = pd.DataFrame([{
        'price': price,
        'positive_ratings': 1000,  # Example fixed value for demonstration
        'negative_ratings': 100,  # Example fixed value for demonstration
        'average_playtime': genre_score * 100,  # Dummy transformation for demonstration
        'english': platform_score,  # Number of platforms as a proxy for accessibility
        'categories_count': categories_score,
        'genres_count': genre_score,
        'tags_count': tags_score,
        'platforms_count': platform_score,
        'achievements': achievements
    }])

    # Predict sales
    predicted_sales = model.predict(input_data)
    print(f"Predicted Sales: {int(predicted_sales[0])}")
    return int(predicted_sales[0])

# Example pre-release input (replace with actual inputs)
example_pre_release_prediction = predict_sales_pre_release(
    price=29.99,
    platforms="windows;mac;linux",
    genres="Action;Adventure",
    categories="Multiplayer;Single-player",
    tags="Strategy;Indie",
    achievements=10
)

example_pre_release_prediction