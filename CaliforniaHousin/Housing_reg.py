import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
data = fetch_california_housing(as_frame=True)
df = data.frame

# --- VISUALIZATION SECTION ---

# 2. Create and Save Histograms for all features
# This helps identify data skewness and outliers
df.hist(figsize=(15, 10), bins=30, edgecolor='black', color='skyblue')
plt.suptitle('California Housing Feature Distributions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('housing_histograms.png')
plt.close()  # Close plot to free memory

# 3. Create and Save Correlation Heatmap
# This shows how features like 'MedInc' relate to 'MedHouseVal'
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# --- REGRESSION SECTION ---

# 4. Prepare Data
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train and Evaluate Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

print(f"{'Model':<20} | {'MSE':<10} | {'R2 Score':<10}")
print("-" * 45)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<20} | {mse:<10.4f} | {r2:<10.4f}")


import numpy as np

# 1. Define 'new' houses (3 examples with random realistic values)
# Features: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long]
new_houses = np.array([
    [8.5, 15, 7.0, 1.1, 1200, 3.0, 37.8, -122.2], # High income, newer house
    [3.2, 35, 5.2, 1.0, 800,  2.5, 34.0, -118.2], # Middle income, older house
    [1.5, 52, 4.1, 1.1, 1500, 3.2, 36.7, -119.8]  # Low income, very old house
])

# 2. Scale the new houses using the same scaler from training
new_houses_scaled = scaler.transform(new_houses)

# 3. Predict using the Random Forest model
rf_model = models["Random Forest"]
predictions = rf_model.predict(new_houses_scaled)

print("Predicted Prices (in $100k blocks):")
for i, price in enumerate(predictions):
    print(f"House {i+1}: ${price * 100000:,.2f}")

# 4. Extract and visualize feature importance
importances = rf_model.feature_importances_
feature_names = data.feature_names

# Create a DataFrame for easy viewing
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance Ranking:")
print(importance_df)

# Create a figure with a grid of subplots
plt.figure(figsize=(16, 12))

for i, (name, model) in enumerate(models.items()):
    # Generate predictions
    y_pred = model.predict(X_test_scaled)

    # Create subplot
    plt.subplot(2, 2, i + 1)

    # Plotting the data
    plt.scatter(y_test, y_pred, alpha=0.3, color='royalblue', s=10)

    # Plotting the "Perfect Prediction" line
    line_coords = [y_test.min(), y_test.max()]
    plt.plot(line_coords, line_coords, color='red', linestyle='--', linewidth=2)

    plt.title(f'{name}: Predicted vs Actual')
    plt.xlabel('Actual Price ($100k)')
    plt.ylabel('Predicted Price ($100k)')
    plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('predicted_vs_actual.png')
plt.show()