import pandas as pd
import numpy as np
from keras.src.losses import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('solubility_data.csv')

# --- DATA CLEANING ---
# Ensure we don't have NaNs in our feature columns
cols_to_use = ['Final_LogP', 'MolecularWeight', 'RotatableBondCount', 'AromaticProportion']
df = df.dropna(subset=cols_to_use)

# NOTE: If you don't have a real solubility column (logS) yet,
# this line creates a dummy one so the code runs.
# Replace 'logS' with your actual target column name.
if 'logS' not in df.columns:
    print("Warning: 'logS' target column not found. Creating dummy data for demonstration.")
    df['logS'] = -0.6 * df['Final_LogP'] - 0.006 * df['MolecularWeight'] + np.random.normal(0, 0.5, len(df))

# 2. Define Features (X) and Target (y)
X = df[cols_to_use]
y = df['logS']

# 3. Split the data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")
print("-" * 30)
print("Coefficients:")
for col, coef in zip(cols_to_use, model.coef_):
    print(f"{col}: {coef:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Calculate Residuals
residuals = y_test - y_pred

# 2. Set up the plotting area (1 row, 2 columns)
plt.figure(figsize=(14, 6))

# --- Plot 1: Predicted vs. Actual ---
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='teal')
# Add a 45-degree line representing "Perfect Prediction"
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', lw=2)
plt.title('Predicted vs. Actual Solubility', fontsize=14)
plt.xlabel('Actual logS', fontsize=12)
plt.ylabel('Predicted logS', fontsize=12)
plt.grid(True, alpha=0.3)

# --- Plot 2: Residual Plot ---
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='darkorange')
# Add a horizontal line at zero
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.title('Residual Plot (Errors)', fontsize=14)
plt.xlabel('Predicted logS', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()