import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 1. Data Setup
data = {
    "Compound": ["Methane", "Propane", "Water", "Ethanol", "Formic Acid",
                 "Acetic Acid", "Butane", "Acetone", "Benzene", "Toluene", "Octane"],
    "MW": [16.04, 44.10, 18.02, 46.07, 46.03, 60.05, 58.12, 58.08, 78.11, 92.14, 114.23],
    "BP": [-161.5, -42.1, 100.0, 78.4, 100.8, 118.1, -0.5, 56.1, 80.1, 110.6, 125.7]
}
df = pd.DataFrame(data)

# Target compound to remove
target_name = "Ethanol"
train_df = df[df['Compound'] != target_name]
test_df = df[df['Compound'] == target_name]

X_train, y_train = train_df[['MW']], train_df['BP']
X_test, y_actual = test_df[['MW']], test_df['BP'].values[0]

# 2. Re-train Linear Regression
lr_model = LinearRegression().fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)[0]

# 3. Re-train MLP (with scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPRegressor(hidden_layer_sizes=(10, 10), solver='lbfgs', max_iter=5000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)[0]

# 4. Plotting & Saving
plt.figure(figsize=(10, 6))

# Plot the training points
plt.scatter(train_df['MW'], train_df['BP'], color='gray', alpha=0.5, label='Training Data')

# Plot the Actual vs Predicted for the missing compound
plt.scatter(X_test['MW'], y_actual, color='black', s=100, edgecolors='white', label=f'Actual {target_name}', zorder=5)
plt.scatter(X_test['MW'], lr_pred, color='blue', marker='x', s=100, label='LR Prediction')
plt.scatter(X_test['MW'], mlp_pred, color='red', marker='+', s=100, label='MLP Prediction')

# Aesthetics
plt.title(f'Boiling Point Prediction: {target_name} (Excluded from Training)')
plt.xlabel('Molecular Weight (MW)')
plt.ylabel('Boiling Point (BP) °C')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save to directory
plt.savefig('bp_prediction_results.png', dpi=300)
print(f"Plot saved as 'bp_prediction_results.png' in your current directory.")
plt.show()