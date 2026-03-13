import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
train_df = df[df['Compound'] != target_name].copy()
test_df = df[df['Compound'] == target_name].copy()

X_train, y_train = train_df[['MW']], train_df['BP']
X_test, y_actual = test_df[['MW']], test_df['BP'].values[0]

# 2. Train Models
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# MLP Regressor (Neural Network)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
mlp_model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)

# 3. Generate Predictions and Metrics
y_train_pred_lr = lr_model.predict(X_train)
y_train_pred_mlp = mlp_model.predict(X_train_scaled)

lr_mae = mean_absolute_error(y_train, y_train_pred_lr)
lr_mse = mean_squared_error(y_train, y_train_pred_lr)
lr_r2 = r2_score(y_train, y_train_pred_lr)

mlp_mae = mean_absolute_error(y_train, y_train_pred_mlp)
mlp_mse = mean_squared_error(y_train, y_train_pred_mlp)
mlp_r2 = r2_score(y_train, y_train_pred_mlp)
mlp_epochs = mlp_model.n_iter_

# Prediction for the missing compound
lr_test_pred = lr_model.predict(X_test)[0]
X_test_scaled = scaler.transform(X_test)
mlp_test_pred = mlp_model.predict(X_test_scaled)[0]

# 4. Plot 1: Model Comparison (BP vs MW)
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Smooth lines for models
mw_range = np.linspace(df['MW'].min() - 5, df['MW'].max() + 5, 200).reshape(-1, 1)
lr_line = lr_model.predict(mw_range)
mw_range_scaled = scaler.transform(mw_range)
mlp_line = mlp_model.predict(mw_range_scaled)

ax1.scatter(train_df['MW'], train_df['BP'], color='black', label='Training Data', zorder=3)
ax1.scatter(X_test['MW'], y_actual, color='red', marker='*', s=200, label=f'Actual {target_name} (Excluded)', zorder=5)
ax1.plot(mw_range, lr_line, color='blue', linestyle='--', label='Linear Regression')
ax1.plot(mw_range, mlp_line, color='green', linewidth=2, label='MLP (ReLU)')

ax1.set_title(f'Boiling Point Prediction Models ({target_name} Excluded)')
ax1.set_xlabel('Molecular Weight')
ax1.set_ylabel('Boiling Point (°C)')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)
plt.savefig('model_comparison.png', dpi=300)

# 5. Plot 2: ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

x_relu = np.linspace(-5, 5, 200)
y_relu = relu(x_relu)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(x_relu, y_relu, color='purple', linewidth=3)
ax2.set_title('ReLU Activation Function: $f(x) = \max(0, x)$')
ax2.set_xlabel('Input ($x$)')
ax2.set_ylabel('Output ($f(x)$)')
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)
ax2.grid(True, linestyle='--', alpha=0.5)
plt.savefig('relu_activation.png', dpi=300)

# Print Summary
print("--- Model Performance Metrics (Training Set) ---")
print(f"Linear Regression: MAE={lr_mae:.2f}, MSE={lr_mse:.2f}, R2={lr_r2:.4f}")
print(f"MLP (Neural Net):  MAE={mlp_mae:.2f}, MSE={mlp_mse:.2f}, R2={mlp_r2:.4f}")
print(f"MLP Epochs to Convergence: {mlp_epochs}")
print(f"\n--- Prediction for {target_name} ---")
print(f"Actual: {y_actual}°C")
print(f"LR Prediction: {lr_test_pred:.2f}°C (Error: {abs(y_actual - lr_test_pred):.2f})")
print(f"MLP Prediction: {mlp_test_pred:.2f}°C (Error: {abs(y_actual - mlp_test_pred):.2f})")