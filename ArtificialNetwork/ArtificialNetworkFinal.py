import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Data Setup
data = {
    "Compound": ["Methane", "Propane", "Water", "Ethanol", "Formic Acid",
                 "Acetic Acid", "Butane", "Acetone", "Benzene", "Toluene", "Octane"],
    "MW": [16.04, 44.10, 18.02, 46.07, 46.03, 60.05, 58.12, 58.08, 78.11, 92.14, 114.23],
    "BP": [-161.5, -42.1, 100.0, 78.4, 100.8, 118.1, -0.5, 56.1, 80.1, 110.6, 125.7]
}
df = pd.DataFrame(data)

# Leave Ethanol out
target_name = "Ethanol"
train_df = df[df['Compound'] != target_name]
X_train, y_train = train_df[['MW']], train_df['BP']

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 2. USER INPUT SECTION ---
print("--- Neural Network Configuration ---")
try:
    layers = int(input("Enter number of layers (1-4): "))
    neurons = int(input("Enter neurons per layer (1-10): "))

    # Validation and clipping
    layers = max(1, min(4, layers))
    neurons = max(1, min(10, neurons))
except ValueError:
    print("Invalid input. Defaulting to 1 layer, 5 neurons.")
    layers, neurons = 1, 5

user_config = tuple([neurons] * layers)
orig_config = (10, 10)


# 3. Train Both Models
def train_mlp(config):
    model = MLPRegressor(
        hidden_layer_sizes=config,
        activation='relu',
        solver='lbfgs',
        max_iter=5000,
        random_state=42
    )
    return model.fit(X_train_scaled, y_train)


mlp_orig = train_mlp(orig_config)
mlp_user = train_mlp(user_config)


# 4. Metrics Table
def print_metrics(name, model, config):
    pred = model.predict(X_train_scaled)
    mae = mean_absolute_error(y_train, pred)
    r2 = r2_score(y_train, pred)
    print(f"{name} {config}: MAE={mae:.2f}, R²={r2:.4f}, Epochs={model.n_iter_}")


print("\n--- Model Comparison Results ---")
print_metrics("Original Model", mlp_orig, orig_config)
print_metrics("User Model    ", mlp_user, user_config)

# 5. Visualization
plt.figure(figsize=(12, 7))
mw_range = np.linspace(-25, 125, 300).reshape(-1, 1)
mw_range_scaled = scaler.transform(mw_range)
plt.ylim(-175, df['BP'].max() + 20)

# Plot lines
plt.plot(mw_range, mlp_orig.predict(mw_range_scaled), label=f'Original MLP {orig_config}', color='blue', linestyle='--')
plt.plot(mw_range, mlp_user.predict(mw_range_scaled), label=f'User MLP {user_config}', color='green', linewidth=2)

# Plot data
plt.scatter(train_df['MW'], train_df['BP'], color='black', label='Training Data')
plt.scatter(df[df['Compound'] == target_name]['MW'], df[df['Compound'] == target_name]['BP'],
            color='red', marker='*', s=200, label=f'Excluded: {target_name}')

plt.title(f'Neural Network Architecture Comparison')
plt.xlabel('Molecular Weight')
plt.ylabel('Boiling Point (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('neuralnet_comparison.png')
plt.show()