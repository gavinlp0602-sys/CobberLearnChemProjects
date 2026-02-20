import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- 1. DATA GENERATION & MODEL (Same as before) ---
x = np.linspace(0, 10, 100)
true_y = 2 * x + 5
y_noisy = true_y + np.random.normal(0, 3.5, size=x.shape)

# Fit model once to find the "optimal" point for the map
model = LinearRegression().fit(x.reshape(-1, 1), y_noisy)
best_m, best_b = model.coef_[0], model.intercept_

# --- 2. GRID SEARCH CALCULATION ---
# Define a range of slopes (m) and intercepts (b) to test
m_range = np.linspace(0, 4, 50)  # Testing slopes from 0 to 4
b_range = np.linspace(0, 10, 50) # Testing intercepts from 0 to 10

# Create a grid (coordinate matrix)
M, B = np.meshgrid(m_range, b_range)

# Vectorized MSE calculation across the whole grid
# We expand x and y to match the grid shape for a speed boost
Z_mse = np.zeros(M.shape)
for i in range(len(m_range)):
    for j in range(len(b_range)):
        y_pred = m_range[i] * x + b_range[j]
        Z_mse[j, i] = np.mean((y_noisy - y_pred)**2)

# --- 3. VISUALIZING THE LOSS LANDSCAPE ---
fig = plt.figure(figsize=(12, 8))

# CHANGE 1: Use 'viridis_r' (reversed) so low values = yellow, high = purple
# CHANGE 2: Use plt.contourf for a 2D filled contour plot
contour = plt.contourf(M, B, Z_mse, levels=50, cmap='viridis_r')

# Add a colorbar to show what the colors represent
plt.colorbar(contour, label='Mean Squared Error (MSE)')

# Mark the "Global Minimum" (the point the model finds)
plt.scatter(best_m, best_b, color='red', marker='x', s=100, label='Optimal Parameters')

plt.title('2D Loss Landscape: MSE vs. Parameters')
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')
plt.legend()
plt.grid(alpha=0.3)


# Print the results
print(f"Model Slope (Coefficient):    {model.coef_[0]:.4f}")
print(f"Model Intercept:             {model.intercept_:.4f}")

print("-" * 30)

# Reminder of what we started with: y = 2x + 5
print(f"True Slope:                  2.0000")
print(f"True Intercept:               5.0000")


def calculate_mse(x, y_actual, slope, intercept):
    """
    Calculates MSE for a linear model given a slope and intercept.
    """
    # 1. Calculate the predicted y values based on the inputs
    y_pred = (slope * x) + intercept

    # 2. Calculate the squared differences
    squared_errors = (y_actual - y_pred) ** 2

    # 3. Return the mean of those errors
    mse = np.mean(squared_errors)
    return mse


# Example usage with your model's results:
current_mse = calculate_mse(x, y_noisy, model.coef_[0], model.intercept_)
print(f"The Mean Squared Error is: {current_mse:.4f}")

plt.show()