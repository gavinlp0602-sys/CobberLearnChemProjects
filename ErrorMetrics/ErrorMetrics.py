import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Creating the arrays
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])
residuals = predicted-actual

# Quick check
print(f"Actual values:    {actual}")
print(f"Predicted values: {predicted}")
print(f"Residuals: {residuals}")

#MAE,MSE, R^2
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)

print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R-squared: {r2:.3f}")

# 2. Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, color='green', edgecolor='k', label='Predicted Points')

# Add the "Perfect Fit" line (where Actual == Predicted)
# We find the min and max values to ensure the line spans the data range
line_coords = [actual.min(), actual.max()]
plt.plot(line_coords, line_coords, color='red', linestyle='-', linewidth=2, label='Perfect Fit (y=x)')

# Add labels and styling
plt.title('Model Evaluation: Predicted vs. Actual', fontsize=14)
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Save or show the plot
output_dir = 'plots'
if not os.path.exists(output_dir): os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3 Residuals Plot
plt.figure(figsize=(8, 5))
plt.scatter(predicted, residuals, color='blue', edgecolor='k')

# Add a horizontal line at 0
plt.axhline(y=0, color='red', linestyle='-', linewidth=2)

# Styling
plt.title('Residual Plot (Errors vs. Predicted)', fontsize=14)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals (Error)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
plt.show()
