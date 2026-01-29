import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error

# 1. Load data and keep a copy of original ages for comparison
titanic = sns.load_dataset('titanic')
original_age = titanic['age'].copy()

# 2. Prepare data for KNN (KNN requires numeric input)
# We'll use 'pclass', 'sibsp', 'parch', and 'fare' to predict 'age'
features = ['pclass', 'sibsp', 'parch', 'fare', 'age']
df_subset = titanic[features]

# 3. Apply KNN Imputation
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(df_subset)
df_imputed = pd.DataFrame(imputed_data, columns=features)

# Update the main dataframe
titanic['age_knn'] = df_imputed['age']

# 4. Calculate Mean Absolute Error (MAE)
# We can only calculate this for rows where we actually had an age to begin with
mask = original_age.notnull()
actual_age = original_age[mask]
predicted_age = titanic['age_knn'][mask]

mae = mean_absolute_error(actual_age, predicted_age)
print(f"Mean Absolute Error (MAE) for KNN Imputation: {mae:.2f}")

# 5. Plotting Actual vs Predicted (for the non-null rows)
plt.figure(figsize=(8, 6))
plt.scatter(actual_age, predicted_age, alpha=0.5, color='teal', edgecolor='white')
plt.plot([actual_age.min(), actual_age.max()], [actual_age.min(), actual_age.max()], 'r--', lw=2)
plt.title(f'Actual Age vs. KNN Predicted Age\n(MAE: {mae:.2f})')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.grid(True, linestyle='--', alpha=0.7)

# 6. Save Plot
output_dir = 'plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, 'knn_age_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()MakingDataWholeKNN.py