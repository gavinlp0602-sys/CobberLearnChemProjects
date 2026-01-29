import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error

# 1. Load data
titanic = sns.load_dataset('titanic')
original_age = titanic['age'].copy()

# 2. Prepare data for Random Forest
features = ['pclass', 'sibsp', 'parch', 'fare', 'age']
df_subset = titanic[features]

# 3. Apply Random Forest Imputation
rf_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    max_iter=10,
    random_state=42
)

imputed_data = rf_imputer.fit_transform(df_subset)
df_imputed = pd.DataFrame(imputed_data, columns=features)
titanic['age_rf'] = df_imputed['age']

# 4. Print Average Age Comparison
avg_before = original_age.mean()
avg_after = titanic['age_rf'].mean()

print(f"Average Age (Before Imputation): {avg_before:.2f}")
print(f"Average Age (After RF Imputation): {avg_after:.2f}")
print(f"Difference: {avg_after - avg_before:.4f}")

# 5. Calculate MAE for known values
mask = original_age.notnull()
mae = mean_absolute_error(original_age[mask], titanic['age_rf'][mask])
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 6. Visualization: Distribution Comparison
plt.figure(figsize=(10, 6))
sns.kdeplot(original_age, label='Original Age (with NaNs)', fill=True, color="gray", alpha=0.3)
sns.kdeplot(titanic['age_rf'], label='Imputed Age (Random Forest)', color="royalblue", lw=2)
plt.title('Age Distribution: Original vs. Random Forest Imputation')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()

# 7. Save and Show
output_dir = 'plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, 'age_distribution_comparison.png'), dpi=300)
plt.show()