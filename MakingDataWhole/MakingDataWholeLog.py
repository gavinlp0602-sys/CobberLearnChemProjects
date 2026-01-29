import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error

# 1. Load data
titanic = sns.load_dataset('titanic')
# Use a copy to avoid SettingWithCopy warnings
df_subset = titanic[['pclass', 'sibsp', 'parch', 'fare', 'age']].copy()

# 2. Log-Transform the Age
# Note: Age 0 (infants) would cause -inf, so we use log1p (log(1+x)) for stability
df_subset['age_log'] = np.log1p(df_subset['age'])

# 3. Apply Log-Linear Imputation
# We use BayesianRidge as the estimator for a standard linear approach
log_imputer = IterativeImputer(
    estimator=BayesianRidge(),
    random_state=42
)

# Impute using the log-age instead of raw age
# We drop the original 'age' for the imputation step so the model only sees the log version
impute_cols = ['pclass', 'sibsp', 'parch', 'fare', 'age_log']
imputed_data = log_imputer.fit_transform(df_subset[impute_cols])
df_imputed = pd.DataFrame(imputed_data, columns=impute_cols)

# 4. Inverse Transform back to original scale
titanic['age_log_linear'] = np.expm1(df_imputed['age_log'])

# 5. Calculate Stats
avg_before = titanic['age'].mean()
avg_after = titanic['age_log_linear'].mean()
mask = titanic['age'].notnull()
mae = mean_absolute_error(titanic.loc[mask, 'age'], titanic.loc[mask, 'age_log_linear'])

print(f"Avg Age Before: {avg_before:.2f}")
print(f"Avg Age After (Log-Linear): {avg_after:.2f}")
print(f"MAE: {mae:.2f}")

# 6. Visualization
plt.figure(figsize=(10, 6))
sns.kdeplot(titanic['age'], label='Original', fill=True, color="gray", alpha=0.3)
sns.kdeplot(titanic['age_log_linear'], label='Log-Linear Imputed', color="darkorange", lw=2)
plt.title('Age Distribution: Log-Linear Imputation')
plt.legend()

# Save
output_dir = 'plots'
if not os.path.exists(output_dir): os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'log_linear_age.png'))
plt.show()