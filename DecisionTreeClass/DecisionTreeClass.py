import pandas as pd

# Data for the molecules
data = {
    'Molecule': [f'Molecule {i}' for i in range(1, 13)],
    'Molecular Weight': [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    'Hydrogen Bond Donors': [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    'Hydrogen Bond Acceptors': [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    'Water Solubility': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set the Molecule name as the index for a cleaner look
df.set_index('Molecule', inplace=True)

print(df)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Split into input features (X) and target variable (y)
X = df[['Hydrogen Bond Acceptors']]
y = df['Water Solubility']

# 2. Initialize and train the Decision Tree
# We'll use 'entropy' or 'gini' to measure the quality of the split
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X, y)

# 3. Visualize the Tree
plt.figure(figsize=(18, 13), dpi=100)
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Insoluble', 'Soluble'],
    filled=True,
    rounded=True,
    fontsize=18,
    precision=1 # Reduces 180.000 to 180.0
)
plt.title("Molecular Solubility Decision Tree", fontsize=16)

# Save as PNG
plt.savefig('solubility_tree.png', dpi=300, bbox_inches='tight')

# Save tree
import os

# Create directory if it doesn't exist
output_dir = "model_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the plot to that specific folder
plt.savefig(os.path.join(output_dir, 'solubility_tree.png'), dpi=300)

plt.show()