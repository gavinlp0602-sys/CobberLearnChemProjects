import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# 1. Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Initialize the K-Means model
# We choose n_clusters=3 because there are 3 species in the Iris dataset
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')

# 3. Use fit_predict to compute cluster centers and predict cluster index for each sample
clusters = kmeans.fit_predict(df)

# 4. Add the cluster assignments back to the DataFrame for inspection
df['cluster'] = clusters

print("First 5 rows with cluster assignments:")
print(df.head())

# Optional: Check the counts per cluster
print("\nSamples per cluster:")
print(df['cluster'].value_counts())

# Create a mapping dictionary based on the cluster assignments
# Note: Check your specific results to ensure the numbers match the species!
cluster_map = {
    0: 'Versicolor',
    1: 'Setosa',
    2: 'Virginica'
}

# Apply the map to create a new column with names
df['species_name'] = df['cluster'].map(cluster_map)

# Now update your plot code to use the names for the legend
sns.scatterplot(
    data=df,
    x='petal length (cm)',
    y='petal width (cm)',
    hue='species_name',  # Use the named column
    palette='viridis'
)

plt.title('K-Means Clustering with Species Names')
plt.show()