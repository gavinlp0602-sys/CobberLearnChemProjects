import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 1. Load data and run K-Means
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(df)

# 2. Initialize the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Define the three features to plot
x = df['sepal length (cm)']
y = df['petal length (cm)']
z = df['petal width (cm)']

# 4. Create the scatter plot
# 'c' uses the cluster labels for color, 's' sets the point size
scatter = ax.scatter(x, y, z, c=df['cluster'], cmap='viridis', s=50)

# 5. Add axis labels, title, and legend
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.set_zlabel('Petal Width (cm)')
plt.title('3D K-Means Clustering: Iris Dataset')

# Use legend_elements to automatically create a legend from the scatter object
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

# 6. Save the plot
plt.show()