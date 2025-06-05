#Aim:-Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same Dataset for clustering using k-Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add Java/Python ML library classes / API in the program.

#Program:-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sm

# Load and prepare the iris dataset
dataset = load_iris()
X = pd.DataFrame(dataset.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(dataset.target, columns=['Targets'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Set up the figure for plotting
plt.figure(figsize=(12, 5))
colormap = np.array(['red', 'lime', 'black'])

# Plot 1: Original Data
plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=50)
plt.title('Original Iris Dataset')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Plot 2: K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[kmeans_labels], s=50)
plt.title('K-Means Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Adjust layout and save plot
plt.tight_layout()
plt.savefig('clustering_comparison.png')

# Print clustering metrics
print("\nK-Means Clustering Performance Metrics:")
print("-" * 30)
print("Silhouette Score:", sm.silhouette_score(X_scaled, kmeans_labels))
print("Calinski-Harabasz Score:", sm.calinski_harabasz_score(X_scaled, kmeans_labels))


#Output :-
#the output diagram will be saved as 'clustering_comparison.png' in this Folder directory.


#Result:- The above program is executed successfully.