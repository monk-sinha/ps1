import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV data (assuming the dataset is saved as 'dataset.csv')
data = pd.read_csv('dataset.csv')

# Convert elapsed_time to seconds and clean data
data['accuracy'] = data['accuracy'].fillna('0')
accuracies = data['accuracy'].astype(float).values
elapsed_times = data['elapsed_time_second'].dropna().values

# Combine features into a 2D array for clustering (only where both are valid)
mask = ~data['accuracy'].isna() & ~data['elapsed_time_second'].isna()
features = np.column_stack((accuracies[mask], elapsed_times[mask]))

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(features)

# Add cluster labels to the data, filling NaN rows with -1
data['cluster'] = -1
data.loc[mask, 'cluster'] = clusters

# Plotting the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('Accuracy (%)')
plt.ylabel('Elapsed Time (seconds)')
plt.title('K-means Clustering of Accuracy and Elapsed Time (4 Clusters)')

# Add cluster centers
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.legend()

# Show the plot
plt.show()

# Print cluster centers and counts for reference
cluster_counts = np.bincount(clusters, minlength=3)
for i in range(4):
    print(f"Cluster {i}: Center (Accuracy: {cluster_centers[i, 0]:.2f}%, Time: {cluster_centers[i, 1]:.2f}s), Count = {cluster_counts[i]}")