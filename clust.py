import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV data (assuming the dataset is saved as 'dataset.csv')
data = pd.read_csv('dataset.csv')

# Extract the 'accuracy' column and clean it (convert to float, handle NaN)
accuracies = data['accuracy'].dropna().astype(float).values.reshape(-1, 1)

# Perform k-means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(accuracies)

# Create a mask for non-NaN accuracy values
mask = ~data['accuracy'].isna()

# Create a new column for clusters, filling NaN rows with -1 (or any sentinel value)
data['cluster'] = -1  # Initialize with a sentinel value for NaN rows
data.loc[mask, 'cluster'] = clusters

# Plotting the results
plt.figure(figsize=(10, 6))
# Plot only the clustered points (excluding NaN rows)
scatter = plt.scatter(accuracies.flatten(), np.zeros_like(accuracies.flatten()), c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('Accuracy (%)')
plt.title('K-means Clustering of Accuracy (4 Clusters)')
plt.yticks([])  # Hide y-axis ticks since it's a 1D projection

# Add cluster centers as horizontal lines
cluster_centers = kmeans.cluster_centers_.flatten()
for center in cluster_centers:
    plt.axhline(y=0, xmin=0, xmax=center/100, color='red', linestyle='--', alpha=0.5)

# Add a legend
plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(4)])

# Show the plot
plt.show()

# Print cluster centers and counts for reference
cluster_counts = np.bincount(clusters, minlength=3)
for i in range(4):
    print(f"Cluster {i}: Center = {cluster_centers[i]:.2f}%, Count = {cluster_counts[i]}")