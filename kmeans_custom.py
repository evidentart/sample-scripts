import numpy as np
import pandas as pd

def initialize_centroids(data, k):
    """Randomly initialize centroids from the dataset."""
    n_samples = data.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[random_indices]
    return centroids

def assign_clusters(data, centroids):
    """Assign each data point to the closest centroid."""
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    """Calculate new centroids as the mean of points in each cluster."""
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(data, k, max_iters=100, tolerance=1e-4):
    """Perform K-Means clustering."""
    # Step 1: Initialize centroids
    centroids = initialize_centroids(data, k)
    
    for i in range(max_iters):
        # Step 2: Assign clusters
        labels = assign_clusters(data, centroids)
        
        # Step 3: Update centroids
        new_centroids = update_centroids(data, labels, k)
        
        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged after {i+1} iterations.")
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Load your actual data
# Replace 'your_data.csv' with the path to your dataset
df = pd.read_csv('Replace with your file path')


# Select only numeric columns for clustering (e.g., 'lead_time', 'adr', 'stays_in_weekend_nights', etc.)
# Update the column names based on your actual data
data = df[['lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']].values

# Set the number of clusters
k = 5

# Run K-means on the actual data
labels, centroids = kmeans(data, k)

print("Cluster assignments:", labels)
print("Final centroids:", centroids)
