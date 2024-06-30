# -*- coding: utf-8 -*-
"""Kohonen_SOM.ipynb

**Kohonen-SOM**
- Oğuzhan Nejat Karabaş

# Required Libraries
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score

"""# Loading and Preprocessing the Dataset
- We will read the dataset from the "dataset.xlsx" file.
- By applying min-max normalization, we will convert all 784 attributes with numerical values between 0-255 to the 0-1 range.
"""

# Read the "dataset.xlsx" file to create a DataFrame.
data = pd.read_excel("dataset.xlsx", header=None)
data

"""**Normalization**"""

data = pd.read_excel("dataset.xlsx")
# Apply Min-Max normalization to scale the values to the 0-1 range
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(data)

# Convert the normalized data to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

normalized_df

"""# Kohonen SOM Model
- At this stage, we will create a Kohonen SOM model for the 4 different clusters we have. We will then train this network for a specified number of epochs.
- Let's define our parameters for the SOM model:

1. Learning rate = 0.5
2. Sigma = 1.0
3. Number of epochs = 100
4. Number of clusters = 4
"""

# Parameters for the Kohonen SOM model
grid_size = 4
learning_rate = 0.5
sigma = 1.0
num_epochs = 100

# Randomly initialize the initial weight values
weights = np.random.rand(grid_size, grid_size, normalized_df.shape[1])

# Gaussian Bell function
def gaussian_bell(x, mean, sigma):
    return np.exp(-((x - mean)**2) / (2 * (sigma**2)))

# Update weight vectors
def update_weights(weights, bmu, data_point, learning_rate, sigma):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = euclidean((i, j), bmu)
            h = gaussian_bell(distance, 0, sigma)
            weights[i, j] += h * learning_rate * (data_point - weights[i, j])

# Find Best Matching Unit (BMU)
def find_bmu(data_point, weights):
    min_distance = float("inf")
    bmu = None
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = euclidean(data_point, weights[i, j])
            if distance < min_distance:
                min_distance = distance
                bmu = (i, j)
    return bmu

"""**Training the Model**

- After creating our model, we need to train it.
- In each epoch, for every example in our dataset:

1. Find the closest weight, i.e., the Best Matching Unit (BMU) using Euclidean distance.
2. Calculate the h(x) topological neighborhood function using the Gaussian Bell function.
3. Update the weights: Update the BMU and neighboring weights according to the learning rate and h(x) values.
4. Decay the learning rate and sigma values.
"""

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # For each data point
    for index, row in normalized_df.iterrows():
        data_point = row.values
        
        # Find the closest weight vector (BMU)
        bmu = find_bmu(data_point, weights)
        
        # Update the BMU and neighboring weight vectors
        update_weights(weights, bmu, data_point, learning_rate, sigma)
        
    # Decay the learning rate and sigma values
    learning_rate *= 0.99
    sigma *= 0.99

"""# Clustering Results
- After training, we cluster each example according to the closest weight (BMU).
- We save our clustering results to the "kume-sonuc.txt" file.
"""

# Gaussian Bell function
def gaussian_bell(x, mean, sigma):
    return np.exp(-((x - mean)**2) / (2 * (sigma**2)))

# Update weight vectors
def update_weights(weights, bmu, data_point, learning_rate, sigma):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = euclidean((i, j), bmu)
            h = gaussian_bell(distance, 0, sigma)
            weights[i, j] += h * learning_rate * (data_point - weights[i, j])

# Euclidean distance
def euclidean(a, b):
    return np.linalg.norm(a - b)

# Find Best Matching Unit (BMU)
def find_bmu(data_point, weights):
    min_distance = float('inf')
    bmu = None

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weight = weights[i, j].reshape(1, -1)  
            distance = euclidean(data_point, weight)

            if distance < min_distance:
                min_distance = distance
                bmu = (i, j)

    return bmu

# Clustering operations
clusters = []

for index, row in normalized_df.iterrows():
    data_point = row.values  
    data_point = data_point.reshape(1, -1)  

    bmu = find_bmu(data_point, weights)

    # Determine which cluster the BMU belongs to
    grid_size = weights.shape[0]  
    if bmu[0] < grid_size / 2:
        if bmu[1] < grid_size / 2:
            cluster = "C1"
        else:
            cluster = "C2"
    else:
        if bmu[1] < grid_size / 2:
            cluster = "C3"
        else:
            cluster = "C4"

    clusters.append(cluster)

# Add the cluster information to the DataFrame
normalized_df["Cluster"] = clusters

# Write the cluster information to a .txt file
normalized_df["Cluster"].to_csv("kume-sonuc.txt", index=True, header=False)

"""# Calculating Accuracy
- By comparing the original labels in the "index.xlsx" file with the assigned clusters in the "kume-sonuc.txt" file, we calculate the clustering accuracy for each label/class.
"""

# Read the index.xlsx file into a DataFrame
index_df = pd.read_excel('index.xlsx')

# Read the kume-sonuc.txt file into a DataFrame
cluster_results = pd.read_csv('kume-sonuc.txt', header=None, names=["instance (record_no)", "Cluster"])

# Merge the DataFrames
merged_df = pd.merge(index_df, cluster_results, left_on='instance (record_no)', right_on='instance (record_no)')

# Determine which label each cluster represents
cluster_mapping = {
    "C1": merged_df[merged_df["Cluster"] == "C1"]["label"].mode()[0],
    "C2": merged_df[merged_df["Cluster"] == "C2"]["label"].mode()[0],
    "C3": merged_df[merged_df["Cluster"] == "C3"]["label"].mode()[0],
    "C4": merged_df[merged_df["Cluster"] == "C4"]["label"].mode()[0],
}

# Convert clustering results to original labels
predicted_labels = merged_df["Cluster"].replace(cluster_mapping)

# Calculate accuracy
accuracy_scores = {}

for cluster, label in cluster_mapping.items():
    # Get only the rows containing a specific cluster and label
    specific_cluster = merged_df[merged_df["Cluster"] == cluster]
    
    # Convert clustering results to original labels
    predicted_labels = specific_cluster["Cluster"].replace(cluster_mapping)
    
    # Calculate accuracy
    accuracy = accuracy_score(specific_cluster["label"], predicted_labels)
    accuracy_scores[cluster] = accuracy

# Print accuracy for each cluster
for cluster, accuracy in accuracy_scores.items():
  print(f"Accuracy for Cluster {cluster}: {accuracy}")
