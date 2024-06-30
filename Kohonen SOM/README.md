# Kohonen SOM Project

This project aims to cluster a dataset using the Kohonen Self-Organizing Map (SOM) algorithm. The project is implemented in Python, utilizing various libraries.

## Required Libraries

The following libraries are required to run the project:

- pandas
- numpy
- sklearn
- random
- scipy

You can install the required libraries using the following code block:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score
```

## Loading and Preprocessing the Dataset

The dataset is loaded from the `dataset.xlsx` file and min-max normalization is applied to scale the values between 0 and 1.

```python
data = pd.read_excel("dataset.xlsx")
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(data)
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
```

## Kohonen SOM Model

Parameters for the Kohonen SOM model:

- Learning rate: 0.5
- Sigma: 1.0
- Number of epochs: 100
- Number of clusters: 4

Initial weight values are assigned randomly, and the Gaussian Bell function is used to update the weights.

```python
grid_size = 4
learning_rate = 0.5
sigma = 1.0
num_epochs = 100
weights = np.random.rand(grid_size, grid_size, normalized_df.shape[1])

def gaussian_bell(x, mean, sigma):
    return np.exp(-((x - mean)**2) / (2 * (sigma**2)))

def update_weights(weights, bmu, data_point, learning_rate, sigma):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = euclidean((i, j), bmu)
            h = gaussian_bell(distance, 0, sigma)
            weights[i, j] += h * learning_rate * (data_point - weights[i, j])

def find_bmu(data_point, weights):
    min_distance = float('inf')
    bmu = None
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = euclidean(data_point, weights[i, j])
            if distance < min_distance:
                min_distance = distance
                bmu = (i, j)
    return bmu

for epoch in range(num_epochs):
    for index, row in normalized_df.iterrows():
        data_point = row.values
        bmu = find_bmu(data_point, weights)
        update_weights(weights, bmu, data_point, learning_rate, sigma)
    learning_rate *= 0.99
    sigma *= 0.99
```

## Clustering Results

After training, each example is clustered based on the nearest weight (BMU), and the results are saved to `kume-sonuc.txt`.

```python
clusters = []

for index, row in normalized_df.iterrows():
    data_point = row.values  
    data_point = data_point.reshape(1, -1)  
    bmu = find_bmu(data_point, weights)
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

normalized_df["Cluster"] = clusters
normalized_df["Cluster"].to_csv("kume-sonuc.txt", index=True, header=False)
```

## Calculating Accuracy

The original labels from `index.xlsx` are compared with the clusters assigned in `kume-sonuc.txt` to calculate the clustering accuracy for each label.

```python
index_df = pd.read_excel('index.xlsx')
cluster_results = pd.read_csv('kume-sonuc.txt', header=None, names=["instance (record_no)", "Cluster"])
merged_df = pd.merge(index_df, cluster_results, left_on='instance (record_no)', right_on='instance (record_no)')

cluster_mapping = {
    "C1": merged_df[merged_df["Cluster"] == "C1"]["label"].mode()[0],
    "C2": merged_df[merged_df["Cluster"] == "C2"]["label"].mode()[0],
    "C3": merged_df[merged_df["Cluster"] == "C3"]["label"].mode()[0],
    "C4": merged_df[merged_df["Cluster"] == "C4"]["label"].mode()[0],
}

accuracy_scores = {}

for cluster, label in cluster_mapping.items():
    specific_cluster = merged_df[merged_df["Cluster"] == cluster]
    predicted_labels = specific_cluster["Cluster"].replace(cluster_mapping)
    accuracy = accuracy_score(specific_cluster["label"], predicted_labels)
    accuracy_scores[cluster] = accuracy

for cluster, accuracy in accuracy_scores.items():
  print(f"Accuracy for Cluster {cluster}: {accuracy}")
```

## Project Owner
- Oğuzhan Nejat Karabaş

This guide includes all the steps required to run the project from start to finish. For any issues or questions, feel free to contact me.
