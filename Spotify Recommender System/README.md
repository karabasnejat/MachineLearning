# Spotify Music Recommendation System

## Project Overview

This project builds a music recommendation system using Spotify song data. The system suggests songs that are similar to a given song by analyzing various features such as `danceability`, `energy`, `tempo`, and more. The project utilizes machine learning algorithms like K-Means clustering for grouping similar songs and dimensionality reduction techniques like PCA and t-SNE for visualizing the song clusters.

## Features

- **Song Clustering:** Groups similar songs using K-Means clustering based on selected features.
- **Dimensionality Reduction:** Reduces the complexity of the data with PCA and t-SNE, making it easier to visualize and understand.
- **Song Recommendations:** Recommends songs similar to a given song, ensuring that the recommendations are diverse and relevant.
- **Export Recommendations:** Saves the recommended songs to a `.txt` file for easy sharing and record-keeping.

## Installation

To get started with the project, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/karabasnejat/spotify-music-recommendation-system.git
   cd spotify-music-recommendation-system
   ```

2. **Install Dependencies:**

   Make sure you have Python installed. Install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Spotify Dataset:**

   The project requires a dataset containing song features from Spotify. Ensure you have the dataset in the correct format (CSV file) and place it in the project directory.

## Usage

1. **Load and Prepare Data:**

   Load the Spotify dataset and prepare the data for analysis by scaling the features:

   ```python
   import pandas as pd
   from sklearn.preprocessing import StandardScaler

   df = pd.read_csv('spotify_data.csv')
   features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
   scaler = StandardScaler()
   df_scaled = scaler.fit_transform(df[features])
   ```

2. **Cluster Songs:**

   Use K-Means to group songs into clusters based on their features:

   ```python
   from sklearn.cluster import KMeans

   kmeans = KMeans(n_clusters=10, random_state=42)
   kmeans.fit(df_scaled)
   df['cluster'] = kmeans.labels_
   ```

3. **Dimensionality Reduction and Visualization:**

   Reduce the data dimensions using PCA and t-SNE and visualize the clusters:

   ```python
   from sklearn.decomposition import PCA
   from sklearn.manifold import TSNE
   import matplotlib.pyplot as plt
   import seaborn as sns

   pca = PCA(n_components=2)
   pca_result = pca.fit_transform(df_scaled)
   df['pca-one'] = pca_result[:, 0]
   df['pca-two'] = pca_result[:, 1]

   tsne = TSNE(n_components=2, random_state=42)
   tsne_result = tsne.fit_transform(df_scaled)
   df['tsne-one'] = tsne_result[:, 0]
   df['tsne-two'] = tsne_result[:, 1]

   plt.figure(figsize=(10, 6))
   sns.scatterplot(x='tsne-one', y='tsne-two', hue='cluster', palette='viridis', data=df, legend='full')
   plt.title('t-SNE Clustering of Songs')
   plt.show()
   ```

4. **Get Song Recommendations:**

   Use the recommendation function to get songs similar to a given song:

   ```python
   def recommend_songs(song_id, num_recommendations=5):
       song_features = df_scaled.iloc[song_id].values.reshape(1, -1)
       cluster_label = df.iloc[song_id]['cluster']
       cluster_songs = df[df['cluster'] == cluster_label]
       cluster_features = df_scaled[df['cluster'] == cluster_label]
       distances = cdist(song_features, cluster_features, 'euclidean')
       closest_songs = np.argsort(distances)[0][:num_recommendations * 2]
       recommended_songs = cluster_songs.iloc[closest_songs]
       recommended_songs = recommended_songs.drop_duplicates(subset=['track_name', 'artists'])
       return recommended_songs.head(num_recommendations)

   recommended_songs = recommend_songs(0)
   print(recommended_songs[['track_name', 'artists']])
   ```

5. **Save Recommendations to a File:**

   Save the recommended songs to a text file:

   ```python
   output_file = 'recommended_songs.txt'
   with open(output_file, 'w') as file:
       file.write("Recommended Songs:\n")
       file.write("===================\n")
       for index, row in recommended_songs.iterrows():
           track_info = f"Track: {row['track_name']}, Artist(s): {row['artists']}\n"
           file.write(track_info)
       print(f"Recommended songs have been saved to {output_file}")
   ```

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome all contributions that help improve the recommendation system or add new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```
