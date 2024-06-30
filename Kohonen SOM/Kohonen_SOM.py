# -*- coding: utf-8 -*-
"""Kohonen_SOM.ipynb



**Kohonen-SOM**
- Oğuzhan Nejat Karabaş

# Gerekli Kütüphaneler
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score

"""# Veri setimizi yükleme ve ön işleme işlemleri
- Veri setini "dataset.xlsx" dosyasından okuyacağız.
- Min-max normalizasyonu uygulayarak 0-255 arası sayısal değerleri olan 784 özniteliğin hepsinin de 0-1 aralığına dönüştüreceğiz.
"""

# "dataset.xlsx" dosyasını okuyarak DataFrame oluşturuyoruz.
data = pd.read_excel("dataset.xlsx", header=None)
data

"""**Normalizasyon**"""

data = pd.read_excel("dataset.xlsx")
# Min-Max normalizasyonunu uygulayarak değerlerimizi 0-1 aralığına indirgiyoruz
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(data)

# normalize ettiğimiz datayı, dataframe haline getiriyoruz.
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

normalized_df

"""# Kohonen SOM Modeli
- Bu aşamada elimizdeki 4 farklı küme için bir Kohonen SOM modeli oluşturacağız. Sonrasında bu ağı, belirli bir sayıda tur (epoch) boyunca eğiteceğiz.
- SOM modelimiz için parametrelerimizi belirleyelim:


1.   Öğrenme hızı = 0.5
2.   Sigma = 1.0
3.   Tur sayısı(epoch) = 100
4.   Küme sayısı = 4




  
"""

# Kohonen SOM modeli için parametreler
grid_size = 4
learning_rate = 0.5
sigma = 1.0
num_epochs = 100

# İlk ağırlık değerlerini rastgele atama işlemi
weights = np.random.rand(grid_size, grid_size, normalized_df.shape[1])

# Gaussian Bell fonksiyonu
def gaussian_bell(x, mean, sigma):
    return np.exp(-((x - mean)**2) / (2 * (sigma**2)))

# Ağırlık vektörlerini güncelleme
def update_weights(weights, bmu, data_point, learning_rate, sigma):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = euclidean((i, j), bmu)
            h = gaussian_bell(distance, 0, sigma)
            weights[i, j] += h * learning_rate * (data_point - weights[i, j])

# Best Matching Unit (BMU) bulma
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

"""**Modeli Eğitme**

- Modelimizi oluşturduktan sonra eğitmemiz gerekiyor.
- Her turda, veri setimizdeki her örnek için:


1.   Öklidyen uzaklığı kullanarak en yakın ağırlığı yani Best Matching Unit (BMU) buluyoruz.
2.   Gaussian Bell kullanarak h(x) topolojik komşu fonksiyonunu hesaplıyoruz.
3.   Ağırlıkları güncelliyoruz: BMU ve komşu ağırlıklarını, öğrenme hızı ve h(x) değerlerine göre güncelliyoruz.
4. Öğrenme hızı ve sigma değerlerimizi azaltıyoruz. ( decay)
"""

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Her veri noktası için
    for index, row in normalized_df.iterrows():
        data_point = row.values
        
        # En yakın ağırlık vektörünü (BMU) buluyoruz.
        bmu = find_bmu(data_point, weights)
        
        # BMU ve komşu ağırlık vektörlerini güncelliyoruz.
        update_weights(weights, bmu, data_point, learning_rate, sigma)
        
    # Öğrenme hızı ve sigma değerlerini azaltıyoruz.(decay)
    learning_rate *= 0.99
    sigma *= 0.99

"""# Kümeleme Sonuçları
- Eğitimimiz tamamlandıktan sonra, her örneği en yakın ağırlığa ( BMU) göre kümeliyoruz.
- Kümeleme sonuçlarımızı " kume-sonuc.txt" dosyasına
yazdırıyoruz.

"""

# Gaussian Bell fonksiyonu
def gaussian_bell(x, mean, sigma):
    return np.exp(-((x - mean)**2) / (2 * (sigma**2)))

# Ağırlık vektörlerini güncelliyoruz.
def update_weights(weights, bmu, data_point, learning_rate, sigma):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = euclidean((i, j), bmu)
            h = gaussian_bell(distance, 0, sigma)
            weights[i, j] += h * learning_rate * (data_point - weights[i, j])

# Öklidyen uzaklık
def euclidean(a, b):
    return np.linalg.norm(a - b)

# Best Matching Unit bulma işlemi (BMU)
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

# Kümeleme İşlemleri
clusters = []

for index, row in normalized_df.iterrows():
    data_point = row.values  
    data_point = data_point.reshape(1, -1)  

    bmu = find_bmu(data_point, weights)

    # BMU'nun ait olduğu kümenin belirlenmesi işlemi
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

    # 
    clusters.append(cluster)

# Küme bilgilerini oluşturduğumuz dataframe dosyasına ekliyoruz.
normalized_df["Cluster"] = clusters

# Küme bilgilerimizi .txt dosyasına yaz
normalized_df["Cluster"].to_csv("kume-sonuc.txt", index=True, header=False)

"""# Başarı Oranlarını Hesaplama
- "index.xlsx" dosyasındaki orijinal etiketlerimizle, " kume-sonuc.txt" dosyasındaki atanan kümelerimizi karşılaştırarak, her bir etiketimiz,sınıfımız için kümeleme başarı oranlarını hesaplıyoruz.
"""

# index.xlsx dosyasını okuyup, dataframe haline getiriyoruz.
index_df = pd.read_excel('index.xlsx')

# Küme-sonuc.txt dosyasını okuyup dataframe haline getiriyoruz.
cluster_results = pd.read_csv('kume-sonuc.txt', header=None, names=["instance (record_no)", "Cluster"])

# Yeni oluşturduğumuz df dosyalarını birleştiriyoruz.
merged_df = pd.merge(index_df, cluster_results, left_on='instance (record_no)', right_on='instance (record_no)')

# Hangi kümenin hangi etiketi temsil ettiğini belirleme işlemi
cluster_mapping = {
    "C1": merged_df[merged_df["Cluster"] == "C1"]["label"].mode()[0],
    "C2": merged_df[merged_df["Cluster"] == "C2"]["label"].mode()[0],
    "C3": merged_df[merged_df["Cluster"] == "C3"]["label"].mode()[0],
    "C4": merged_df[merged_df["Cluster"] == "C4"]["label"].mode()[0],
}

# Kümeleme sonuçlarını orijinal etiketlere dönüştürme
predicted_labels = merged_df["Cluster"].replace(cluster_mapping)

# Başarı oranını hesaplama
accuracy_scores = {}

for cluster, label in cluster_mapping.items():
    # Sadece belirli bir küme ve etiketi içeren satırları alıyoruz.
    specific_cluster = merged_df[merged_df["Cluster"] == cluster]
    
    # Kümeleme sonuçlarımızı orijinal etiketlere dönüştürüyoruz.
    predicted_labels = specific_cluster["Cluster"].replace(cluster_mapping)
    
    # Başarı oranını hesaplama işlemi
    accuracy = accuracy_score(specific_cluster["label"], predicted_labels)
    accuracy_scores[cluster] = accuracy

# Her bir küme için başarı oranını yazdırıyoruz.
for cluster, accuracy in accuracy_scores.items():
  print(f"Küme {cluster} için başarı oranı: {accuracy}")