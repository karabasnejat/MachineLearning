
# Fantasy Premier League Oyuncu Analizi

Bu proje, Fantasy Premier League (FPL) 2024-2025 sezonu için stoper oyuncularının (defans oyuncularının) yeteneklerine göre kümelendiği bir analiz uygulamasıdır. Uygulama, oyuncuları performans metriklerine göre gruplandırarak, kullanıcıların en iyi oyuncuları incelemesine olanak tanır.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Uygulama Özellikleri](#uygulama-özellikleri)
- [Katkıda Bulunma](#katkıda-bulunma)


## Proje Hakkında

Bu proje, Python kullanarak oluşturulmuş bir veri görselleştirme ve analiz uygulamasıdır. Uygulama, FPL oyuncularının çeşitli performans metriklerini kullanarak K-means kümeleme algoritması ile gruplandırılmasını sağlar. İngiltere Premier Lig oyuncuları için oluşturulan bu analiz, oyuncuların hangi kümelerde yer aldığını ve birbirlerine ne kadar benzediklerini görsel olarak sunar.

## Kullanılan Teknolojiler

- **Python 3.8+**
- **Pandas**
- **Scikit-learn**
- **Plotly**

## Kurulum

Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. **Gereksinimleri Yükleyin:**

   Öncelikle, gerekli Python paketlerini yükleyin:

   ```bash
   pip install pandas scikit-learn plotly
   ```

2. **Projeyi Kopyalayın:**

   Proje dosyalarını bilgisayarınıza indirin veya klonlayın:

   ```bash
   git clone https://github.com/karabasnejat/MachineLearning.git
   cd KMeansClusteringwithPremierLeague
   ```

3. **Veri Dosyasını Ekleyin:**

   `players.csv` dosyasını proje dizinine ekleyin. Bu dosya, FPL oyuncularının istatistiklerini içermelidir.

## Kullanım

Python dosyasını çalıştırarak analizi gerçekleştirebilirsiniz:

```bash
python app.py
```

Bu komut, belirlenen pozisyondaki (stoper) FPL oyuncularını inceleyecek ve performanslarına göre hangi kümelerde yer aldıklarını gösterecektir.

## Uygulama Özellikleri

- **K-means Kümeleme:** FPL stoperlerini performans metriklerine göre gruplandırır.
- **PCA ile Boyut İndirgeme:** Yüksek boyutlu verileri iki boyutlu bir grafikte görselleştirir.
- **Etkileşimli Grafikler:** Oyuncu isimlerini küme renkleriyle birlikte gösterir, böylece hangi oyuncunun hangi kümeye ait olduğunu kolayca görebilirsiniz.

## Katkıda Bulunma

Bu projeye katkıda bulunmak isterseniz, lütfen bir "fork" yapın, kendi dalınızı oluşturun ve değişikliklerinizi gönderin:

1. Projeyi "fork"layın
2. Bir dal oluşturun: `git checkout -b my-feature`
3. Değişikliklerinizi yapın ve bir commit yapın: `git commit -m 'Yeni özellik ekle'`
4. Dalınıza "push" yapın: `git push origin my-feature`
5. Bir "Pull Request" açın

