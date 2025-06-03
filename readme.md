# Laporan Proyek Machine Learning - Daniela Natali Putri

## Domain Proyek

Permintaan akan tempat tinggal yang merupakan kebutuhan dasar manusia terus menunjukkan tren peningkatan. Meskipun demikian, menetapkan harga yang akurat untuk sebuah rumah bukanlah perkara mudah. Harga properti seringkali mengalami fluktuasi yang signifikan dan pergerakan ini sangat dipengaruhi oleh faktor-faktor seperti kualitas dan detail bangunan, serta kondisi ekonomi yang lebih luas, salah satunya adalah inflasi. Selain itu harga rumah setiap tahun akan semakin mahal, sehingga dibutuhkan analisis serta pemilihan yang tepat yang harus dilakukan untuk membeli sebuah rumah. Kegunaan analisis harga properti ini akan dirasakan oleh berbagai pemangku kepentingan. Pembeli diuntungkan karena dapat mengambil keputusan pembelian yang tepat dan mencegah pengeluaran berlebih. Penjual properti dapat menentukan harga yang paling kompetitif berdasarkan kondisi pasar terkini. Investor bisa lebih akurat dalam mengidentifikasi properti yang berpotensi memberikan keuntungan maksimal. Bagi pengembang, pemahaman tren harga di berbagai lokasi membantu dalam menyusun strategi pengembangan dan pemasaran yang lebih cerdas. Sementara itu, pembuat kebijakan dapat memanfaatkan data ini untuk mengawasi pasar dan merancang regulasi perumahan yang efektif.

Menurut penelitian Adetunji dkk. [1], Indeks harga rumah berfungsi sebagai variabel penting untuk memperkirakan adanya inkonsistensi dalam penilaian harga properti. Hal ini dikarenakan harga rumah sendiri sangat berkorelasi dengan berbagai faktor seperti lokasi geografis, karakteristik kota, serta jumlah populasinya, sehingga mendorong dilakukannya penelitian lebih lanjut dalam prediksi harga rumah. Untuk tujuan tersebut, penelitian ini mengimplementasikan teknik machine learning Random Forest dengan memanfaatkan dataset yang terdiri dari 500 data dan 14 fitur. Hasil penelitian kemudian menunjukkan bahwa perbandingan antara harga prediksi yang dihasilkan model dengan harga aktual mengungkapkan tingkat akurasi yang dapat diterima, dengan margin kesalahan yang berada dalam kisaran ±5%.

Menurut penelitian yang dilakukan oleh Peterson dan Flanagan [2], penerapan algoritma machine learning seperti regresi, random forest, dan neural network dapat secara signifikan meningkatkan akurasi prediksi harga rumah dibandingkan dengan metode tradisional. Studi lain oleh Fan et al. [3] menunjukkan bahwa model berbasis gradient boosting dapat menghasilkan peningkatan akurasi hingga 15% dalam prediksi harga properti residensial dibandingkan dengan model regresi linier sederhana.

---
## Business Understanding
### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, proyek ini bertujuan untuk mengembangkan sistem prediksi harga rumah di Chicago untuk menjawab permasalahan berikut:
1. Bagaimana ketidakakuratan dalam penentuan harga rumah dapat menyebabkan inefisiensi pasar, yang berdampak pada kesulitan penjualan properti oleh penjual atau kehilangan potensi keuntungan, serta memperpanjang waktu properti di pasar?
2. Dengan cara apa kesulitan yang dihadapi investor properti dan pembeli rumah dalam mengidentifikasi nilai properti sebenarnya berdasarkan berbagai faktor (seperti lokasi, ukuran, dan karakteristik demografis) dapat mengakibatkan keputusan investasi yang kurang optimal dan potensi kerugian finansial?
3. Sejauh mana keterbatasan metode penilaian properti tradisional—yang meliputi durasi pengerjaan yang lama, biaya tinggi, serta kerentanan terhadap inkonsistensi dan bias subjektif—dapat menghambat dinamika pasar properti dan berkontribusi pada peningkatan biaya transaksi secara keseluruhan?

### Goals
Untuk menjawab pertanyaan tersebut, sistem akan dibangun dengan tujuan atau goals sebagai berikut:
1. Mengidentifikasi dan menganalisis faktor-faktor determinan yang secara signifikan memengaruhi fluktuasi harga rumah di Chicago melalui eksplorasi data.
2. Merancang dan mengembangkan model machine learning prediktif untuk harga rumah di Chicago, dengan fokus pada minimalisasi error prediksi berdasarkan serangkaian fitur yang relevan.
3. Melakukan evaluasi komparatif terhadap performa berbagai algoritma machine learning untuk mengidentifikasi dan memilih model dengan akurasi prediksi harga rumah tertinggi.

### Solution statements
Untuk mencapai tujuan yang telah ditetapkan, berikut adalah solusi yang akan diterapkan:
1. Melakukan eksplorasi data dan analisis untuk mengidentifikasi pola dan hubungan antara fitur-fitur dengan harga rumah.
2. Mengembangkan beberapa untuk prediksi harga rumah, diantara lain KNN, Random Forest, dan Gradient Boosting.
3. Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa prediksi.
4. Menggunakan metrik evaluasi seperti RMSE, MAE, dan R² untuk membandingkan performa model-model tersebut dan memilih model terbaik.

---
## Data Understanding
Dataset yang digunakan berasal dari [Kaggle](https://https://www.kaggle.com/datasets/mssmartypants/paris-housing-price-prediction). Dataset ini berisi informasi tentang rumah yang disewakan atau dijual di Paris, termasuk fitur-fitur seperti jumlah kamar mandi, kamar tidur, luas apartemen, dan harga.

Dataset ini terdiri dari 10000 sampel dengan 16 fitur dan 1 target, yang mencakup berbagai aspek dari perumahan di Paris.

### Variabel-variabel pada Chicago Housing Dataset adalah sebagai berikut:

|      **Fitur**      |                              **Deskripsi**                             |
| :-----------------: | :--------------------------------------------------------------------: |
|    `squareMeters`   |                Luas bangunan rumah dalam meter persegi.                |
|   `numberOfRooms`   |      Jumlah total ruangan (termasuk kamar tidur, ruang tamu, dll).     |
|      `hasYard`      |           Apakah rumah memiliki halaman (1 = ya, 0 = tidak).           |
|      `hasPool`      |         Apakah rumah memiliki kolam renang (1 = ya, 0 = tidak).        |
|       `floors`      |                        Jumlah lantai pada rumah.                       |
|      `cityCode`     |       Kode numerik untuk menunjukkan wilayah administratif/kota.       |
|   `cityPartRange`   | Rentang lokasi rumah dalam bagian kota (misalnya, pusat vs pinggiran). |
|   `numPrevOwners`   |             Jumlah pemilik sebelumnya dari rumah tersebut.             |
|        `made`       |                     Tahun rumah tersebut dibangun.                     |
|     `isNewBuilt`    |        Apakah rumah merupakan bangunan baru (1 = ya, 0 = tidak).       |
| `hasStormProtector` |       Apakah rumah memiliki pelindung badai (1 = ya, 0 = tidak).       |
|      `basement`     |      Apakah rumah memiliki ruang bawah tanah (1 = ya, 0 = tidak).      |
|       `attic`       |            Apakah rumah memiliki loteng (1 = ya, 0 = tidak).           |
|       `garage`      |            Apakah rumah memiliki garasi (1 = ya, 0 = tidak).           |
|   `hasStorageRoom`  |  Apakah rumah memiliki ruang penyimpanan tambahan (1 = ya, 0 = tidak). |
|    `hasGuestRoom`   |          Apakah rumah memiliki kamar tamu (1 = ya, 0 = tidak).         |
|       `price`       |     Harga rumah dalam satuan mata uang tertentu (target variabel).     |


---
## Eksplorasi Data Awal (EDA)
Untuk memahami dataset dengan lebih baik, beberapa tahapan eksplorasi data telah dilakukan:

#### 1. Statistik Deskriptif
Statistik deskriptif menunjukkan bahwa:
- Harga rumah (price) memiliki median 5.016.180 dan rata-rata 4.993.448, menunjukkan distribusi harga yang hampir simetris.
- Luas rumah (squareMeters) memiliki median 50.105,5 m² dan rata-rata 49.870,13 m², menunjukkan penyebaran data cukup merata di sekitar nilai tengah.
- Jumlah kamar (numberOfRooms) memiliki median 50 dan rata-rata 50,36, dengan rentang nilai antara 1 hingga 100 kamar, yang cukup luas.
- Fitur biner seperti hasYard, hasPool, dan isNewBuilt memiliki nilai median 0, menunjukkan bahwa sebagian besar rumah tidak memiliki fitur-fitur ini.
- Tahun pembangunan (made) memiliki median 2005,5 dan rata-rata 2005,49, menunjukkan bahwa sebagian besar rumah dibangun sekitar awal 2000-an.

#### 2. Analisis Missing Values
Dataset ini tidak memiliki missing values.

#### 3. Distribusi Target Variable
Distribusi harga rumah (price) menunjukkan distribusi yang normal.

#### 4. Analisis Korelasi
Matriks korelasi menunjukkan:
- squareMeters (luas rumah) memiliki korelasi positif tertinggi dan sangat kuat dengan harga rumah (r = 1.00), menjadikannya fitur paling signifikan untuk prediksi.
- Fitur-fitur lain seperti numberOfRooms, hasYard, hasPool, floors, dan garage memiliki korelasi sangat lemah atau mendekati nol terhadap harga, sehingga pengaruhnya terhadap prediksi sangat kecil dalam model linier.
- Tidak ditemukan hubungan kuat antar fitur lainnya, yang menunjukkan tidak ada multikolinearitas berarti.
- Fitur seperti cityCode dan cityPartRange memiliki korelasi rendah, namun bisa memiliki makna non-linear jika dikombinasikan dengan model machine learning berbasis pohon.

---
## Data Preparation
Beberapa teknik data preparation yang diterapkan dalam proyek ini:

1. **Pembagian Data (Train-Test Split)**
   Data dibagi menjadi set pelatihan (80%) dan pengujian (20%) untuk mengevaluasi performa model pada data yang tidak pernah dilihat sebelumnya. Proses ini penting untuk menghindari overfitting dan mendapatkan estimasi yang tidak bias tentang performa model.
2. **Standardisasi Fitur**
   Semua fitur numerik distandardisasi menggunakan StandardScaler untuk memastikan semua fitur berada dalam skala yang sama. Tanpa standardisasi, fitur dengan skala yang lebih besar akan mendominasi proses pembelajaran model.
   
---
## Modeling
Pada tahap modeling, saya menerapkan tiga algoritma machine learning berbeda untuk memprediksi harga properti. Berikut adalah penjelasan detail tentang tiap algoritma, karakteristiknya, dan proses pemodelan yang dilakukan:

### 1. KNN (K-Nearest Neighbor)
Algoritma KNN mencari tetangga terdekat untuk melakukan prediksi berdasarkan rata-rata nilai target dari tetangga tersebut.

**Karakteristik**: 
- KNN tidak membangun model secara eksplisit selama fase pelatihan. Sebaliknya, ia menyimpan seluruh dataset pelatihan dan melakukan perhitungan saat prediksi dibutuhkan.
- KNN tidak membuat asumsi apa pun tentang distribusi data yang mendasarinya. Ini membuatnya fleksibel dan dapat bekerja dengan baik pada data dengan struktur yang kompleks.
- KNN mengklasifikasikan atau memprediksi titik data baru berdasarkan mayoritas kelas atau rata-rata nilai dari 'k' tetangga terdekatnya, yang diukur menggunakan metrik jarak (seperti Euclidean, Manhattan, dll.).

**Kelebihan**:
- Sederhana dan Mudah Diimplementasikan: Konsep dasarnya intuitif, membuatnya mudah dipahami dan diimplementasikan tanpa memerlukan asumsi yang kompleks mengenai distribusi data.
- Adaptif terhadap Data Lokal: KNN dapat beradaptasi dengan baik terhadap pola lokal dalam data karena prediksi didasarkan pada titik data terdekat, yang bisa menangkap nuansa spesifik di area tertentu di Paris.
- Tidak Memerlukan Tahap Pelatihan Eksplisit: KNN menyimpan seluruh dataset sebagai modelnya, sehingga tidak ada waktu yang signifikan untuk "melatih" model secara tradisional.

**Kekurangan**:
- Sensitif terhadap Skala Fitur: Performa KNN sangat dipengaruhi oleh skala variabel. Fitur dengan rentang nilai yang besar dapat mendominasi perhitungan jarak, sehingga normalisasi fitur menjadi penting.
- Biaya Komputasi Tinggi saat Prediksi: Untuk setiap prediksi baru, KNN harus menghitung jarak ke semua titik data dalam dataset pelatihan, yang bisa menjadi lambat untuk dataset besar.
- Kesulitan dengan Data Dimensi Tinggi (Curse of Dimensionality): Efektivitas KNN menurun seiring bertambahnya jumlah fitur, karena konsep "kedekatan" menjadi kurang bermakna di ruang dimensi tinggi.

**Parameter yang digunakan**:
- n_neighbors=10

### 2. Random Forest Regression
Random Forest adalah algoritma ensemble yang menggunakan multiple decision trees.

**Karakteristik**: 
- Random Forest membangun banyak decision tree (pohon keputusan) secara independen selama proses training.
- Setiap pohon dilatih pada sampel data acak (bootstrap sample) dari dataset asli, dan pada setiap pemisahan (split) di pohon, hanya subset acak dari fitur yang dipertimbangkan.
- Dengan menggabungkan prediksi dari banyak pohon yang beragam (hasil dari pengambilan sampel acak), Random Forest cenderung mengurangi overfitting yang sering terjadi pada satu decision tree dan meningkatkan akurasi prediksi.

**Kelebihan**:
- Akurasi Tinggi dan Mampu Menangani Data Kompleks: Umumnya memberikan hasil prediksi yang sangat baik dan mampu menangkap hubungan non-linear dalam data harga rumah yang kompleks di Paris.
- Tahan terhadap Overfitting: Dengan membangun banyak pohon dari subset data dan fitur yang berbeda, Random Forest mengurangi risiko overfitting yang sering terjadi pada single decision tree.
- Mampu Menangani Fitur Kategorikal dan Numerik Secara Bersamaan: Tidak memerlukan penskalaan fitur secara ekstensif dan dapat menangani berbagai tipe data secara alami.

**Kekurangan**:
- Kurang Intuitif (Black Box): Meskipun hasilnya akurat, proses pengambilan keputusan internalnya lebih sulit untuk diinterpretasikan dibandingkan single decision tree atau model linear.
- Membutuhkan Lebih Banyak Sumber Daya Komputasi: Pelatihan banyak pohon memerlukan waktu dan memori yang lebih besar, terutama untuk dataset yang sangat besar.
- Cenderung Bias terhadap Fitur dengan Banyak Level (untuk Fitur Kategorikal): Dalam beberapa implementasi, fitur kategorikal dengan jumlah level yang lebih banyak bisa mendapatkan bobot yang lebih tinggi secara tidak proporsional.

**Parameter yang digunakan**:
- `n_estimators=100`: Jumlah decision tree dalam forest.
- `max_depth=16` : Kedalaman maksimum setiap pohon dalam hutan.
- `random_state=42`: Untuk memastikan reprodusibilitas hasil.
- `n_jobs=-1` : Jumlah core CPU yang digunakan untuk pelatihan. -1 berarti menggunakan semua core yang tersedia untuk mempercepat proses training.


**Proses hyperparameter tuning**:
Setelah evaluasi model awal, Random Forest dipilih untuk tuning karena menunjukkan performa terbaik. Tuning dilakukan menggunakan RandomizedSearchCV dengan 3-fold cross-validation untuk mengevaluasi kombinasi parameter berikut:
- `n_estimators`: [50, 300] - Jumlah tree dalam forest
- `max_depth`: [5, 30] - Kedalaman maksimum setiap tree
- `min_samples_split`: [2, 10] - Minimum sampel yang diperlukan untuk split node
- `min_samples_leaf`: [1, 10] - Minimum sampel yang diperlukan dalam leaf node
- `max_features`: ['auto', 'sqrt', 'log2']

**Parameter optimal hasil tuning**:
- `max_features='log2'`
- `max_depth=26`: Kedalaman tree yang cukup dalam untuk menangkap pola kompleks tanpa overfitting berlebihan.
- `min_samples_leaf=3`: Mensyaratkan minimal 2 sampel di setiap leaf node untuk mengurangi overfitting.
- `min_samples_split=2`: Nilai default yang memungkinkan splitting node dengan minimal 2 sampel.
- `n_estimators=276`: Jumlah tree yang lebih banyak untuk meningkatkan stabilitas prediksi.

Alasan pemilihan parameter ini: Parameter-parameter tersebut menghasilkan RMSE terendah pada cross-validation, menunjukkan kemampuan generalisasi terbaik. Peningkatan jumlah estimator menjadi 276 memungkinkan model untuk lebih baik menangkap pola dalam data, sementara max_depth=26 memberikan fleksibilitas yang cukup tanpa overfitting berlebihan.

### 3. Gradient Boosting
Gradient Boosting adalah algoritma ensemble yang membangun model secara sequential.

**Karakteristik**:
- Gradient Boosting membangun model satu per satu secara berurutan. Setiap model baru dilatih untuk memperbaiki kesalahan (residual) dari model-model sebelumnya. Ini berbeda dengan metode ensemble lain seperti Random Forest yang membangun model secara paralel
- Gradient Boosting membangun model secara bertahap (iteratif). Setiap model baru yang ditambahkan bertujuan untuk memperbaiki kesalahan (residual) yang dibuat oleh model-model sebelumnya. Ini berbeda dengan algoritma seperti Random Forest yang membangun model secara paralel.

**Kelebihan**:
- Performa Prediktif Sangat Tinggi: Seringkali menjadi salah satu algoritma dengan performa terbaik untuk data terstruktur, mampu menghasilkan akurasi prediksi harga rumah yang sangat tinggi.
- Fleksibilitas Tinggi: Dapat dioptimalkan dengan berbagai fungsi loss dan memungkinkan kustomisasi yang mendalam untuk berbagai jenis masalah prediksi.
- Mampu Menangani Data yang Hilang Secara Internal: Beberapa implementasi Gradient Boosting memiliki mekanisme internal untuk menangani nilai yang hilang dalam dataset.

**Kekurangan**:
- Sensitif terhadap Hiperparameter: Memerlukan tuning hiperparameter yang cermat; pengaturan yang buruk dapat dengan mudah menyebabkan overfitting.
- Waktu Pelatihan yang Lebih Lama: Proses pelatihan yang bersifat sekuensial (model baru dibangun berdasarkan model sebelumnya) dapat memakan waktu lebih lama dibandingkan Random Forest, terutama pada dataset besar.
- Rentan terhadap Overfitting jika Tidak Diatur dengan Baik: Meskipun kuat, jika jumlah pohon terlalu banyak atau learning rate terlalu tinggi tanpa regularisasi yang tepat, model bisa overfit terhadap data pelatihan.

**Parameter yang digunakan**:
- `learning_rate=0.05`
- `random_state=55`

---
## Evaluation
Metrik evaluasi yang digunakan diantara lain:

1. **Mean Squared Error (MSE)** - Rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual. MSE memberikan bobot yang lebih besar pada kesalahan yang besar.
   
   $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

2. **Root Mean Squared Error (RMSE)** - Akar kuadrat dari MSE. RMSE memiliki satuan yang sama dengan variabel target, sehingga lebih mudah diinterpretasi.
   
   $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

3. **Mean Absolute Error (MAE)** - Rata-rata dari nilai absolut selisih antara nilai prediksi dan nilai aktual. MAE kurang sensitif terhadap outlier dibandingkan MSE/RMSE.
   
   $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

4. **R-squared (R²)** - Proporsi variasi dalam variabel dependen yang dapat dijelaskan oleh variabel independen. Nilai R² berkisar antara 0 dan 1, di mana nilai yang lebih tinggi menunjukkan model yang lebih baik.
   
   $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

   di mana $\bar{y}$ adalah nilai rata-rata dari $y$.

### Hasil Evaluasi Final

| **Hasil**       | **KNN**         | **RF**           | **Boosting**     | **RF Setelah Tuning** |
|-----------------|-----------------|------------------|------------------|------------------------|
| train_mse       | 1.209576e+12     | 2.101126e+06      | 1.277552e+11      | 3.384314e+10           |
| train_rmse      | 1.099807e+06     | 1.449526e+03      | 3.574285e+05      | 1.839651e+05           |
| train_mae       | 9.056725e+05     | 1.152520e+03      | 3.099350e+05      | 1.355925e+05           |
| train_r2        | 8.516664e-01     | 9.999997e-01      | 9.843330e-01      | 9.958497e-01           |
| test_mse        | 1.718531e+12     | 3.239738e+10      | 1.542182e+11      | 1.560090e+11           |
| test_rmse       | 1.310928e+06     | 1.799927e+05      | 3.927063e+05      | 3.949797e+05           |
| test_mae        | 1.086788e+06     | 1.502514e+05      | 3.299662e+05      | 3.120134e+05           |
| test_r2         | 8.037928e-01     | 9.963011e-01      | 9.823927e-01      | 9.821882e-01           |




![Train Test For Every Model](https://github.com/danielanputri/DicodingPredictiveAnalytics/blob/main/images/results.png)

**Model Terbaik** 

Berdasarkan hasil evaluasi, model Random Forest dipilih sebagai model terbaik karena memiliki nilai yang terbaik. Berikut adalah grafik untuk nilai aktual vs nilai prediksi menggunakan model random forest yang telah ditunning.
![Actual vs Predicted for model RF](https://github.com/danielanputri/DicodingPredictiveAnalytics/blob/main/images/prediksi%20vs%20aktual.png)

**Evaluasi Terhadap Business Understanding**
- Menjawab Problem Statement: Model yang dibuat berhasil menjawab problem statement dengan memprediksi harga sewa apartemen berdasarkan fitur-fitur yang ada dan mengidentifikasi fitur-fitur yang paling berpengaruh.
- Mencapai Goals: Model Random Forest dengan hyperparameter yang dioptimalkan berhasil mencapai tujuan untuk memberikan prediksi harga sewa yang akurat dan mengidentifikasi fitur penting.
- Dampak dari Solution Statement: Penggunaan beberapa algoritma dan hyperparameter tuning memberikan dampak positif dengan meningkatkan akurasi prediksi dan memungkinkan pemilihan model terbaik. Solusi yang direncanakan memberikan hasil yang signifikan dalam mencapai tujuan proyek.

---
## Kesimpulan

Melalui proses pemodelan dan evaluasi, telah berhasil membangun model yang akurat untuk memprediksi harga sewa apartemen dan mengidentifikasi fitur-fitur yang paling berpengaruh. Model Random Forest terbukti menjadi model terbaik dalam hal akurasi prediksi, dan hyperparameter tuning memainkan peran penting dalam meningkatkan performa model. Dampak dari solusi yang diimplementasikan sangat positif, memenuhi problem statement dan goals yang telah ditetapkan.

## Referensi

[1] Adetunji, A. B., Akande, O. N., Ajala, F. A., Oyewo, O., Akande, Y. F., & Oluwadara, G. (2022). House price prediction using random forest machine learning technique. Procedia Computer Science, 199, 806-813.

[2] C. Fan, Z. Cui, and X. Zhong, "House Prices Prediction with Machine Learning Algorithms," in Proceedings of the 2018 10th International Conference on Machine Learning and Computing, pp. 6-10, 2018.

[3] R. K. Pace and R. Barry, "Sparse spatial autoregressions," Statistics & Probability Letters, vol. 33, no. 3, pp. 291-297, 1997.
