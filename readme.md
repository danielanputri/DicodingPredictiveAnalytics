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
Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com/datasets/manjitbaishya001/house-prices-2023). Dataset ini berisi informasi tentang rumah yang disewakan atau dijual di pakistan pada tahun 2023, termasuk fitur-fitur seperti jumlah kamar mandi, kamar tidur, luas apartemen, dan harga.

Dataset ini terdiri dari 99499 sampel dengan 9 fitur dan 1 target, yang mencakup berbagai aspek dari perumahan di Pakistan.

### Variabel-variabel pada Chicago Housing Dataset adalah sebagai berikut:

| **Fitur** | **Deskripsi** |
|:-------:|:-----------:|
| Unnamed: 0 | Kemungkinan adalah indeks atau ID unik yang diberikan untuk setiap baris atau entri properti dalam dataset. |
| property_type | Jenis properti yang dijual atau disewakan. |
| price | Harga properti yang ditawarkan. - **Target Variable** |
| location | Detail lokasi spesifik atau nama area/sektor dari properti tersebut di dalam kota |
| city | Nama kota tempat properti itu berada. |
| baths |Jumlah kamar mandi yang ada di properti tersebut. |
| purpose | Tujuan dari iklan properti tersebut. |
| bedrooms | Jumlah kamar tidur yang tersedia di properti. |
| Area_in_Marla | Luas properti yang diukur dalam satuan Marla. |

---
## Eksplorasi Data Awal (EDA)
Untuk memahami dataset dengan lebih baik, beberapa tahapan eksplorasi data telah dilakukan:

#### 1. Statistik Deskriptif
Statistik deskriptif menunjukkan bahwa:
- Harga (price) memiliki Median 7.500.000 dan Rata-rata 10.375.920
- Jumlah kamar mandi (baths) memiliki Median 3 Rata-rata 3,53
- Jumlah kamar tidur (bedrooms) memiliki Median 3 dan Rata-rata 3,35
- Luas properti (Area_in_Marla) memiliki Median 6,7 Marla dan Rata-rata 8,76 Marla

#### 2. Analisis Missing Values
Dataset ini tidak memiliki missing values.

#### 3. Distribusi Target Variable
Distribusi harga rumah (price) menunjukkan distribusi yang positively skewed, dengan Sebagian besar data berada di sisi harga rendah, Ekor distribusi menjulur panjang ke arah kanan (harga tinggi), serta Rata-rata (mean) lebih besar dari median (terlihat dari garis merah di kanan garis hijau). Hal ini mengindikasikan bahwa harga rumah di Pakistan tidak terdistribusi secara normal, dengan sebagian besar rumah memiliki harga di kisaran menengah, dan sebagian kecil dengan harga sangat tinggi.

#### 4. Cek Outliers dan Menangani Outliers
Terdapat outliers dalam fitur price, Area_in_Marla, baths, dan bathroom. Outliers ditangani dengan cara dihapus menggunakan IQR Method.

#### 4. Analisis Korelasi
Matriks korelasi menunjukkan:
- Baths (Kamar Mandi) memiliki korelasi positif tertinggi dengan harga rumah (r = 0.53)
- Fitur Area_in_Marla (luas bangunan) memiliki hubungan yang cukup kuat dengan baths dan bedrooms, yang menunjukkan bahwa luas bangunan adalah indikator penting dari ukuran dan fasilitas apartemen.

---
## Data Preparation
Beberapa teknik data preparation yang diterapkan dalam proyek ini:

1. **Menghapus Fitur yang Tidak Diperlukan**
   Fitur yang tidak diperlukan (Unnamed:0) dihapus karena fitur ini tidak memiliki pengaruh terhadap pembuatan model.
2. **Encoding Categorical Variables**
   Mengubah variabel kategorikal menjadi variabel numerik menggunakan teknik frequency encoding, one-hot encoding, dan label encoding.
3. **Pembagian Data (Train-Test Split)**
   Data dibagi menjadi set pelatihan (80%) dan pengujian (20%) untuk mengevaluasi performa model pada data yang tidak pernah dilihat sebelumnya. Proses ini penting untuk menghindari overfitting dan mendapatkan estimasi yang tidak bias tentang performa model.
4. **Standardisasi Fitur**
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

**Parameter yang digunakan**:
- n_neighbors=10

### 2. Random Forest Regression
Random Forest adalah algoritma ensemble yang menggunakan multiple decision trees.

**Karakteristik**: 
- Random Forest membangun banyak decision tree (pohon keputusan) secara independen selama proses training.
- Setiap pohon dilatih pada sampel data acak (bootstrap sample) dari dataset asli, dan pada setiap pemisahan (split) di pohon, hanya subset acak dari fitur yang dipertimbangkan.
- Dengan menggabungkan prediksi dari banyak pohon yang beragam (hasil dari pengambilan sampel acak), Random Forest cenderung mengurangi overfitting yang sering terjadi pada satu decision tree dan meningkatkan akurasi prediksi.

**Parameter yang digunakan**:
- `n_estimators=100`: Jumlah decision tree dalam forest.
- `random_state=42`: Untuk memastikan reprodusibilitas hasil.

**Proses hyperparameter tuning**:
Setelah evaluasi model awal, Random Forest dipilih untuk tuning karena menunjukkan performa terbaik. Tuning dilakukan menggunakan GridSearchCV dengan 5-fold cross-validation untuk mengevaluasi kombinasi parameter berikut:
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

### 3. Gradient Boosting
Gradient Boosting adalah algoritma ensemble yang membangun model secara sequential.

**Karakteristik**:
- Gradient Boosting membangun model satu per satu secara berurutan. Setiap model baru dilatih untuk memperbaiki kesalahan (residual) dari model-model sebelumnya. Ini berbeda dengan metode ensemble lain seperti Random Forest yang membangun model secara paralel
- Gradient Boosting membangun model secara bertahap (iteratif). Setiap model baru yang ditambahkan bertujuan untuk memperbaiki kesalahan (residual) yang dibuat oleh model-model sebelumnya. Ini berbeda dengan algoritma seperti Random Forest yang membangun model secara paralel.

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

| **Hasil** | **KNN** | **RF** | **Boosting** | **RF Setelah Tuning** |
|:------------------------|:----------------|:-------------:|:-------------:| :-------------:|
| train_mse |  9.788781e+12 |  5.890061e+12 | 2.178827e+13 | 7.702493e+12 |
| train_rmse | 3.128703e+06 |  2.426945e+06 |  4.667791e+06 | 2.775337e+06 |
| train_mae |   1.808073e+06 |  1.432871e+06 |  3.082030e+06 | 1.675942e+06 |
| train_r2 |    8.755708e-01 |  9.251290e-01 |  7.230403e-01 | 9.020904e-01 |
| test_mse |   1.202162e+13 |  9.391332e+12 | 2.101028e+13 | 9.354967e+12 |
| test_rmse |   3.467221e+06 |  3.064528e+06 | 4.583697e+06 | 3.058589e+06 |
| test_mae |    1.994047e+06 |  1.769583e+06 | 3.025828e+06 | 1.840509e+06 |
| test_r2 |    8.474768e-01 |  8.808483e-01  | 7.334340e-01 | 8.813097e-01 |



![Train Test For Every Model](https://github.com/danielanputri/DicodingPredictiveAnalytics/blob/main/images/result.png)

**Model Terbaik** 

Berdasarkan hasil evaluasi, model Random Forest dipilih sebagai model terbaik karena memiliki nilai yang terbaik. Berikut adalah grafik untuk nilai aktual vs nilai prediksi menggunakan model random forest yang telah ditunning.
![Actual vs Predicted for model RF](https://github.com/danielanputri/DicodingPredictiveAnalytics/blob/main/images/actual%20vs%20predict.png)

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