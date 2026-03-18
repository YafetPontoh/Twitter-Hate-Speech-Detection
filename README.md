# Twitter-Hate-Speech-Detection
Twitter Hate Speech &amp; Racism Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-green)

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk membangun model **Natural Language Processing (NLP)** yang secara spesifik mampu mendeteksi dan mengklasifikasikan *tweet* yang mengandung ujaran kebencian (*Hate Speech*), rasisme, dan seksisme. 

Meskipun sering disamakan dengan *Sentiment Analysis*, proyek ini berfokus pada **Hate Speech Detection**, di mana AI diajarkan untuk membedakan antara keluhan/emosi negatif biasa dengan serangan yang berlandaskan SARA (Suku, Agama, Ras, dan Antargolongan).

## 🧠 Batasan Masalah & Definisi Label (Strict Definition)
Satu hal krusial dalam klasifikasi teks adalah mendefinisikan batas antara "Ujaran Kebencian" dan "Kata Kasar/Toxic". Dalam proyek ini, kita menggunakan definisi yang ketat:
* **Label 1 (Hate Speech / Racist / Sexist):** Cuitan yang secara eksplisit menyerang identitas kelompok tertentu.
* **Label 0 (Normal / Non-Hate):** Cuitan normal sehari-hari. **PENTING:** Kata-kata kasar atau makian (*insult/toxic*) yang diucapkan karena marah (tanpa konteks SARA) akan tetap diklasifikasikan sebagai Label 0.

Pendekatan ini membuat model menjadi "spesialis" yang tidak mudah tertipu oleh emosi marah biasa dan murni fokus menangkap propaganda rasisme/seksisme.

## 📊 Exploratory Data Analysis (EDA)
Melalui analisis ekstraksi teks (*Wordcloud*), ditemukan kontras yang sangat kuat pada pembagian kelas di dalam dataset:

* **Kelas Target (Label 1):** Didominasi oleh kata kunci berkonteks identitas dan SARA seperti `racist`, `white`, `black`, `hate`, dan `allahsoil`.
* **Kelas Mayoritas (Label 0):** Didominasi oleh sentimen kehidupan sehari-hari seperti `love`, `happy`, `day`, dan `smile`.

**WORDCLOUD HATE SPEECH**
<img width="859" height="683" alt="image" src="https://github.com/user-attachments/assets/70557ea8-e4c8-4345-b1d1-e90d9fc71348" />
<img width="822" height="667" alt="image" src="https://github.com/user-attachments/assets/b268002e-a818-4303-bcbb-c047f27a5d64" />


## 🔬 Model Benchmarking (Eksperimen Bertahap)
Dataset ini memiliki tantangan berupa kelas yang sangat tidak seimbang (*highly imbalanced*), di mana Label 1 (Hate Speech) hanya berjumlah sekitar 7%. Oleh karena itu, metrik evaluasi utama yang diincar adalah **F1-Score** pada kelas minoritas.

Proyek ini dilakukan dalam dua tahap eksperimen:

### Tahap 1: Traditional Machine Learning (Baseline)
Pendekatan pertama menggunakan ekstraksi fitur **TF-IDF (Term Frequency-Inverse Document Frequency)** dengan N-Grams (1,2). Tiga algoritma klasik diuji secara komparatif:
1. **Naive Bayes**
2. **Logistic Regression**
3. **Linear SVM**

**Hasil Tahap 1:** **Linear SVM** keluar sebagai algoritma tradisional terbaik dengan Akurasi ~94% dan **F1-Score: 0.60**. 
*Alasan Upgrade:* Meskipun hasilnya lumayan, TF-IDF memiliki kelemahan mendasar: ia hanya menghitung frekuensi kata dan **mengabaikan urutan serta konteks semantik kalimat**. Untuk mendeteksi *hate speech* yang lebih implisit, kita butuh model yang bisa "membaca" urutan kata.

### Tahap 2: Deep Learning Expansion
Untuk mengatasi kelemahan TF-IDF, teks diubah representasinya menggunakan **Word Embedding** (sehingga AI paham makna kemiripan kata). Fitur ini kemudian disuapkan ke dalam 4 arsitektur *Deep Learning* yang memang dirancang untuk data berurutan (*sequential*):
1. **LSTM (Long Short-Term Memory)**
2. **Bi-LSTM (Bidirectional LSTM)**
3. **GRU (Gated Recurrent Unit)**
4. **CNN 1D (Convolutional Neural Network for Text)**

## ⚙️ Hyperparameter Tuning & Optimalisasi
Untuk mendapatkan hasil yang stabil dan mencegah *Overfitting*, dilakukan penyesuaian (*tuning*) pada arsitektur Deep Learning:
* **Random Seed (42):** Dikunci untuk memastikan hasil eksperimen konsisten dan dapat direproduksi.
* **Learning Rate (0.0005):** Diturunkan menggunakan *Adam Optimizer* agar proses konvergensi (*gradient descent*) berjalan mulus.
* **Batch Size (32):** Diperkecil untuk *update* bobot AI yang lebih teliti.
* **Early Stopping:** Dipasang dengan `patience=2` dan `restore_best_weights=True` untuk mengerem proses *training* dan mengunci otak AI pada *Epoch* dengan performa tertinggi secara otomatis.

## 🏆 Hasil Akhir (Sang Juara: CNN 1D)
Setelah dilakukan *tuning*, arsitektur **CNN 1D** secara mengejutkan keluar sebagai model yang paling stabil, cepat, dan presisi untuk dataset ini!

**Performa Akhir CNN 1D (Diuji pada Data Validasi):**
* Waktu Training: ~50 detik
* Akurasi Global: **94.49%**
* Precision: **0.59**
* Recall: **0.71**
* **F1-Score:** **0.65** *(Peningkatan signifikan dari baseline ML Klasik)*

### 📈 Grafik Learning Curve (CNN 1D)
Grafik pergerakan Loss dan Accuracy di bawah ini menunjukkan bahwa model berhasil belajar dengan sangat baik (*textbook perfect U-shape curve pada Validation Loss*) sebelum diselamatkan oleh *Early Stopping* di Epoch ke-4.

<img width="830" height="337" alt="image" src="https://github.com/user-attachments/assets/8451499f-6df0-4370-b16c-208a4563b7f5" />

## 🛠️ Cara Menggunakan (Reproducibility)
Model otak AI dan kamus (*Tokenizer*) telah diekspor agar dapat digunakan langsung pada aplikasi atau *website* tanpa perlu *training* ulang.
1. `model_lstm_hatespeech.keras` -> Arsitektur dan bobot AI yang sudah dilatih. *(Note: Sesuaikan nama file jika kamu menyimpannya dengan nama cnn1d)*
2. `tokenizer_hatespeech.pickle` -> Kamus konversi kata ke representasi angka.

### 💻 Instalasi (Reproducibility)
Jika Anda ingin menjalankan ulang *notebook* ini di *local environment*, pastikan Anda sudah menginstal seluruh *dependencies* yang dibutuhkan.

Buka terminal dan jalankan perintah berikut:
```bash
pip install -r requirements.txt
