import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Buat tampilan web menggunakan Streamlit
st.title('Aplikasi Klasifikasi Status Stunting')

# Load dataset langsung di dalam kode
@st.cache_data
def load_data():
    return pd.read_csv('dataset_stunting_skripsi.csv')

df = load_data()

# Buat sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ("Informasi Dataset", "Visualisasi", "Model LSTM", "Input Data Baru"))

if page == "Informasi Dataset":
    st.header("Informasi Dataset")
    st.write("### Apa itu Stunting?")
    st.write("Stunting adalah kondisi di mana tinggi badan seorang anak jauh lebih rendah dibandingkan dengan standar tinggi badan anak seusianya. Hal ini biasanya disebabkan oleh malnutrisi kronis, terutama pada usia dini. Stunting dapat memengaruhi pertumbuhan fisik dan perkembangan kognitif anak, serta berpotensi menyebabkan masalah kesehatan di kemudian hari.")

    st.header("Informasi Dataset")
    st.write("Dataframe:")
    st.write(df)

    st.write("### Deskripsi Dataset")
    st.write("Dataset ini berisi informasi tentang status gizi anak, khususnya terkait dengan stunting. Dataset ini digunakan untuk menganalisis dan memprediksi status gizi anak berdasarkan berbagai faktor, termasuk jenis kelamin, umur, berat badan, tinggi badan, dan beberapa variabel lainnya.")
    
    st.write("### Fitur-Fitur dalam Dataset")
    st.write("- **JK (Jenis Kelamin)**: Kategori yang menunjukkan jenis kelamin anak.")
    st.write("- **Umur**: Usia anak dalam bulan.")
    st.write("- **Berat**: Berat badan anak dalam kilogram.")
    st.write("- **Tinggi**: Tinggi badan anak dalam sentimeter.")
    st.write("- **BB_Lahir**: Berat badan anak saat lahir.")
    st.write("- **TB_Lahir**: Tinggi badan anak saat lahir.")
    st.write("- **ZS_TB_U**: Z-Score tinggi badan menurut umur.")
    st.write("- **Status**: Kategori status anak.")
    
    st.write("### Tujuan Penggunaan Dataset")
    st.write("Dataset ini digunakan untuk menganalisis faktor-faktor yang berkontribusi terhadap stunting pada anak, serta untuk membangun model klasifikasi untuk mengidentifikasi risiko stunting.")
    
    st.write("### Sumber Dataset")
    st.write("Sumber dataset ini berasal dari Dinas Kesehatan Kota Bogor.")

    st.write("### Split Dataset")
    st.write("Pembagian dataset ini menggunakan 80:20, di mana 80% akan digunakan sebagai data test dan 20% sebagai data latih.")

elif page == "Visualisasi":
    st.header("Visualisasi Data")
    
    # Visualisasi distribusi jenis kelamin
    st.subheader("Distribusi Jenis Kelamin")
    st.write("Grafik ini menunjukkan distribusi jenis kelamin dalam dataset. "
             "Dari grafik ini, kita dapat melihat perbandingan antara jumlah anak laki-laki dan perempuan.")
    st.bar_chart(df['JK'].value_counts())

    # Visualisasi distribusi tinggi badan
    st.subheader("Distribusi Tinggi Badan")
    st.write("Grafik ini menunjukkan distribusi tinggi badan anak-anak dalam dataset. "
             "Garis KDE (Kernel Density Estimate) memberikan gambaran yang lebih halus mengenai "
             "distribusi tinggi badan tersebut.")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Tinggi'], kde=True, ax=ax1, color='purple', bins=30)
    st.pyplot(fig1)

    # Visualisasi distribusi Z-Score tinggi badan
    st.subheader("Distribusi Z-Score Tinggi Badan")
    st.write("Grafik ini menunjukkan distribusi Z-Score tinggi badan. Z-Score digunakan untuk "
             "mengukur seberapa jauh tinggi badan anak dari rata-rata tinggi badan populasi sebayanya.")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['ZS_TB_U'], kde=True, ax=ax2, color='green', bins=30)
    st.pyplot(fig2)

    # Visualisasi distribusi Status
    st.subheader("Distribusi Status Gizi")
    st.write("Grafik ini menunjukkan distribusi status gizi anak berdasarkan kategori Status (Normal, Severely Stunting, Stunting).")
    
    # Hitung frekuensi setiap kategori
    status_counts = df['Status'].value_counts()

    # Buat histogram
    fig3, ax3 = plt.subplots()
    sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax3, palette='Set2')
    ax3.set_title('Distribusi Status Gizi Anak')
    ax3.set_xlabel('Status')
    ax3.set_ylabel('Jumlah Anak')
    
    # Tampilkan histogram
    st.pyplot(fig3)

elif page == "Model LSTM":
    # Data preprocessing
    df['BB_Lahir'].replace(0, np.nan, inplace=True)
    df['TB_Lahir'].replace(0, np.nan, inplace=True)
    df = df.dropna()
    df = df.drop(columns=['Tanggal_Pengukuran'])

    # Label Encoding
    encode = LabelEncoder()
    df['JK'] = encode.fit_transform(df['JK'].values)
    df['TB_U'] = encode.fit_transform(df['TB_U'].values)
    df['Status'] = encode.fit_transform(df['Status'].values)

    # Menentukan X dan y
    st.header('Data Selection')
    X = df[['JK', 'Umur', 'Berat', 'Tinggi', 'BB_Lahir', 'TB_Lahir', 'ZS_TB_U']]
    st.write('Features (X):')
    st.write(X)

    y = df['Status']
    st.write('Target (y):')
    st.write(y)

    # Penjelasan tentang features dan target
    st.subheader("Penjelasan tentang Features (X) dan Target (y)")
    st.write("Features (X) adalah variabel independen yang digunakan untuk memprediksi status stunting. "
             "Ini mencakup informasi seperti jenis kelamin, umur, berat badan, tinggi badan, dan data kelahiran. "
             "Target (y) adalah variabel dependen yang ingin kita prediksi, yaitu status stunting anak.")

    st.header('Latih Model LSTM')

    # Skalakan fitur ke rentang [0, 1] menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Terapkan SMOTE pada seluruh dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # One-Hot Encoding untuk label
    y_resampled_encoded = to_categorical(y_resampled, num_classes=3)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled_encoded, test_size=0.2, random_state=42)

    # Menambahkan dimensi waktu (1) ke data pelatihan dan pengujian
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Tidak ada input parameter dari pengguna, hard-code parameter training
    neurons = 64
    epochs = 50
    batch_size = 128
    learning_rate = 0.001

    st.write(f"Jumlah Neuron: {neurons}")
    st.write(f"Jumlah Epoch: {epochs}")
    st.write(f"Batch Size: {batch_size}")
    st.write(f"Learning Rate: {learning_rate}")

    # Tambahkan tombol untuk melatih model
    if st.button('Latih Model'):
        # Membangun Model LSTM
        model = Sequential()
        model.add(LSTM(neurons, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(neurons, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        # Compile Model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Latih model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

        # Simpan model ke dalam session state setelah dilatih
        st.session_state['model'] = model

        # Evaluasi model
        loss, accuracy = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        st.subheader('Evaluasi Model')
        st.write(f"Loss: {loss:.4f}")
        st.write(f"Accuracy: {accuracy:.4f}")

        # Plot accuracy dan loss
        st.subheader('Grafik Akurasi dan Loss')
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
        ax[0].set_title('Accuracy')

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        ax[1].set_title('Loss')

        st.pyplot(fig)

        # Tampilkan Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        st.subheader('Confusion Matrix')
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        # Menghitung metrik evaluasi
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (FP + FN + TP)

        # Akurasi
        accuracy = np.sum(TP) / np.sum(cm)

        # Presisi dan Recall per kelas
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        # F1-Score per kelas
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Menampilkan metrik evaluasi
        st.subheader("Metrik Evaluasi")
        st.write(f"Akurasi: {accuracy:.4f}")

        metrics_df = pd.DataFrame({
            'Kelas': ['Class 0', 'Class 1', 'Class 2'],
            'Presisi': precision,
            'Recall': recall,
            'F1-Score': f1_score
        })

        st.write(metrics_df)

elif page == "Input Data Baru":
    st.header('Input Data Baru untuk Prediksi Status Stunting')

    # Pastikan model sudah dilatih dan disimpan di session state
    if 'model' not in st.session_state:
        st.warning("Model belum dilatih. Silakan latih model terlebih dahulu di halaman 'Jalankan Model'.")
    else:
        # Form input data
        JK = st.selectbox('Jenis Kelamin (0: Laki-laki, 1: Perempuan)', [0, 1])
        Umur = st.number_input('Umur (bulan)', min_value=0, max_value=60)
        Berat = st.number_input('Berat (kg)', min_value=0.0, max_value=30.0)
        Tinggi = st.number_input('Tinggi (cm)', min_value=0.0, max_value=150.0)
        BB_Lahir = st.number_input('Berat Lahir (kg)', min_value=0.0, max_value=5.0)
        TB_Lahir = st.number_input('Tinggi Lahir (cm)', min_value=0.0, max_value=60.0)
        ZS_TB_U = st.number_input('Z-Score Tinggi Badan menurut Umur', min_value=-5.0, max_value=5.0)

        # Tombol untuk melakukan prediksi
        if st.button('Prediksi Status'):
            # Preprocessing data input seperti di halaman model
            input_data = pd.DataFrame({
                'JK': [JK],
                'Umur': [Umur],
                'Berat': [Berat],
                'Tinggi': [Tinggi],
                'BB_Lahir': [BB_Lahir],
                'TB_Lahir': [TB_Lahir],
                'ZS_TB_U': [ZS_TB_U]
            })

            # Skalakan input data
            scaler = MinMaxScaler()
            input_data_scaled = scaler.fit_transform(input_data)

            # Reshape untuk cocok dengan input model
            input_data_scaled = input_data_scaled.reshape(1, input_data_scaled.shape[1], 1)

            # Prediksi dengan model yang sudah dilatih
            model = st.session_state['model']
            prediksi = model.predict(input_data_scaled)
            prediksi_class = np.argmax(prediksi, axis=1)

            # Tampilkan hasil prediksi
            status = ['Normal', 'Saverely Stunting', 'Stunting']
            st.write(f"Hasil prediksi: {status[prediksi_class[0]]}")

            # Prediksi probabilitas untuk input data
            prediksi_prob = model.predict(input_data_scaled)
            prediksi_class = np.argmax(prediksi_prob, axis=1)

            # Tampilkan hasil prediksi dengan probabilitas
            st.write(f"Probabilitas untuk Status Normal: {prediksi_prob[0][0]*100:.2f}%")
            st.write(f"Probabilitas untuk Status Saverely Stunting {prediksi_prob[0][1]*100:.2f}%")
            st.write(f"Probabilitas untuk Status Stunting: {prediksi_prob[0][2]*100:.2f}%")
