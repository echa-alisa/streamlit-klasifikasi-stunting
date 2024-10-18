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

if page == "Model LSTM":
    st.header('Model LSTM')

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
    st.header('Input Data Baru untuk Prediksi Status Kesehatan Bayi')

    if 'model' not in st.session_state:
        st.warning("Model belum dilatih. Silakan latih model terlebih dahulu di halaman 'Model LSTM'.")
    else:
        # Input fitur data baru
        JK = st.selectbox('Jenis Kelamin (0: Laki-laki, 1: Perempuan)', [0, 1])
        Umur = st.number_input('Umur (bulan)', min_value=0, max_value=60)
        Berat = st.number_input('Berat (kg)', min_value=0.0, max_value=30.0)
        Tinggi = st.number_input('Tinggi (cm)', min_value=0.0, max_value=150.0)
        BB_Lahir = st.number_input('Berat Lahir (kg)', min_value=0.0, max_value=5.0)
        TB_Lahir = st.number_input('Tinggi Lahir (cm)', min_value=0.0, max_value=60.0)
        ZS_TB_U = st.number_input('Z-Score Tinggi Badan menurut Umur', min_value=-5.0, max_value=5.0)

        # Tombol untuk prediksi
        if st.button('Prediksi Status'):
            input_data = pd.DataFrame({
                'JK': [JK],
                'Umur': [Umur],
                'Berat': [Berat],
                'Tinggi': [Tinggi],
                'BB_Lahir': [BB_Lahir],
                'TB_Lahir': [TB_Lahir],
                'ZS_TB_U': [ZS_TB_U]
            })

            # Scaling data baru
            scaler = MinMaxScaler()
            input_data_scaled = scaler.fit_transform(input_data)
            input_data_scaled = input_data_scaled.reshape(1, input_data_scaled.shape[1], 1)

            # Ambil model dari session state
            model = st.session_state['model']
            prediksi = model.predict(input_data_scaled)
            prediksi_class = np.argmax(prediksi, axis=1)

            # Definisikan label untuk kelas prediksi
            status = ['Normal', 'Severely Stunting', 'Stunting']

            # Tampilkan hasil prediksi
            st.write(f"Hasil prediksi: {status[prediksi_class[0]]}")
