# ======================================================================================
# DASHBOARD SKRINING RISIKO PENYAKIT JANTUNG
# Versi: 4.2
# Model: LogisticRegression_v1.0
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import altair as alt
import time

# --- 1. KONFIGURASI HALAMAN & GAYA VISUAL ---
st.set_page_config(
    page_title="Skrining Risiko Jantung",
    layout="wide",
    page_icon="❤️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
    .risk-high { background-color: #FFF0F0; border-left: 6px solid #D9534F; padding: 1.5rem; border-radius: 10px; color: #333333; }
    .risk-high h3 { color: #A83C38; }
    .risk-low { background-color: #F0FFF0; border-left: 6px solid #5CB85C; padding: 1.5rem; border-radius: 10px; color: #333333; }
    .risk-low h3 { color: #3E8E41; }
    .sidebar-metric { background-color: #ffffff; border: 1px solid #e6e6e6; padding: 10px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .sidebar-metric p { margin-bottom: 5px; color: #333; }
</style>
""", unsafe_allow_html=True)

# --- 2. MUAT MODEL & ASET LAINNYA ---
@st.cache_resource
def load_model_assets():
    """Memuat pipeline model dan data background untuk SHAP."""
    try:
        model_pipeline = joblib.load("LogisticRegression_HeartDisease_v1.0.pkl")
        shap_background_data = joblib.load("shap_background_LogisticRegression_HeartDisease_v1.0.joblib")
        return model_pipeline, shap_background_data
    except FileNotFoundError:
        st.error("❌ Gagal: Pastikan file `LogisticRegression_HeartDisease_v1.0.pkl` dan `shap_background...` ada di direktori yang sama.")
        return None, None
    except Exception as e:
        st.error(f"❌ Gagal memuat aset model: {e}")
        return None, None

model_pipeline, shap_background_data = load_model_assets()
if not model_pipeline or shap_background_data is None:
    st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Kalkulator Skrining Risiko")
    st.markdown("---")
    st.subheader("Performa Model Acuan")
    
    st.markdown("""
    <div class="sidebar-metric"> <p>🎯 <strong>Recall (Sensitivitas):</strong> 79%</p> </div>
    <div class="sidebar-metric"> <p>🔍 <strong>Precision:</strong> 19%</p> </div>
    <div class="sidebar-metric"> <p>📈 <strong>AUC Score:</strong> 0.82</p> </div>
    """, unsafe_allow_html=True)
    st.caption("Model ini dioptimalkan untuk **Recall tinggi** (menemukan sebanyak mungkin kasus berisiko), yang ideal untuk skrining awal.")

    st.markdown("---")
    st.subheader("Panduan Penggunaan")
    st.info("1. Isi 7 faktor risiko utama pada form.\n2. Klik tombol 'Analisis Risiko'.\n3. Lihat hasil dan interpretasinya.")
    
    st.markdown("---")
    st.warning("**Disclaimer:** Alat ini adalah alat skrining, bukan alat diagnosis medis. Selalu konsultasikan dengan dokter untuk nasihat medis.")

# --- 4. HEADER UTAMA ---
st.title("💓 Kalkulator Skrining Risiko Penyakit Jantung")
st.markdown("Gunakan alat ini sebagai **langkah awal** untuk memahami profil risiko Anda berdasarkan 7 faktor kunci. Cocok untuk keperluan skrining umum.")

# --- 5. FORM INPUT ---
with st.form(key="user_input_form"):
    st.subheader("📝 Masukkan 7 Faktor Risiko Anda")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MODIFIKASI: Nilai default usia diatur ke 0
        age = st.number_input("1. Usia Anda", min_value=0, max_value=100, value=0, help="Usia minimal untuk prediksi adalah 18 tahun.")
        
        with st.expander("Lihat Panduan Skala Kesehatan Umum"):
            st.markdown("- **Sangat Baik:** Merasa prima, berenergi tinggi.\n- **Baik:** Merasa bugar dan sehat.\n- **Cukup:** Kondisi standar, tanpa keluhan berarti.\n- **Kurang:** Sering lesu atau punya keluhan ringan.\n- **Buruk:** Kesehatan membatasi aktivitas harian.")
        
        general_health_map = {"Buruk": 0, "Kurang": 1, "Cukup": 2, "Baik": 3, "Sangat Baik": 4}
        general_health_selection = st.selectbox("2. Bagaimana Anda menilai kesehatan Anda secara umum?", options=general_health_map.keys(), index=2)
        
        diabetes_map = {'Tidak': 0, 'Tidak, tapi pra-diabetes': 0.5, 'Ya': 1, 'Ya, tapi hanya saat hamil': 0.75}
        diabetes_selection = st.selectbox("3. Riwayat Penyakit Diabetes Anda?", options=diabetes_map.keys())

    with col2:
        exercise = st.radio("4. Apakah Anda rutin berolahraga (min. 30 menit/hari)?", ['Ya', 'Tidak'], horizontal=True, index=1)
        smoking_history = st.radio("5. Apakah Anda memiliki riwayat merokok?", ['Ya', 'Tidak'], horizontal=True, index=1)
        arthritis = st.radio("6. Apakah Anda menderita radang sendi (Arthritis)?", ['Ya', 'Tidak'], horizontal=True, index=1)
        
        st.markdown("---")
        st.markdown("**7. Indeks Massa Tubuh (BMI)**")
        height = st.number_input("Tinggi Badan (cm)", min_value=0.0, max_value=250.0, value=0.0)
        weight = st.number_input("Berat Badan (kg)", min_value=0.0, max_value=250.0, value=0.0)

    submit_button = st.form_submit_button("Analisis Risiko Saya 🔍")

# --- 6. VALIDASI & PREDIKSI ---
if submit_button:
    # MODIFIKASI BARU: Validasi untuk usia ditambahkan di paling atas
    if age < 18:
        st.error("⚠️ Mohon masukkan usia yang valid. Usia minimal untuk prediksi adalah 18 tahun.")
    elif height == 0.0 or weight == 0.0:
        st.error("⚠️ Mohon masukkan Tinggi dan Berat Badan Anda. Nilai tidak boleh 0.")
    else:
        with st.status("Data Anda sedang diproses...", expanded=True) as status:
            st.write("Memvalidasi input...")
            time.sleep(0.5)
            bmi = weight / ((height / 100) ** 2)
            if bmi < 15 or bmi > 60:
                st.error(f"⚠️ BMI Anda ({bmi:.1f}) terlihat tidak wajar. Mohon periksa kembali input Tinggi dan Berat Badan Anda.")
                status.update(label="Proses Gagal: BMI tidak valid", state="error")
                st.stop()
            
            st.write("Mempersiapkan data untuk model...")
            time.sleep(0.5)
            def age_to_category(age):
                if age < 25: return 0
                elif age < 30: return 1
                elif age < 35: return 2
                elif age < 40: return 3
                elif age < 45: return 4
                elif age < 50: return 5
                elif age < 55: return 6
                elif age < 60: return 7
                elif age < 65: return 8
                elif age < 70: return 9
                elif age < 75: return 10
                elif age < 80: return 11
                else: return 12

            input_data = pd.DataFrame({
                'Age_Category': [age_to_category(age)],
                'General_Health': [general_health_map[general_health_selection]],
                'Diabetes': [diabetes_map[diabetes_selection]],
                'Arthritis': [1 if arthritis == 'Ya' else 0],
                'Smoking_History': [1 if smoking_history == 'Ya' else 0],
                'Exercise': [1 if exercise == 'Ya' else 0],
                'BMI': [bmi]
            })
            
            st.write("Menjalankan prediksi risiko...")
            time.sleep(1)
            try:
                proba = model_pipeline.predict_proba(input_data)[0][1]
                result = (proba >= 0.5).astype(int)
                status.update(label="✅ Analisis Selesai!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses prediksi: {e}")
                status.update(label="Proses Gagal", state="error")
                st.stop()
        
        st.subheader("📊 Hasil Skrining Anda")
        
        if result == 1:
            st.markdown(f'<div class="risk-high"><h3>⚠️ TERIDENTIFIKASI BERISIKO TINGGI</h3>'
                        f'<p>Berdasarkan data Anda, skor risiko Anda adalah <strong>{proba:.1%}</strong>. '
                        f'Skor ini menempatkan Anda pada kelompok yang memerlukan perhatian lebih lanjut.</p><hr>'
                        f'<strong>Langkah Selanjutnya yang Direkomendasikan:</strong>'
                        f'<ul><li>Ini <strong>BUKAN diagnosis</strong>, melainkan sinyal awal.</li>'
                        f'<li>Sangat disarankan untuk menjadwalkan konsultasi dengan dokter atau fasilitas kesehatan.</li>'
                        f'<li>Tunjukkan hasil ini kepada dokter sebagai bahan diskusi awal.</li></ul></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low"><h3>✅ TERIDENTIFIKASI BERISIKO RENDAH</h3>'
                        f'<p>Berdasarkan data Anda, skor risiko Anda adalah <strong>{proba:.1%}</strong>. '
                        f'Ini adalah hasil yang baik dan menunjukkan profil risiko yang lebih rendah.</p><hr>'
                        f'<strong>Langkah Selanjutnya yang Direkomendasikan:</strong>'
                        f'<ul><li>Terus pertahankan dan tingkatkan gaya hidup sehat Anda.</li>'
                        f'<li>Lakukan check-up kesehatan rutin sesuai anjuran dokter.</li></ul></div>', unsafe_allow_html=True)

        st.subheader("📈 Faktor yang Mempengaruhi Skor Anda")
        
        with st.spinner('Menghasilkan grafik analisis faktor risiko...'):
            try:
                feature_translation = {
                    'Age_Category': 'Kategori Usia', 'General_Health': 'Kesehatan Umum', 
                    'Diabetes': 'Riwayat Diabetes', 'Arthritis': 'Riwayat Arthritis', 
                    'Smoking_History': 'Riwayat Merokok', 'Exercise': 'Aktivitas Olahraga', 
                    'BMI': 'Indeks Massa Tubuh'
                }
                
                preprocessor = model_pipeline.named_steps['preprocessor']
                classifier = model_pipeline.named_steps['classifier']
                data_processed = preprocessor.transform(input_data)
                
                explainer = shap.LinearExplainer(classifier, shap_background_data)
                shap_values = explainer.shap_values(data_processed)
                
                shap_df = pd.DataFrame({
                    'Faktor Risiko': [feature_translation.get(col, col) for col in input_data.columns],
                    'Kontribusi': shap_values[0]
                })
                shap_df['Warna'] = ['Meningkatkan Risiko' if x > 0 else 'Menurunkan Risiko' for x in shap_df['Kontribusi']]
                
                chart = alt.Chart(shap_df).mark_bar().encode(
                    x=alt.X('Kontribusi:Q', title='Kontribusi terhadap Skor Risiko'),
                    y=alt.Y('Faktor Risiko:N', sort='-x', title='Faktor Risiko'),
                    color=alt.Color('Warna:N',
                        scale=alt.Scale(domain=['Meningkatkan Risiko', 'Menurunkan Risiko'], range=['#D9534F', '#5CB85C']),
                        legend=alt.Legend(title="Efek")
                    ),
                    tooltip=[alt.Tooltip('Faktor Risiko:N'), alt.Tooltip('Kontribusi:Q', format='.4f')]
                ).properties(title='Kontribusi Setiap Faktor pada Hasil Skrining Anda')
                
                st.altair_chart(chart, use_container_width=True)
                st.caption("Grafik ini menunjukkan seberapa besar setiap faktor mendorong skor Anda ke arah 'berisiko tinggi' (merah) atau 'berisiko rendah' (hijau).")

            except Exception as e:
                st.warning(f"⚠️ Gagal menampilkan analisis faktor risiko: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2025 | Dibuat untuk Tujuan Portofolio | **Bukan Pengganti Nasihat Medis Profesional**")