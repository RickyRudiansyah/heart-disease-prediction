# Heart Disease Prediction (Population Screening)

Proyek ini membangun **sistem peringatan dini** berbasis machine learning untuk memprediksi risiko penyakit jantung dari **data survei populasi (subset BRFSS 2022)**. Fokus metrik: **Recall tinggi** agar kasus berisiko tidak terlewat.

> **Author:** Ricky Rudiansyah — BINUS (Computer Science, Database Technology)  
> **Tools:** Python, Pandas, scikit-learn, (opsional) XGBoost, Matplotlib, Streamlit

---

## 📂 Struktur
project-penyakit-jantung/
dataset_preview_clean/ # sample 2k & head (sudah preprocessing)
dataset_preview_raw/ # sample 2k & head (raw)
notebooks/
Full_Code.ipynb # end-to-end: EDA → prep → modeling → evaluasi
screenshots/ # hasil visual
app.py # (opsional) prototype Streamlit
Link Dataset,Presentasi,Demo.txt
Sistem peringatan dini untuk skrining risiko penyakit jantung.pdf


---

## 📊 Dataset
Subset terkurasi dari **CDC BRFSS 2022** (Behavioral Risk Factor Surveillance System).  
Variabel contoh: `General_Health`, `Checkup`, `Exercise`, `Heart_Disease` (target), `Diabetes`, `Smoking_History`, `Alcohol_Consumption`, `Fruit_Consumption`, `Green_Vegetables_Consumption`, `FriedPotato_Consumption`, `Sex`, `Age_Category`, `Height_(cm)`, `Weight_(kg)`, `BMI`.  
**Data source:** subset/rekoding BRFSS 2022. Versi kurasi yang mirip: 
https://www.kaggle.com/datasets/harshwardhanfartale/cardiovascular-disease-risk-prediction
> Catatan: kolom & label sudah disederhanakan dari versi resmi (subset/rekoding). Data penuh tidak di-commit.

---

## ⚙️ Metodologi
- **CRISP-DM**: Understanding → Preparation → Modeling → Evaluation → (Prototype)
- **Preprocessing**: encoding kategorikal, penanganan missing/imbalance (bila perlu)
- **Model**: Logistic Regression, Random Forest, (opsional) XGBoost
- **Validasi**: Stratified K-Fold
- **Metrik utama**: Recall & AUC

---

## 📈 Hasil & Visual (contoh)
- Logistic Regression: **Recall ≈ 0.79**, **AUC ≈ 0.82**  
- Random Forest / XGBoost: bandingkan di notebook

**Cuplikan visual**## 📈 Hasil & Visual

- Confusion Matrix  
  ![Confusion Matrix](project-penyakit-jantung/screenshots/confusion_matrix_LogisticRegression_HeartDisease_v1.0.png)

- ROC Curve  
  ![ROC](project-penyakit-jantung/screenshots/roc_curve_comparison_v2.png)

- Heatmap  
  ![Heatmap](project-penyakit-jantung/screenshots/heatmap.png)

- Feature Importance  
  ![Feature Importance](project-penyakit-jantung/screenshots/feature_importance_LogisticRegression.png)

- Distribusi Target  
  ![Distribusi Target](project-penyakit-jantung/screenshots/distribusi_target_heart_disease.png)

- Distribusi Usia vs BMI  
  ![Distribusi Usia vs BMI](project-penyakit-jantung/screenshots/distribusi_usia_vs_bmi.png)

- Prototype App (Streamlit)  
  ![Dashboard](project-penyakit-jantung/screenshots/Dashboard.png)
  ![Dashboard 2](project-penyakit-jantung/screenshots/Dashboard_2.png)
  ![Dashboard 3](project-penyakit-jantung/screenshots/Dashboard_3.png)
  ![Dashboard 4](project-penyakit-jantung/screenshots/Dashboard_4.png)
