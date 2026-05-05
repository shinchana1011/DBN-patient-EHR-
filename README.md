# 🧬 DBN-Based Synthetic EHR Generator

A deep learning project for generating **privacy-preserving synthetic healthcare data** using **Deep Belief Networks (DBNs)**.

This system creates realistic Electronic Health Records (EHRs) without using real patient data, enabling safe use of healthcare data for research and machine learning.

---

## 📌 Overview

Healthcare data is sensitive and cannot be freely shared due to privacy regulations.  
This project addresses this problem by:

- Generating synthetic patient records
- Preserving statistical properties of real data
- Preventing memorization of actual patient data

---

## 🧠 Methodology

The project uses a **Deep Belief Network (DBN)**, built by stacking multiple **Restricted Boltzmann Machines (RBMs)**.

- RBM 1 → learns low-level feature relationships  
- RBM 2 → learns disease patterns  
- RBM 3 → learns abstract patient representations  

Training is done using **Contrastive Divergence (CD-k)**, and data is generated using **Gibbs Sampling**.

---

## 📊 Features in Dataset

### 👤 Demographics
- Age, Gender, BMI

### ❤️ Vitals
- Heart Rate, Blood Pressure, Temperature, SpO2, Respiratory Rate

### 🧪 Lab Values
- Glucose, Creatinine, WBC, Hemoglobin, Platelets, Sodium, Potassium, HbA1c

### 🏥 Diagnoses
- Diabetes, Hypertension, CKD, CHF, COPD

### 🎯 Target
- ICU Admission

---

## ⚙️ Tech Stack

- Python  
- NumPy, Pandas  
- Scikit-learn  
- Streamlit  
- Matplotlib, Seaborn  

---

## 🚀 How It Works

1. **Data Generation**
   - Synthetic EHR data is created using probabilistic rules with clinical correlations.

2. **Preprocessing**
   - Continuous features are normalized to [0,1].

3. **DBN Training**
   - RBMs are trained layer-wise using Contrastive Divergence.

4. **Data Generation**
   - Gibbs sampling is used to generate synthetic patient records.

5. **Evaluation**
   - Statistical similarity, ML utility, and privacy are measured.

---

## 📈 Evaluation Metrics

### Statistical Metrics
- KS Test (distribution similarity)
- Correlation Difference

### Machine Learning Utility
- TSTR (Train on Synthetic, Test on Real)
- ROC-AUC Scores

### Privacy Metrics
- DCR (Distance to Closest Record)
- NNDR (Nearest Neighbor Distance Ratio)

---

## 🔐 Privacy Assurance

- Uses L2 distance to measure similarity between records  
- Ensures synthetic data is not too close to real data  

**Safe Condition:**  
NNDR ≥ 0.8 → No memorization of real data  
