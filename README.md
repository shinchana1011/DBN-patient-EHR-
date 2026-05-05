DBN-Based Synthetic EHR Generator

A Deep Learning project for privacy-preserving healthcare data generation using Deep Belief Networks (DBNs).

This system generates realistic synthetic Electronic Health Records (EHRs) while maintaining patient privacy, enabling safe use of healthcare data for machine learning and research.

📌 Overview

Healthcare datasets are sensitive and restricted due to privacy regulations (HIPAA, etc.).
This project solves that problem by:

👉 Generating synthetic patient records
👉 Preserving statistical properties of real data
👉 Preventing data leakage and memorization

🧠 Core Idea

We use a Deep Belief Network (DBN), which is built by stacking multiple:

👉 Restricted Boltzmann Machines (RBMs)

Each layer learns patterns at different levels:

Layer 1 → Low-level correlations (age ↔ BP)
Layer 2 → Disease patterns
Layer 3 → High-level patient representations
🏗️ Project Architecture
Synthetic Data Generator → Preprocessing → DBN (RBM Stack) → Synthetic Data
                                         ↓
                                  Evaluation Metrics
📊 Features Generated
👤 Demographics
Age, Gender, BMI
❤️ Vitals
Heart rate, Blood pressure, Temperature, SpO2, Respiratory rate
🧪 Lab Values
Glucose, Creatinine, WBC, Hemoglobin, Platelets, Sodium, Potassium, HbA1c
🏥 Diagnoses
Diabetes, Hypertension, CKD, CHF, COPD
🎯 Target
ICU Admission
⚙️ Technologies Used
Python 🐍
NumPy, Pandas
Scikit-learn
Streamlit (UI Dashboard)
Matplotlib & Seaborn (Visualization)
🚀 How It Works
Step 1: Generate Synthetic EHR Data
Uses probabilistic rules to simulate patient data
Maintains realistic clinical relationships
Step 2: Preprocessing
Normalize data to range [0,1]
Required for RBM sigmoid activation
Step 3: Train DBN
Greedy layer-wise training
Each RBM learns feature patterns
Step 4: Generate Synthetic Data
Gibbs sampling on top RBM
Data is reconstructed through layers
Step 5: Evaluation
📊 Statistical Metrics
KS Test (distribution similarity)
Correlation difference
🤖 ML Utility
TSTR (Train on Synthetic, Test on Real)
AUC score comparison
🔐 Privacy Metrics
DCR (Distance to Closest Record)
NNDR (Nearest Neighbor Distance Ratio)
📈 Key Results
Synthetic data closely matches real distributions
ML models trained on synthetic data perform well on real data
Privacy is preserved (NNDR ≥ 0.8)
🔐 Privacy Assurance

We ensure no memorization using:

L2 Distance between records
NNDR metric

👉 If NNDR ≥ 0.8 → Safe
👉 Synthetic data is not copying real patients
