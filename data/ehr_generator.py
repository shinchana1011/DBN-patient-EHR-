"""
Synthetic EHR Dataset Generator

Mimics MIMIC-III style structure with realistic clinical correlations.
No real patient data needed — fully synthetic but statistically plausible.

Features Generated:
    Demographics:  age, gender, bmi
    Vitals:        heart_rate, systolic_bp, diastolic_bp, temperature, spo2, respiratory_rate
    Labs:          glucose, creatinine, wbc, hemoglobin, platelets, sodium, potassium, hba1c
    Diagnoses:     diabetes, hypertension, ckd, chf, copd (binary flags)
    Outcome:       icu_admission (target for downstream ML)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURE_NAMES = [
    # Demographics
    "age", "gender", "bmi",
    # Vitals
    "heart_rate", "systolic_bp", "diastolic_bp",
    "temperature", "spo2", "respiratory_rate",
    # Labs
    "glucose", "creatinine", "wbc", "hemoglobin",
    "platelets", "sodium", "potassium", "hba1c",
    # Diagnoses (binary)
    "diabetes", "hypertension", "ckd", "chf", "copd",
    # Target
    "icu_admission"
]

N_FEATURES = len(FEATURE_NAMES)


def generate_ehr_dataset(n_samples=1000, random_state=42):
    """
    Generate synthetic EHR data with realistic clinical correlations.
    
    Correlations modeled:
        - Diabetic patients → higher glucose, HbA1c
        - Hypertensive patients → higher BP
        - CKD patients → higher creatinine
        - CHF/COPD patients → lower SpO2, higher respiratory rate
        - Severe cases → ICU admission
    """
    rng = np.random.RandomState(random_state)
    n = n_samples

    # ── Demographics
    age = rng.normal(62, 15, n).clip(18, 95)
    gender = rng.binomial(1, 0.52, n)          # 0=F, 1=M
    bmi = rng.normal(27, 5, n).clip(15, 55)

    # ── Comorbidities (correlated with age)
    p_diabetes = 0.15 + 0.003 * (age - 40).clip(0)
    diabetes = rng.binomial(1, p_diabetes.clip(0, 0.6))

    p_hypertension = 0.20 + 0.005 * (age - 40).clip(0)
    hypertension = rng.binomial(1, p_hypertension.clip(0, 0.75))

    p_ckd = 0.05 + 0.003 * (age - 40).clip(0) + 0.1 * diabetes
    ckd = rng.binomial(1, p_ckd.clip(0, 0.5))

    p_chf = 0.05 + 0.003 * (age - 50).clip(0) + 0.1 * hypertension
    chf = rng.binomial(1, p_chf.clip(0, 0.4))

    p_copd = 0.05 + 0.002 * (age - 50).clip(0)
    copd = rng.binomial(1, p_copd.clip(0, 0.35))

    # ── Vitals (correlated with comorbidities) ─────────────────────────────
    heart_rate = rng.normal(78 + 5*chf + 3*copd, 12, n).clip(40, 160)
    systolic_bp = rng.normal(125 + 20*hypertension + 5*(age > 65).astype(int), 18, n).clip(70, 220)
    diastolic_bp = (systolic_bp * 0.6 + rng.normal(0, 8, n)).clip(40, 130)
    temperature = rng.normal(37.0, 0.5, n).clip(35, 40)
    spo2 = rng.normal(97 - 3*chf - 4*copd, 1.5, n).clip(80, 100)
    respiratory_rate = rng.normal(16 + 4*chf + 5*copd, 3, n).clip(8, 40)

    glucose = rng.normal(100 + 80*diabetes, 30, n).clip(50, 600)
    creatinine = rng.normal(0.9 + 2.5*ckd, 0.4, n).clip(0.4, 12)
    wbc = rng.normal(8.0, 2.5, n).clip(1, 30)
    hemoglobin = rng.normal(13.5 - 1.5*ckd - 0.5*chf, 1.5, n).clip(5, 18)
    platelets = rng.normal(250, 70, n).clip(20, 700)
    sodium = rng.normal(139 - 2*chf, 4, n).clip(120, 160)
    potassium = rng.normal(4.0 + 0.5*ckd, 0.5, n).clip(2.5, 7.0)
    hba1c = rng.normal(5.7 + 2.0*diabetes, 0.8, n).clip(4.5, 14)

    # ── ICU Admission (outcome) 
    severity_score = (
        0.3 * (age > 70).astype(int) +
        0.2 * chf + 0.2 * copd + 0.15 * ckd +
        0.1 * (spo2 < 92).astype(int) +
        0.1 * (systolic_bp < 90).astype(int) +
        0.05 * (heart_rate > 110).astype(int) +
        rng.normal(0, 0.1, n)
    )
    p_icu = 1 / (1 + np.exp(-3 * (severity_score - 0.5)))
    icu_admission = rng.binomial(1, p_icu.clip(0.02, 0.95))

    # ── Assemble DataFrame
    data = pd.DataFrame({
        "age": age, "gender": gender, "bmi": bmi,
        "heart_rate": heart_rate, "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp, "temperature": temperature,
        "spo2": spo2, "respiratory_rate": respiratory_rate,
        "glucose": glucose, "creatinine": creatinine, "wbc": wbc,
        "hemoglobin": hemoglobin, "platelets": platelets,
        "sodium": sodium, "potassium": potassium, "hba1c": hba1c,
        "diabetes": diabetes, "hypertension": hypertension,
        "ckd": ckd, "chf": chf, "copd": copd,
        "icu_admission": icu_admission
    })

    return data


class EHRPreprocessor:
    """Normalise continuous features to [0,1] for RBM binary/sigmoid units"""

    CONTINUOUS_COLS = [
        "age", "bmi", "heart_rate", "systolic_bp", "diastolic_bp",
        "temperature", "spo2", "respiratory_rate", "glucose", "creatinine",
        "wbc", "hemoglobin", "platelets", "sodium", "potassium", "hba1c"
    ]

    BINARY_COLS = [
        "gender", "diabetes", "hypertension", "ckd", "chf", "copd", "icu_admission"
    ]

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False

    def fit_transform(self, df):
        df = df.copy()
        df[self.CONTINUOUS_COLS] = self.scaler.fit_transform(df[self.CONTINUOUS_COLS])
        self.fitted = True
        return df[FEATURE_NAMES].values.astype(float)

    def transform(self, df):
        df = df.copy()
        df[self.CONTINUOUS_COLS] = self.scaler.transform(df[self.CONTINUOUS_COLS])
        return df[FEATURE_NAMES].values.astype(float)

    def inverse_transform(self, X):
        """Convert normalised array back to clinical values"""
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df[self.CONTINUOUS_COLS] = self.scaler.inverse_transform(df[self.CONTINUOUS_COLS])
      
        thresholds = {
                "gender": 0.5,
                "diabetes": 0.4,
                "hypertension": 0.35,
                "ckd": 0.40,
                "chf": 0.40,
                "copd": 0.40,
                "icu_admission": 0.5
                }
        for col in self.BINARY_COLS:
            df[col] = (df[col] > thresholds[col]).astype(int)
        
        return df