"""
Evaluation Metrics for Synthetic EHR Quality

Three dimensions measured:
1. Statistical Fidelity  – Do synthetic/real distributions match?
2. ML Utility            – Can a model trained on synthetic data work on real data?
3. Privacy               – How different are synthetic records from training records?
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA


# ─── 1. Statistical Fidelity ────────────────────────────────────────────────

def column_wise_ks_test(real: np.ndarray, synthetic: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov test per feature.
    H0: real and synthetic come from the same distribution.
    Returns p-values and KS statistics.
    """
    n_features = real.shape[1]
    ks_stats, p_values = [], []
    for j in range(n_features):
        stat, p = stats.ks_2samp(real[:, j], synthetic[:, j])
        ks_stats.append(stat)
        p_values.append(p)
    return {"ks_statistic": np.array(ks_stats), "p_value": np.array(p_values)}


def feature_correlation_difference(real: np.ndarray, synthetic: np.ndarray) -> float:
    """
    Frobenius norm of difference between correlation matrices.
    Lower = better (synthetic captures real correlations).
    """
    real_corr = np.corrcoef(real.T)
    syn_corr = np.corrcoef(synthetic.T)
    diff = real_corr - syn_corr
    # Handle NaN in case of zero-variance columns
    diff = np.nan_to_num(diff)
    return float(np.linalg.norm(diff, "fro"))


def mean_std_similarity(real: np.ndarray, synthetic: np.ndarray) -> dict:
    """Mean absolute difference of per-feature mean and std"""
    mean_diff = np.abs(real.mean(0) - synthetic.mean(0)).mean()
    std_diff = np.abs(real.std(0) - synthetic.std(0)).mean()
    return {"mean_diff": float(mean_diff), "std_diff": float(std_diff)}


# ─── 2. ML Utility (Train on Synthetic, Test on Real — TSTR) ────────────────

def tstr_evaluation(
    real_X, real_y,
    synthetic_X, synthetic_y,
    target_col_idx=-1
):
    """
    Train-on-Synthetic, Test-on-Real evaluation.
    Safe version: handles one-class synthetic labels.
    """

    real_y = np.asarray(real_y).astype(int)
    synthetic_y = np.asarray(synthetic_y).astype(int)

    # Safety check for real labels
    if len(np.unique(real_y)) < 2:
        return {
            "TRTR_AUC_mean": 0.0,
            "TRTR_AUC_std": 0.0,
            "TSTR_LR_AUC": 0.0,
            "TSTR_RF_ACC": 0.0,
            "TSTR_RF_F1": 0.0,
            "TSTR_RF_AUC": 0.0,
            "utility_ratio": 0.0,
            "error": "Real labels contain only one class. Utility evaluation skipped."
        }

    # Safety check for synthetic labels
    if len(np.unique(synthetic_y)) < 2:
        return {
            "TRTR_AUC_mean": 0.0,
            "TRTR_AUC_std": 0.0,
            "TSTR_LR_AUC": 0.0,
            "TSTR_RF_ACC": 0.0,
            "TSTR_RF_F1": 0.0,
            "TSTR_RF_AUC": 0.0,
            "utility_ratio": 0.0,
            "error": "Synthetic labels contain only one class. TSTR evaluation skipped."
        }

    scaler = StandardScaler()

    # TRTR
    Xr_scaled = scaler.fit_transform(real_X)

    trtr_scores = cross_val_score(
        LogisticRegression(max_iter=500, random_state=42),
        Xr_scaled,
        real_y,
        cv=5,
        scoring="roc_auc"
    )

    # TSTR Logistic Regression
    Xs_scaled = scaler.fit_transform(synthetic_X)
    Xr_test_scaled = scaler.transform(real_X)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(Xs_scaled, synthetic_y)

    tstr_prob = clf.predict_proba(Xr_test_scaled)[:, 1]
    tstr_auc = roc_auc_score(real_y, tstr_prob)

    # TSTR Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(synthetic_X, synthetic_y)

    rf_pred = rf.predict(real_X)
    rf_proba = rf.predict_proba(real_X)[:, 1]

    rf_acc = accuracy_score(real_y, rf_pred)
    rf_f1 = f1_score(real_y, rf_pred, zero_division=0)
    rf_auc = roc_auc_score(real_y, rf_proba)

    return {
        "TRTR_AUC_mean": float(trtr_scores.mean()),
        "TRTR_AUC_std": float(trtr_scores.std()),
        "TSTR_LR_AUC": float(tstr_auc),
        "TSTR_RF_ACC": float(rf_acc),
        "TSTR_RF_F1": float(rf_f1),
        "TSTR_RF_AUC": float(rf_auc),
        "utility_ratio": float(tstr_auc / (trtr_scores.mean() + 1e-9)),
    }


# ─── 4. PCA Overlap ─────────────────────────────────────────────────────────

def pca_overlap_data(real: np.ndarray, synthetic: np.ndarray, n_components=2):
    """
    Project both distributions to 2D PCA space.
    Returns coordinates for plotting.
    """
    pca = PCA(n_components=n_components, random_state=42)
    combined = np.vstack([real, synthetic])
    pca.fit(combined)
    real_2d = pca.transform(real)
    syn_2d = pca.transform(synthetic)
    return real_2d, syn_2d, pca.explained_variance_ratio_


# ─── 5. Reconstruction Quality ──────────────────────────────────────────────

def reconstruction_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))

def nearest_neighbor_distance(real: np.ndarray, synthetic: np.ndarray, sample_size=500) -> dict:
    rng = np.random.RandomState(42)

    if real.shape[0] > sample_size:
        idx_r = rng.choice(real.shape[0], sample_size, replace=False)
        real_s = real[idx_r]
    else:
        real_s = real

    if synthetic.shape[0] > sample_size:
        idx_s = rng.choice(synthetic.shape[0], sample_size, replace=False)
        syn_s = synthetic[idx_s]
    else:
        syn_s = synthetic

    syn_real_dists = []
    for s in syn_s:
        dists = np.linalg.norm(real_s - s, axis=1)
        syn_real_dists.append(dists.min())

    real_real_dists = []
    for i, r in enumerate(real_s):
        others = np.delete(real_s, i, axis=0)
        dists = np.linalg.norm(others - r, axis=1)
        real_real_dists.append(dists.min())

    dcr_mean = float(np.mean(syn_real_dists))
    real_mean = float(np.mean(real_real_dists))

    nndr = dcr_mean / (real_mean + 1e-9)

    return {
        "DCR_mean": dcr_mean,
        "DCR_std": float(np.std(syn_real_dists)),
        "NNDR": float(nndr),
        "privacy_safe": nndr >= 0.8,
    }


# ─── 6. Summary Report ──────────────────────────────────────────────────────

def full_evaluation_report(real_X, real_y, syn_X, syn_y):
    """Run all metrics and return a unified dict"""
    ks = column_wise_ks_test(real_X, syn_X)
    corr_diff = feature_correlation_difference(real_X, syn_X)
    ms = mean_std_similarity(real_X, syn_X)
    tstr = tstr_evaluation(real_X, real_y, syn_X, syn_y)
    priv = nearest_neighbor_distance(real_X, syn_X)
    pct_features_pass_ks = float((ks["p_value"] > 0.05).mean() * 100)

    return {
        "statistical": {
            "pct_features_pass_ks": pct_features_pass_ks,
            "corr_matrix_diff": corr_diff,
            "mean_feature_diff": ms["mean_diff"],
            "std_feature_diff": ms["std_diff"],
            "ks_stats": ks["ks_statistic"].tolist(),
            "ks_pvalues": ks["p_value"].tolist(),
        },
        "utility": tstr,
        "privacy": priv,
    }

