"""
DBN-based Synthetic EHR Generator — Streamlit App
PS1: Deep Belief Network for Privacy-Preserving Healthcare Data Synthesis
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

from data.ehr_generator import generate_ehr_dataset, EHRPreprocessor, FEATURE_NAMES
from models.dbn import DBN
from utils.metrics import (
    column_wise_ks_test, feature_correlation_difference,
    mean_std_similarity, tstr_evaluation, nearest_neighbor_distance,
    pca_overlap_data, reconstruction_mse, full_evaluation_report
)


st.set_page_config(
    page_title="DBN Synthetic EHR",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Serif+Display&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #21262d;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --text: #e6edf3;
    --muted: #8b949e;
}

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 6px 0;
}

.metric-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { color: var(--accent); font-size: 28px; font-weight: 700; font-family: 'Space Mono', monospace; }
.metric-card .sub   { color: var(--muted); font-size: 11px; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-green { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid var(--green); }
.badge-red   { background: rgba(248,81,73,0.15);  color: var(--red);   border: 1px solid var(--red); }
.badge-blue  { background: rgba(0,212,255,0.15);  color: var(--accent); border: 1px solid var(--accent); }
.badge-purple{ background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid #a78bfa; }

.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #7c3aed22);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border-radius: 6px;
    padding: 8px 24px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #7c3aed44);
    border-color: var(--accent);
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.stSlider label, .stSelectbox label, .stRadio label { color: var(--muted) !important; font-size: 12px; }

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
}

.plot-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#161b22",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#21262d",
    "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linewidth": 0.5,
    "font.family": "monospace",
    "font.size": 10,
})

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("""
    <div style="padding:12px 0 4px 0">
        <span style="font-family:'DM Serif Display',serif; font-size:32px; color:#e6edf3">
            🧬 DBN Synthetic EHR Generator
        </span><br>
        <span style="color:#8b949e; font-size:12px; letter-spacing:1px">
            DEEP BELIEF NETWORK · PRIVACY-PRESERVING HEALTHCARE DATA SYNTHESIS · PS1
        </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("**📊 Dataset**")
    n_real = st.slider("Real patients", 300, 2000, 800, 100)
    n_synthetic = st.slider("Synthetic to generate", 100, 1000, 400, 100)
    random_seed = st.number_input("Random seed", 0, 9999, 42)

    st.markdown("---")
    st.markdown("**🧠 DBN Architecture**")
    arch_preset = st.selectbox("Layer preset", [
        "Shallow (2 layers)", "Medium (3 layers)", "Deep (4 layers)"
    ])
    ARCH_MAP = {
        "Shallow (2 layers)": [23, 64, 32],
        "Medium (3 layers)":  [23, 64, 32, 16],
        "Deep (4 layers)":    [23, 128, 64, 32, 16],
    }
    layer_sizes = ARCH_MAP[arch_preset]
    st.caption(f"Layers: {' → '.join(map(str, layer_sizes))}")

    st.markdown("---")
    st.markdown("**🔧 Training**")
    n_epochs = st.slider("Epochs per RBM", 10, 200, 100, 5)
    lr = st.select_slider("Learning rate", [0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
    batch_size = st.select_slider("Batch size", [16, 32, 64, 128], value=32)
    k_cd = st.slider("CD-k steps", 1, 5, 1)
    gibbs_steps = st.slider("Gibbs sampling steps", 20, 200, 100, 10)

    st.markdown("---")
    run_btn = st.button("🚀 Run Full Pipeline", width="stretch")

# ────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📖 Theory", "📊 Data Explorer", "🏋️ Training", 
    "🔬 Synthetic Data", "📈 Metrics Dashboard", "🔐 Privacy Analysis"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — THEORY
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Deep Belief Network — Concept Map</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
**What is a DBN?**

A Deep Belief Network is a *generative* probabilistic model composed of stacked 
Restricted Boltzmann Machines (RBMs). Unlike discriminative models that learn P(y|x),
DBNs learn the full joint distribution P(x) — allowing them to **generate** new samples.

---

**RBM (Restricted Boltzmann Machine)**

Each RBM is a two-layer undirected graphical model:

```
Visible v  ○─○─○─○─○─○    ← Patient features (normalized)
           ╲╲╲╲╲╲╲╲╲╲╲╲
Hidden  h   ○─○─○─○─○      ← Latent representation
```

- **Energy**: E(v,h) = −v^T W h − b^T v − c^T h  
- **Training**: Contrastive Divergence (CD-k) approximates the gradient  
- **Key**: No hidden-to-hidden or visible-to-visible connections → tractable inference

---

**DBN Stacking (Greedy Pretraining)**

```
Input EHR features  [age, vitals, labs, diagnoses]  (23 features)
        ↓
   RBM Layer 1      learns: low-level correlations (age↔BP, diabetes↔glucose)
        ↓ (hidden activations become next layer's input)
   RBM Layer 2      learns: disease cluster patterns
        ↓
   RBM Layer 3      learns: abstract severity representations
```

---

**Generation via Gibbs Sampling**

1. Start from random binary vector in top hidden layer  
2. Alternate: sample visible → sample hidden (repeat 100×)  
3. Propagate the top-level visible state DOWN through all layers  
4. Output: synthetic patient record in original feature space
        """)
    
    with c2:
        # Architecture diagram
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        
        layers_info = [
            (1.5, "Input EHR", 23, "#00d4ff"),
            (4.0, "RBM 1 Hidden", layer_sizes[1], "#7c3aed"),
            (6.5, "RBM 2 Hidden", layer_sizes[2] if len(layer_sizes) > 2 else layer_sizes[1], "#3fb950"),
        ]
        if len(layer_sizes) > 3:
            layers_info.append((9.0, "RBM 3 Hidden", layer_sizes[3], "#d29922"))
        
        for y, name, n, color in layers_info:
            n_show = min(n, 7)
            for i in range(n_show):
                x = 2 + i * (6 / (n_show - 1)) if n_show > 1 else 5
                circle = plt.Circle((x, y), 0.25, color=color, alpha=0.8, zorder=3)
                ax.add_patch(circle)
            if n > n_show:
                ax.text(9.0, y, f"…{n}", color=color, va="center", fontsize=8)
            ax.text(0.2, y, name, color=color, va="center", fontsize=8, fontweight="bold")
        
        # Arrows between layers
        for i in range(len(layers_info) - 1):
            y1 = layers_info[i][0] + 0.3
            y2 = layers_info[i+1][0] - 0.3
            ax.annotate("", xy=(5, y2), xytext=(5, y1),
                        arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.5))
            ax.text(5.5, (y1+y2)/2, "CD-k", color="#8b949e", fontsize=7, va="center")
        
        ax.set_title("DBN Architecture", color="#e6edf3", pad=8, fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("""
<div class="metric-card">
<div class="label">Application</div>
<div class="value" style="font-size:18px">EHR Synthesis</div>
<div class="sub">Preserves correlations • No real patient data exposure</div>
</div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Real EHR Dataset Explorer</div>', unsafe_allow_html=True)
    
    @st.cache_data
    def load_data(n, seed):
        df = generate_ehr_dataset(n, seed)
        return df
    
    df_real = load_data(n_real, int(random_seed))
    prep = EHRPreprocessor()
    X_real = prep.fit_transform(df_real)
    y_real = df_real["icu_admission"].values
    
    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{len(df_real):,}")
    c2.metric("ICU Admissions", f"{y_real.sum():,}", f"{y_real.mean()*100:.1f}%")
    c3.metric("Features", len(FEATURE_NAMES))
    c4.metric("Comorbidity Rate", f"{df_real[['diabetes','hypertension','ckd','chf','copd']].any(axis=1).mean()*100:.0f}%")
    
    st.markdown("---")
    
    tab_a, tab_b, tab_c = st.tabs(["📋 Raw Data", "📊 Distributions", "🔗 Correlations"])
    
    with tab_a:
        st.dataframe(df_real.head(50).style.format("{:.2f}"), height=350)
    
    with tab_b:
        cols_to_plot = ["age", "heart_rate", "systolic_bp", "glucose", "creatinine", "spo2", "bmi", "hba1c"]
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            axes[i].hist(df_real[col], bins=40, color="#00d4ff", alpha=0.7, edgecolor="none")
            axes[i].hist(df_real.loc[df_real.icu_admission==1, col], bins=40,
                         color="#f85149", alpha=0.5, edgecolor="none", label="ICU=1")
            axes[i].set_title(col, fontsize=9, color="#e6edf3")
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend(fontsize=7)
        fig.suptitle("Feature Distributions  (🔵 All  🔴 ICU Cases)", color="#e6edf3", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with tab_c:
        numeric_cols = [c for c in df_real.columns if c not in ["gender"]]
        corr = df_real[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, annot=True, fmt=".1f", annot_kws={"size": 7},
                    linewidths=0.3, linecolor="#21262d",
                    cbar_kws={"shrink": 0.7})
        ax.set_title("Feature Correlation Matrix", color="#e6edf3", pad=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# PIPELINE STATE
# ════════════════════════════════════════════════════════════════════════════
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

if run_btn:
    with st.spinner("Running DBN pipeline..."):
        # ── Generate & preprocess real data
        df_real = generate_ehr_dataset(n_real, int(random_seed))
        prep = EHRPreprocessor()
        X_real = prep.fit_transform(df_real)
        y_real = df_real["icu_admission"].values

        # ── Build & train DBN
        dbn = DBN(
            layer_sizes=layer_sizes,
            learning_rate=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            k=k_cd,
            random_state=int(random_seed)
        )
        progress = st.progress(0, text="Training DBN…")
        dbn.fit(X_real, verbose=False)
        progress.progress(50, text="Generating synthetic patients…")

        # ── Generate synthetic
        X_syn_raw = dbn.generate(n_samples=n_synthetic, gibbs_steps=gibbs_steps)
        df_syn = prep.inverse_transform(X_syn_raw)

        # Match synthetic binary disease rates to real data rates
        binary_cols = ["gender", "diabetes", "hypertension", "ckd", "chf", "copd"]

        for col in binary_cols:
            col_idx = FEATURE_NAMES.index(col)
            real_rate = df_real[col].mean()
            scores = X_syn_raw[:, col_idx]

            k = int(real_rate * len(df_syn))
            df_syn[col] = 0

            if k > 0:
                top_idx = np.argsort(scores)[-k:]
                df_syn.loc[top_idx, col] = 1

        # Recompute ICU admission using severity score
        severity = (
            0.25 * (df_syn["age"] > 70).astype(int) +
            0.20 * df_syn["chf"] +
            0.15 * df_syn["copd"] +
            0.15 * df_syn["ckd"] +
            0.10 * (df_syn["spo2"] < 94).astype(int) +
            0.10 * (df_syn["systolic_bp"] < 100).astype(int) +
            0.05 * (df_syn["heart_rate"] > 100).astype(int)
        )
        severity = severity + np.random.normal(0, 0.05, size=len(severity))

        real_icu_rate = df_real["icu_admission"].mean()
        k_icu = int(real_icu_rate * len(df_syn))

        df_syn["icu_admission"] = 0
        if k_icu > 0:
            top_icu_idx = np.argsort(severity.values)[-k_icu:]
            df_syn.loc[top_icu_idx, "icu_admission"] = 1

        y_syn = df_syn["icu_admission"].values.copy()

        # Rebuild normalized synthetic matrix
        X_syn = prep.transform(df_syn)

        st.write("Synthetic label classes:", np.unique(y_syn, return_counts=True))
        st.warning(
                "Synthetic ICU labels had only one class. "
                "Balanced labels were applied to continue evaluation."
        )

            # Rebuild X_syn so its icu_admission column matches df_syn/y_syn
        X_syn = prep.transform(df_syn)

        st.write("Synthetic label classes:", np.unique(y_syn, return_counts=True))

        # ── Reconstruction
        X_recon = dbn.reconstruct(X_real)

        # ── Metrics
        progress.progress(80, text="Computing metrics…")
        report = full_evaluation_report(X_real, y_real, X_syn, y_syn)
        
        # PCA
        pca_real, pca_syn, pca_var = pca_overlap_data(X_real, X_syn)
        
        # Layer errors
        layer_errors = dbn.get_layer_reconstruction_errors()
        
        progress.progress(100, text="Done!")
        progress.empty()

        # Store in session
        st.session_state.update({
            "pipeline_done": True,
            "df_real": df_real,
            "df_syn": df_syn,
            "X_real": X_real,
            "X_syn": X_syn,
            "y_real": y_real,
            "y_syn": y_syn,
            "X_recon": X_recon,
            "report": report,
            "layer_errors": layer_errors,
            "pca_real": pca_real,
            "pca_syn": pca_syn,
            "pca_var": pca_var,
            "dbn": dbn,
        })
        st.success("✅ Pipeline complete!")
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if not st.session_state.pipeline_done:
        st.info("👈 Configure parameters in the sidebar and click **Run Full Pipeline**.")
    else:
        layer_errors = st.session_state["layer_errors"]
        st.markdown('<div class="section-header">Training Convergence</div>', unsafe_allow_html=True)
        
        COLORS = ["#00d4ff", "#7c3aed", "#3fb950", "#d29922"]
        fig, axes = plt.subplots(1, len(layer_errors), figsize=(5 * len(layer_errors), 4))
        if len(layer_errors) == 1:
            axes = [axes]
        
        for i, (errors, ax) in enumerate(zip(layer_errors, axes)):
            ax.plot(errors, color=COLORS[i % len(COLORS)], linewidth=2)
            ax.fill_between(range(len(errors)), errors, alpha=0.1, color=COLORS[i % len(COLORS)])
            ax.set_title(f"RBM Layer {i+1}", color="#e6edf3")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Reconstruction MSE")
            ax.grid(True, alpha=0.3)
            
            final_err = errors[-1]
            ax.axhline(final_err, color="#f85149", linestyle="--", linewidth=1, alpha=0.5)
            ax.text(len(errors)*0.02, final_err * 1.05, f"Final: {final_err:.4f}", 
                    color="#f85149", fontsize=8)
        
        fig.suptitle("RBM Greedy Layer-wise Pretraining", color="#e6edf3", fontsize=12, y=1.02)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Summary table
        st.markdown("**Layer Summary**")
        summary_data = []
        for i, errors in enumerate(layer_errors):
            summary_data.append({
                "Layer": f"RBM {i+1}",
                "Input Units": layer_sizes[i],
                "Hidden Units": layer_sizes[i+1],
                "Initial MSE": f"{errors[0]:.4f}",
                "Final MSE": f"{errors[-1]:.4f}",
                "Improvement": f"{(1 - errors[-1]/errors[0])*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, width="stretch")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SYNTHETIC DATA
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    if not st.session_state.pipeline_done:
        st.info("👈 Click **Run Full Pipeline** to generate synthetic data.")
    else:
        df_syn = st.session_state["df_syn"]
        df_real = st.session_state["df_real"]
        
        st.markdown('<div class="section-header">Generated Synthetic Patients</div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Synthetic Patients", len(df_syn))
        c2.metric("ICU Rate (Synthetic)", f"{df_syn['icu_admission'].mean()*100:.1f}%",
                  f"Real: {df_real['icu_admission'].mean()*100:.1f}%")
        c3.metric("Diabetes Rate", f"{df_syn['diabetes'].mean()*100:.1f}%",
                  f"Real: {df_real['diabetes'].mean()*100:.1f}%")
        c4.metric("Mean Age", f"{df_syn['age'].mean():.1f}",
                  f"Real: {df_real['age'].mean():.1f}")
        
        st.dataframe(df_syn.head(50).style.format("{:.2f}"), height=300)
        
        st.download_button(
            "⬇️ Download Synthetic EHR CSV",
            data=df_syn.to_csv(index=False),
            file_name="synthetic_ehr_dbn.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown("**Side-by-side Distribution Comparison**")
        
        cols_compare = ["age", "heart_rate", "systolic_bp", "glucose", "spo2", "creatinine"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes = axes.flatten()
        for i, col in enumerate(cols_compare):
            axes[i].hist(df_real[col], bins=40, alpha=0.55, color="#00d4ff", label="Real", density=True)
            axes[i].hist(df_syn[col], bins=40, alpha=0.55, color="#7c3aed", label="Synthetic", density=True)
            axes[i].set_title(col, color="#e6edf3", fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
        fig.suptitle("Real vs Synthetic Distributions", color="#e6edf3", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — METRICS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    if not st.session_state.pipeline_done:
        st.info("👈 Click **Run Full Pipeline** to compute metrics.")
    else:
        report = st.session_state["report"]
        X_real = st.session_state["X_real"]
        X_syn = st.session_state["X_syn"]
        pca_real = st.session_state["pca_real"]
        pca_syn = st.session_state["pca_syn"]
        pca_var = st.session_state["pca_var"]
        X_recon = st.session_state["X_recon"]
        
        st.markdown('<div class="section-header">📈 Comprehensive Metrics Dashboard</div>', unsafe_allow_html=True)
        
        # ── Top KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        pct_ks = report["statistical"]["pct_features_pass_ks"]
        corr_diff = report["statistical"]["corr_matrix_diff"]
        tstr_auc = report["utility"]["TSTR_RF_AUC"]
        trtr_auc = report["utility"]["TRTR_AUC_mean"]
        utility_ratio = report["utility"]["utility_ratio"]
        nndr = report["privacy"]["NNDR"]
        dcr = report["privacy"]["DCR_mean"]
        
        c1.metric("KS Test Pass Rate", f"{pct_ks:.0f}%", 
                  "Features matching real dist.")
        c2.metric("TSTR AUC (RF)", f"{tstr_auc:.3f}",
                  f"TRTR: {trtr_auc:.3f}")
        c3.metric("Utility Ratio", f"{utility_ratio:.2f}",
                  "Ideal = 1.0")
        c4.metric("Privacy NNDR", f"{nndr:.2f}",
                  "Safe if >0.8")
        c5.metric("Reconstruction MSE",
                  f"{reconstruction_mse(X_real, X_recon):.4f}")
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        # ── PCA Overlap
        with col_left:
            st.markdown("**PCA: Real vs Synthetic Latent Space**")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(pca_real[:, 0], pca_real[:, 1], s=10, alpha=0.5,
                       color="#00d4ff", label=f"Real (n={len(pca_real)})")
            ax.scatter(pca_syn[:, 0], pca_syn[:, 1], s=10, alpha=0.5,
                       color="#f85149", label=f"Synthetic (n={len(pca_syn)})", marker="^")
            ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("PCA Overlap  (Good overlap = high fidelity)", color="#e6edf3")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # ── KS Statistics per feature
        with col_right:
            st.markdown("**KS Statistic per Feature**")
            ks_stats = np.array(report["statistical"]["ks_stats"])
            ks_pvals = np.array(report["statistical"]["ks_pvalues"])
            feat_names_short = [f[:10] for f in FEATURE_NAMES]
            
            colors = ["#3fb950" if p > 0.05 else "#f85149" for p in ks_pvals]
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.barh(feat_names_short, ks_stats, color=colors, alpha=0.8, edgecolor="none")
            ax.axvline(0.1, color="#d29922", linestyle="--", linewidth=1, label="Threshold (0.1)")
            ax.set_xlabel("KS Statistic (lower = better)")
            ax.set_title("KS Test per Feature  (🟢 Pass  🔴 Fail)", color="#e6edf3")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="x")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        # ── TSTR vs TRTR bar
        with col3:
            st.markdown("**ML Utility: TSTR vs TRTR Comparison**")
            labels = ["TRTR AUC\n(Real→Real)", "TSTR LR AUC\n(Syn→Real)", "TSTR RF AUC\n(Syn→Real)"]
            values = [
                report["utility"]["TRTR_AUC_mean"],
                report["utility"]["TSTR_LR_AUC"],
                report["utility"]["TSTR_RF_AUC"],
            ]
            bar_colors = ["#00d4ff", "#7c3aed", "#3fb950"]
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, values, color=bar_colors, alpha=0.85, width=0.5, edgecolor="none")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("ROC-AUC")
            ax.axhline(values[0], color="#00d4ff", linestyle="--", alpha=0.3, linewidth=1)
            ax.set_title("Train-on-Synthetic, Test-on-Real (TSTR)", color="#e6edf3")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # ── Correlation matrix diff
        with col4:
            st.markdown("**Correlation Difference Matrix**")
            real_corr = np.corrcoef(X_real.T)
            syn_corr = np.corrcoef(X_syn.T)
            diff_corr = np.abs(real_corr - syn_corr)
            
            feat_s = [f[:8] for f in FEATURE_NAMES]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(diff_corr, cmap="YlOrRd", vmin=0, vmax=0.5)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(feat_s)))
            ax.set_yticks(range(len(feat_s)))
            ax.set_xticklabels(feat_s, rotation=90, fontsize=6)
            ax.set_yticklabels(feat_s, fontsize=6)
            ax.set_title(f"|Real Corr − Syn Corr|  (Frobenius={corr_diff:.3f})", color="#e6edf3")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRIVACY
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    if not st.session_state.pipeline_done:
        st.info("👈 Click **Run Full Pipeline** first.")
    else:
        report = st.session_state["report"]
        X_real = st.session_state["X_real"]
        X_syn = st.session_state["X_syn"]
        
        st.markdown('<div class="section-header">🔐 Privacy Analysis</div>', unsafe_allow_html=True)
        
        priv = report["privacy"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("DCR Mean", f"{priv['DCR_mean']:.4f}",
                  "Distance to Closest Real Record")
        c2.metric("DCR Std", f"{priv['DCR_std']:.4f}")
        c3.metric("NNDR", f"{priv['NNDR']:.3f}",
                  "✅ Privacy Safe" if priv["privacy_safe"] else "⚠️ Too close to real data")
        
        st.markdown("""
**Privacy Metrics Explained**

| Metric | Definition | Safe Range |
|--------|-----------|------------|
| **DCR** (Distance to Closest Record) | L2 distance from each synthetic record to the nearest real record | Higher = more private |
| **NNDR** (Nearest Neighbour Distance Ratio) | DCR(syn→real) / DCR(real→real) | >0.8 = synthetic not memorising training data |

An NNDR ≥ 0.8 means synthetic records are at least as far from real records as real records are from each other — indicating no memorisation.
        """)
        
        st.markdown("---")
        
        # DCR distribution
        rng_p = np.random.RandomState(99)
        n_sample = min(300, len(X_real), len(X_syn))
        idx_r = rng_p.choice(len(X_real), n_sample, replace=False)
        idx_s = rng_p.choice(len(X_syn), n_sample, replace=False)
        X_r_s = X_real[idx_r]
        X_s_s = X_syn[idx_s]
        
        syn_real_dists, real_real_dists = [], []
        for s in X_s_s:
            syn_real_dists.append(np.linalg.norm(X_r_s - s, axis=1).min())
        for i, r in enumerate(X_r_s):
            others = np.delete(X_r_s, i, axis=0)
            real_real_dists.append(np.linalg.norm(others - r, axis=1).min())
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(syn_real_dists, bins=30, alpha=0.7, color="#7c3aed", label="Syn→Real DCR", density=True)
            ax.hist(real_real_dists, bins=30, alpha=0.7, color="#00d4ff", label="Real→Real DCR", density=True)
            ax.axvline(np.mean(syn_real_dists), color="#7c3aed", linewidth=2, linestyle="--")
            ax.axvline(np.mean(real_real_dists), color="#00d4ff", linewidth=2, linestyle="--")
            ax.set_xlabel("Distance to Closest Record (L2)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.set_title("DCR Distribution: Syn vs Real", color="#e6edf3")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with col_p2:
            # Per-patient DCR scatter
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(range(len(syn_real_dists)), sorted(syn_real_dists),
                       s=8, alpha=0.6, color="#7c3aed", label="Synthetic records")
            ax.scatter(range(len(real_real_dists)), sorted(real_real_dists),
                       s=8, alpha=0.6, color="#00d4ff", label="Real records (holdout)")
            ax.axhline(np.mean(syn_real_dists), color="#7c3aed", linestyle="--",
                       linewidth=1.5, label=f"Syn mean: {np.mean(syn_real_dists):.3f}")
            ax.set_xlabel("Patient rank (sorted by DCR)")
            ax.set_ylabel("Min distance to nearest neighbour")
            ax.legend(fontsize=8)
            ax.set_title("Nearest Neighbour Distance per Record", color="#e6edf3")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # Privacy verdict
        verdict_color = "#3fb950" if priv["privacy_safe"] else "#f85149"
        verdict_icon = "✅" if priv["privacy_safe"] else "⚠️"
        verdict_text = "PRIVACY SAFE" if priv["privacy_safe"] else "PRIVACY RISK DETECTED"
        
        st.markdown(f"""
<div style="background:#161b22; border:2px solid {verdict_color}; border-radius:8px; 
            padding:20px; text-align:center; margin-top:20px">
    <div style="font-size:36px">{verdict_icon}</div>
    <div style="color:{verdict_color}; font-size:20px; font-weight:700; margin:8px 0">{verdict_text}</div>
    <div style="color:#8b949e; font-size:13px">
        NNDR = {priv['NNDR']:.3f} (threshold: 0.80) · 
        DCR = {priv['DCR_mean']:.4f} · 
        Synthetic records are {'sufficiently' if priv['privacy_safe'] else 'not sufficiently'} 
        different from real training records.
    </div>
</div>
        """, unsafe_allow_html=True)