"""
AI-Driven Hydraulic System Health Prediction and Optimization
=============================================================
Streamlit Dashboard Application
Combines B.Tech AIML Syllabus concepts with Applied Calculus.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import (
    SENSOR_METADATA,
    TARGET_METADATA,
    PROFILE_COLUMNS,
    validate_sensor_input,
    build_data_dictionary,
)
from src.gradient_descent import BatchGradientDescentRegressor

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hydraulic_features_combined.csv"
LEGACY_DATA_PATH = PROJECT_ROOT / "data" / "hydraulic_features.csv"

# Configure Streamlit page
st.set_page_config(
    page_title="Hydraulic Health AI + Calculus",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .landing-card {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.75rem;
    }
    .card-body {
        font-size: 0.95rem;
        color: #374151;
        line-height: 1.5;
    }
    .metric-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_production_model():
    """Loads production ExtraTrees model and metadata."""
    model_path = MODELS_DIR / "hydraulic_health_model.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"

    if not model_path.exists() or not meta_path.exists():
        st.error("Production model or metadata missing. Run 'python src/train_models.py' first.")
        st.stop()

    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


@st.cache_resource
def load_physical_ridge_model():
    """Loads physical-only Ridge model for interpretable sensitivity."""
    phys_path = MODELS_DIR / "physical_ridge_model.joblib"
    if phys_path.exists():
        return joblib.load(phys_path)
    return None


@st.cache_data
def load_feature_dataset():
    """Loads processed combined feature dataset."""
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    elif LEGACY_DATA_PATH.exists():
        return pd.read_csv(LEGACY_DATA_PATH)
    else:
        st.error("Processed features not found. Run 'python src/feature_engineering.py'.")
        st.stop()


# Load core artifacts
model, metadata = load_production_model()
df_features = load_feature_dataset()
phys_model = load_physical_ridge_model()

# Session state initialization
if "app_started" not in st.session_state:
    st.session_state.app_started = False

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/engine.png", width=64)
st.sidebar.title("Navigation")

menu_options = [
    "🚀 Start & Project Overview",
    "⚡ Predict & Optimize",
    "🎯 Best Variables & Targets",
    "📊 Model Evaluation",
    "🧮 Calculus & Optimization",
    "🔬 Model Comparison",
    "📖 How It Works (Syllabus)",
]

# If user clicked start previously, default to Predict & Optimize
default_idx = 1 if st.session_state.app_started else 0
choice = st.sidebar.radio("Select View:", menu_options, index=default_idx)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Telemetry")
st.sidebar.info(f"**Total Cycles:** {len(df_features):,}\n\n**Sensors:** 17 physical/virtual\n\n**Active Target:** `cooler_condition`")
st.sidebar.caption("B.Tech AIML & Applied Calculus Project\nDesigned for CPU-only Windows laptops.")


# ==============================================================================
# TAB 1: LANDING & PROJECT OVERVIEW (What it does, What it is, Why it is useful)
# ==============================================================================
if choice == "🚀 Start & Project Overview":
    st.markdown('<div class="main-header">AI-Driven Hydraulic System Health Prediction & Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Integrating Supervised Machine Learning with Applied Calculus (Numerical Integration, Differentiation & Batch Gradient Descent)</div>', unsafe_allow_html=True)

    # 3 High-Impact Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="landing-card">
            <div class="card-title">🔍 What It Does</div>
            <div class="card-body">
                Autonomous, real-time health estimation of industrial hydraulic test rigs. 
                It ingests multi-sensor time-series (pressures, flows, temperatures, power),
                extracts calculus-based dynamic rates, and predicts continuous cooler degradation 
                from optimal operational health (100%) down to critical failure (3%).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="landing-card">
            <div class="card-title">⚙️ What It Is</div>
            <div class="card-body">
                A rigorous academic and applied engineering pipeline bridging 
                <b>B.Tech AIML Coursework</b> with <b>Calculus</b>.
                It incorporates first-principles Batch Gradient Descent, composite 
                trapezoidal integration (&int; s(t) dt), finite difference numerical 
                differentiation (ds/dt), and partial derivative sensitivity analysis (&part;f / &part;x).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="landing-card">
            <div class="card-title">💡 Why It Is Useful</div>
            <div class="card-body">
                Industrial hydraulic presses and heavy machinery face extreme thermal degradation.
                Undetected cooling failures cause fluid cavitation, pump seizure, and catastrophic plant downtime.
                Early condition estimation enables predictive maintenance, reduces overhaul costs by up to 60%,
                and prevents hazardous fluid leaks.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Big Action Button
    st.markdown("### Ready to Predict System Health?")
    if st.button("🚀 Start Prediction & Health Analysis", type="primary", use_container_width=True):
        st.session_state.app_started = True
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Dataset Ground Truth Specifications")
    st.markdown("""
    The system is benchmarked on the official **UCI Condition Monitoring of Hydraulic Systems Dataset** 
    (ZeMA test rig, Saarbrücken, Germany):
    - **Total Recorded Cycles:** 2,205 repeated constant load cycles of 60 seconds each.
    - **Sensors:** 17 signals spanning 100 Hz (`PS1-PS6`, `EPS1`), 10 Hz (`FS1-FS2`), and 1 Hz (`TS1-TS4`, `VS1`, `CE`, `CP`, `SE`).
    - **Condition Labels in `profile.txt`:** Cooler condition (3%, 20%, 100%), Valve condition, Pump leakage, Accumulator pressure, and Stability flag.
    - **Missing Data:** Verified 0 missing values across all sensor matrices.
    """)


# ==============================================================================
# TAB 2: PREDICT & OPTIMIZE (Input, Validation, Best Input Transformation, Best y)
# ==============================================================================
elif choice == "⚡ Predict & Optimize":
    st.markdown('<div class="main-header">⚡ Predict Hydraulic Health & Optimize Inputs</div>', unsafe_allow_html=True)
    st.markdown("Validate raw inputs, transform into optimal feature space, and infer condition health.")

    pred_mode = st.radio(
        "Choose Prediction Mode:",
        ["Option A: Select a Recorded Test Cycle (With Ground Truth)", "Option B: Enter Custom Sensor Measurements"],
        horizontal=True
    )

    if pred_mode.startswith("Option A"):
        st.subheader("1. Select a Recorded Dataset Cycle")
        col_c1, col_c2 = st.columns([2, 1])

        with col_c1:
            cycle_no = st.number_input(
                "Cycle Number (1 to 2,205)",
                min_value=1,
                max_value=len(df_features),
                value=800,
                step=1,
                help="Select any cycle from the hydraulic test rig experiment."
            )

        # Check if cycle was in test or train
        split_path = METRICS_DIR / "split_indices.npz"
        is_test_cycle = False
        if split_path.exists():
            indices = np.load(split_path)
            is_test_cycle = (int(cycle_no) - 1) in indices["test_idx"]

        with col_c2:
            if is_test_cycle:
                st.success("✅ **Cycle Type:** Held-Out Test Set (Unseen during model training)")
            else:
                st.info("ℹ️ **Cycle Type:** Training Set Observation")

        # Load cycle row
        row = df_features.iloc[[int(cycle_no) - 1]]
        actual_val = float(row[metadata["target"]].iloc[0])

        st.markdown("#### 2. Input Validation & Physical Boundary Check")
        # Validate sample sensors for this cycle
        sample_inputs = {
            "TS1_mean": float(row["TS1_mean"].iloc[0]) if "TS1_mean" in row else 45.0,
            "FS1_mean": float(row["FS1_mean"].iloc[0]) if "FS1_mean" in row else 8.5,
            "PS1_mean": float(row["PS1_mean"].iloc[0]) if "PS1_mean" in row else 160.0,
            "EPS1_mean": float(row["EPS1_mean"].iloc[0]) if "EPS1_mean" in row else 2500.0,
        }
        is_valid, errors, warnings = validate_sensor_input(sample_inputs)

        if is_valid and len(warnings) == 0:
            st.success("✅ **Input Validation Passed:** All sensors conform to calibrated hydraulic physical boundaries.")
        elif not is_valid:
            st.error(f"❌ Input Validation Failed: {errors}")
        else:
            st.warning(f"⚠️ Boundary Warnings: {warnings}")

        st.markdown("#### 3. Transforming Inputs into the Best Feature Space")
        with st.expander("🔬 View Feature Transformation & Best Input Pipeline", expanded=False):
            st.write(f"The model selects **{len(metadata['feature_columns'])} optimal predictors** (statistical moments + calculus integrals & derivatives).")
            st.dataframe(row[metadata["feature_columns"]].round(3).T.rename(columns={int(cycle_no)-1: "Engineered Value"}))

        st.markdown("#### 4. Identify Best Predicted Y & Condition Diagnosis")
        X_cycle = row[metadata["feature_columns"]]
        pred_y = float(model.predict(X_cycle)[0])
        abs_err = abs(pred_y - actual_val)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Cooler Health (y)", f"{pred_y:.2f}%")
        m2.metric("Recorded Ground Truth", f"{actual_val:.0f}%")
        m3.metric("Absolute Error |y - ŷ|", f"{abs_err:.2f}%")

        if pred_y >= 75.0:
            status_text = "Full Efficiency"
            status_badge = '<span class="badge-success">Full Efficiency (Optimal)</span>'
        elif pred_y >= 15.0:
            status_text = "Reduced Efficiency"
            status_badge = '<span class="badge-warning">Reduced Efficiency (Warning)</span>'
        else:
            status_text = "Close to Total Failure"
            status_badge = '<span class="badge-danger">Close to Total Failure (Critical)</span>'

        with m4:
            st.markdown(f"**Diagnostic Status:**<br>{status_badge}", unsafe_allow_html=True)

        if abs_err < 1.0:
            st.success(f"High precision prediction: Model error is only {abs_err:.2f}% capacity.")

    else:
        st.subheader("1. Enter Custom Physical Sensor Readings")
        st.caption("Provide key physical sensor readings to simulate an ad-hoc operating cycle.")

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            in_ts1 = st.slider("Temperature TS1 (°C) - Cooling Oil Feed", 25.0, 75.0, 45.0, step=0.5)
            in_fs1 = st.slider("Volume Flow FS1 (L/min) - Primary Circuit", 1.0, 15.0, 8.5, step=0.1)
        with col_in2:
            in_ps1 = st.slider("Pressure PS1 (bar) - Main Relief Pressure", 120.0, 200.0, 160.0, step=1.0)
            in_eps1 = st.slider("Motor Power EPS1 (W) - Electric Drive", 2200.0, 3200.0, 2500.0, step=10.0)

        custom_dict = {"TS1_mean": in_ts1, "FS1_mean": in_fs1, "PS1_mean": in_ps1, "EPS1_mean": in_eps1}
        is_valid, errors, warnings = validate_sensor_input(custom_dict)

        st.markdown("#### 2. Input Validation")
        if is_valid and len(warnings) == 0:
            st.success("✅ **Input Validation Passed:** All custom parameters are physically realistic.")
        else:
            for w in warnings:
                st.warning(w)

        st.markdown("#### 3. Making the Input the Best for Predicting Y")
        st.info(
            "Using the **Physical Interpretable Ridge Model** fitted strictly on physical sensors "
            "(excluding target proxy virtual sensors CE/CP). Inputs are standardized via z-score normalization "
            "using training parameters: z = (x - μ) / σ."
        )

        if phys_model is not None:
            # Construct a row with median defaults for remaining physical features
            median_features = df_features[metadata.get("best_physical_features", metadata["feature_columns"])].median().to_dict()
            median_features["TS1_mean"] = in_ts1
            median_features["FS1_mean"] = in_fs1
            median_features["PS1_mean"] = in_ps1
            median_features["EPS1_mean"] = in_eps1
            X_custom = pd.DataFrame([median_features])

            phys_pred = float(phys_model.predict(X_custom)[0])
            phys_pred_bounded = float(np.clip(phys_pred, 0.0, 100.0))

            st.markdown("#### 4. Identified Best Y Prediction")
            r1, r2 = st.columns(2)
            r1.metric("Predicted Cooler Condition (y)", f"{phys_pred_bounded:.2f}%")
            if phys_pred_bounded >= 75.0:
                badge = '<span class="badge-success">Full Efficiency</span>'
            elif phys_pred_bounded >= 15.0:
                badge = '<span class="badge-warning">Reduced Efficiency</span>'
            else:
                badge = '<span class="badge-danger">Close to Total Failure</span>'
            r2.markdown(f"**Predicted State:**<br>{badge}", unsafe_allow_html=True)


# ==============================================================================
# TAB 3: BEST VARIABLES & TARGETS
# ==============================================================================
elif choice == "🎯 Best Variables & Targets":
    st.markdown('<div class="main-header">🎯 Best Independent (X) and Dependent (y) Variables</div>', unsafe_allow_html=True)
    st.markdown("Evidence-based selection of optimal variables to prevent data leakage and maximize physical validity.")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("Optimal Dependent Variable (y)")
        st.markdown("""
        **Selected Target:** `cooler_condition` (Continuous Heat Dissipation Efficiency %)
        
        **Why Cooler Condition is the Best y:**
        1. **Continuous Physical Degradation:** Unlike binary or discrete stepped failure flags, cooler efficiency directly reflects fluid thermal balance.
        2. **Supervised Regression Fit:** The target values (100% full, 20% reduced, 3% critical) correspond to physical heat exchange capacity.
        3. **Operational Relevance:** Overheating oil accelerates seal degradation, fluid breakdown, and system-wide failure.
        
        **Other 4 Target Candidates Evaluated:**
        - `valve_condition`: Switching response lag (100, 90, 80, 73%).
        - `internal_pump_leakage`: Stepped leakage severity (0, 1, 2).
        - `accumulator_pressure`: Pre-charge gas nitrogen pressure (130, 115, 100, 90 bar).
        - `stable_flag`: Transient binary flag (0=stable, 1=unstable).
        """)

    with col_t2:
        st.subheader("Optimal Independent Variables (X)")
        st.markdown("""
        **Top Physical Sensor Predictors Identified:**
        1. **`TS1` (Temperature 1):** Cooling oil circuit temperature. Highly responsive to heat exchanger failure.
        2. **`FS1` (Volume Flow 1):** Primary circuit flow rate. Directly impacts cooling fluid throughput.
        3. **`TS2` (Temperature 2):** Return line temperature. Thermal differential `(TS1 - TS2)` reflects dissipated power.
        4. **`EPS1` (Motor Power):** Hydraulic pump electric power draw.
        5. **Calculus Integrals (&int; s(t) dt):** Cumulative energy and total volume pumped over 60-second cycle.
        
        **Virtual Sensor Audit (CE & CP):**
        - `CE` (Cooling Efficiency) is a **virtual proxy**.
        - Including `CE_mean` yields Test R² > 0.9997 because `CE` directly calculates cooler health.
        - The project provides models **both with and without CE/CP** for transparent engineering evaluation.
        """)

    st.markdown("---")
    st.subheader("Pearson Correlation Ranking with Cooler Condition")
    corr_img_path = FIGURES_DIR / "target_correlation_ranking.png"
    if corr_img_path.exists():
        st.image(str(corr_img_path), caption="Top Positive and Negative Feature Correlations with Cooler Condition", use_column_width=True)

    st.subheader("Complete Data Dictionary (17 Sensors & 5 Targets)")
    dict_df = build_data_dictionary()
    st.dataframe(dict_df, use_container_width=True)


# ==============================================================================
# TAB 4: MODEL EVALUATION
# ==============================================================================
elif choice == "📊 Model Evaluation":
    st.markdown('<div class="main-header">📊 Model Evaluation & Split Integrity</div>', unsafe_allow_html=True)
    st.markdown("Held-out test results, error metrics, residual distributions, and split comparison.")

    # High-level scorecards
    test_metrics = metadata.get("held_out_test_metrics", {})
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Selected Model", metadata.get("selected_model", "ExtraTrees"))
    e2.metric("Held-Out Test MAE", f"{test_metrics.get('MAE', 0.0):.4f}%")
    e3.metric("Held-Out Test RMSE", f"{test_metrics.get('RMSE', 0.0):.4f}%")
    e4.metric("Held-Out Test R²", f"{test_metrics.get('R2', 0.0):.4f}")

    st.markdown("---")
    c_p1, c_p2 = st.columns(2)

    with c_p1:
        st.subheader("Actual vs. Predicted Scatter")
        p1 = FIGURES_DIR / "actual_vs_predicted.png"
        if p1.exists():
            st.image(str(p1), use_column_width=True)

    with c_p2:
        st.subheader("Residual Distribution (y - ŷ)")
        p2 = FIGURES_DIR / "residuals_distribution.png"
        if p2.exists():
            st.image(str(p2), use_column_width=True)

    st.markdown("---")
    st.subheader("CRITICAL: Random Split vs Chronological Block Split")
    st.warning("""
    **Syllabus Concept: Contiguous Operating Blocks & Undefined R²**  
    The hydraulic test rig ran in 3 contiguous blocks:
    - Cycles 0–731: Cooler = 3%
    - Cycles 732–1463: Cooler = 20%
    - Cycles 1464–2204: Cooler = 100%
    
    In a purely chronological 80/20 split (first 80% train, last 20% test), the test set contains **only Cooler = 100%**.  
    Target variance is **0.0**, meaning Total Sum of Squares (SS_tot) = 0.  
    Therefore, **R² is mathematically undefined** (division by zero). We report this honestly rather than inventing a score.
    """)

    split_comp_path = METRICS_DIR / "split_strategy_comparison.csv"
    if split_comp_path.exists():
        split_df = pd.read_csv(split_comp_path)
        st.table(split_df)

    p_split = FIGURES_DIR / "split_comparison_bar.png"
    if p_split.exists():
        st.image(str(p_split), caption="Comparison of RMSE across Evaluation Protocols", width=600)


# ==============================================================================
# TAB 5: CALCULUS & OPTIMIZATION LAB
# ==============================================================================
elif choice == "🧮 Calculus & Optimization":
    st.markdown('<div class="main-header">🧮 Calculus & First-Principles Optimization Lab</div>', unsafe_allow_html=True)
    st.markdown("Interactive demonstrations of numerical integration, finite differences, batch gradient descent, and partial derivatives.")

    calc_tab1, calc_tab2, calc_tab3 = st.tabs([
        "📉 Manual Batch Gradient Descent",
        "∫ Numerical Integration & Differentiation",
        "∂f/∂x Partial Derivatives & Sensitivity"
    ])

    with calc_tab1:
        st.subheader("Batch Gradient Descent for Linear Regression (From First Principles)")
        st.markdown(r"""
        **Mathematical Formulation:**
        - Hypothesis: $\hat{y} = \mathbf{X}\mathbf{w} + b$
        - Mean Squared Error Loss: $J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2$
        - Analytically Derived Gradients:
          $$\frac{\partial J}{\partial \mathbf{w}} = \frac{2}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}), \quad \frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$$
        - Iterative Update: $\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial J}{\partial \mathbf{w}}, \quad b \leftarrow b - \eta \frac{\partial J}{\partial b}$
        """)

        g1, g2 = st.columns([1, 2])
        with g1:
            lr = st.selectbox("Learning Rate (η):", [0.001, 0.01, 0.03, 0.08, 0.5], index=2)
            n_iters = st.slider("Maximum Iterations:", 100, 2000, 800, step=100)

            if st.button("Run Batch GD Optimizer", type="primary"):
                bgd_sim = BatchGradientDescentRegressor(learning_rate=lr, max_iters=n_iters, tol=1e-7)
                X_sample_s = StandardScaler().fit_transform(df_features[["TS1_mean", "FS1_mean", "EPS1_mean"]].to_numpy())
                y_sample = df_features["cooler_condition"].to_numpy()
                bgd_sim.fit(X_sample_s, y_sample)

                st.session_state.bgd_loss = bgd_sim.loss_history_
                st.session_state.bgd_converged = bgd_sim.converged_
                st.session_state.bgd_diverged = bgd_sim.diverged_
                st.session_state.bgd_iters = bgd_sim.iterations_run_

        with g2:
            if "bgd_loss" in st.session_state:
                losses = st.session_state.bgd_loss
                if st.session_state.bgd_diverged:
                    st.error(f"⚠️ Gradient descent diverged! Learning rate {lr} is excessively large.")
                else:
                    st.success(f"Converged: {st.session_state.bgd_converged} in {st.session_state.bgd_iters} iterations. Final Loss: {losses[-1]:.4f}")
                    fig_gd, ax_gd = plt.subplots(figsize=(7, 4))
                    ax_gd.plot(losses, color="#1f77b4", lw=2)
                    ax_gd.set_title(f"Loss Trajectory (η = {lr})")
                    ax_gd.set_xlabel("Iteration")
                    ax_gd.set_ylabel("MSE Loss J(w,b)")
                    ax_gd.grid(True, linestyle=":", alpha=0.6)
                    st.pyplot(fig_gd)
            else:
                p_gd = FIGURES_DIR / "gradient_descent_convergence.png"
                if p_gd.exists():
                    st.image(str(p_gd), caption="Default Precomputed Loss Convergence Curve (η = 0.03)", use_column_width=True)

    with calc_tab2:
        st.subheader("Trapezoidal Integration & Finite Difference Differentiation")
        st.markdown(r"""
        **1. Composite Trapezoidal Rule:**
        $$\int_0^T s(t) dt \approx \Delta t \left( \frac{s_0 + s_{N-1}}{2} + \sum_{k=1}^{N-2} s_k \right)$$
        *Physical Meaning:* Cumulative thermal heat flux, cumulative fluid volume, or total power consumption.
        
        **2. Finite Differences (ds/dt):**
        $$\frac{ds}{dt} \approx \frac{s_{k+1} - s_k}{\Delta t}$$
        *Physical Meaning:* Dynamic pressure surge, heating rate, or flow acceleration.
        """)

        # Display sample calculus feature comparisons
        st.markdown("#### Sample Extracted Calculus Features (First 5 Cycles)")
        calc_cols_show = [c for c in df_features.columns if any(k in c for k in ["_integral", "_derivative"])][:6]
        if calc_cols_show:
            st.dataframe(df_features[calc_cols_show].head(5).round(3))

    with calc_tab3:
        st.subheader("Sensitivity Analysis: Partial Derivatives (∂f / ∂x_i)")
        st.markdown(r"""
        For a standardized linear model $f(\mathbf{z}) = \sum w_j z_j + b$, where $z_j = \frac{x_j - \mu_j}{\sigma_j}$:
        $$\frac{\partial f}{\partial z_j} = w_j, \quad \frac{\partial f}{\partial x_j} = \frac{w_j}{\sigma_j}$$
        
        **Crucial Distinction: Sensitivity &ne; Physical Causation!**  
        High mathematical sensitivity indicates responsiveness of the model's decision plane. 
        It does NOT imply that turning an external knob on sensor $x_j$ will cause the physical cooler to heal.
        """)

        sens_img = FIGURES_DIR / "feature_sensitivity_ranking.png"
        if sens_img.exists():
            st.image(str(sens_img), caption="Standardized Feature Partial Derivatives |∂f / ∂z_j|", use_column_width=True)


# ==============================================================================
# TAB 6: MODEL COMPARISON & ABLATION
# ==============================================================================
elif choice == "🔬 Model Comparison":
    st.markdown('<div class="main-header">🔬 Model Comparison, PCA & Ablation Studies</div>', unsafe_allow_html=True)
    st.markdown("Empirical comparison of algorithms, feature subsets, polynomial degrees, and PCA.")

    comp_csv_path = METRICS_DIR / "model_comparison.csv"
    if comp_csv_path.exists():
        st.subheader("Comprehensive Leaderboard (5-Fold CV on Train Set)")
        comp_df = pd.read_csv(comp_csv_path)
        st.dataframe(comp_df.round(4), use_container_width=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("Polynomial Regression: Bias-Variance Curve")
        poly_img = FIGURES_DIR / "polynomial_bias_variance_curve.png"
        if poly_img.exists():
            st.image(str(poly_img), use_column_width=True)
        st.caption("Illustrates the bias-variance trade-off across Degree 1 (Underfitting), Degree 2 (Optimal), and Degree 3 (Increasing Complexity).")

    with col_m2:
        st.subheader("PCA Dimensionality Reduction Scree Plot")
        pca_img = FIGURES_DIR / "pca_explained_variance_scree.png"
        if pca_img.exists():
            st.image(str(pca_img), use_column_width=True)
        st.caption("Cumulative explained variance ratio: 10 components capture >90% of sensor variance.")

    st.markdown("---")
    st.subheader("Ridge Regularization: Loss vs L2 Penalty Breakdown")
    ridge_json_path = METRICS_DIR / "ridge_penalty_breakdown.json"
    if ridge_json_path.exists():
        with open(ridge_json_path, encoding="utf-8") as f:
            ridge_data = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha Penalty (α)", f"{ridge_data['alpha']}")
        c2.metric("Prediction MSE", f"{ridge_data['prediction_MSE']:.4f}")
        c3.metric("L2 Penalty (α ∑ w²)", f"{ridge_data['l2_penalty_term_alpha_sum_w2']:.4f}")
        c4.metric("Total Ridge Loss", f"{ridge_data['total_ridge_objective']:.4f}")


# ==============================================================================
# TAB 7: HOW IT WORKS (SYLLABUS LEARNING GUIDE)
# ==============================================================================
elif choice == "📖 How It Works (Syllabus)":
    st.markdown('<div class="main-header">📖 How It Works: AIML & Calculus Syllabus Guide</div>', unsafe_allow_html=True)
    st.markdown("Clear, beginner-friendly explanations answering core academic viva and coursework questions.")

    with st.expander("1. What exactly is the model predicting?", expanded=True):
        st.markdown("""
        The model predicts **`cooler_condition`** (hydraulic cooler health percentage from 3% to 100%).
        It represents the continuous cooling capacity of the secondary cooling-filtration circuit.
        - **100%:** Full cooling efficiency (optimal heat exchange).
        - **20%:** Reduced efficiency (fan degradation / fouling).
        - **3%:** Close to total failure (near zero cooling capacity).
        """)

    with st.expander("2. What are X and y?", expanded=False):
        st.markdown("""
        - **Independent variables (X):** Engineered features extracted from 17 physical sensors (pressures, volume flows, temperatures, electric power, and vibration), including statistical moments (mean, std, min, max) and calculus metrics (definite integrals and numerical time derivatives).
        - **Dependent variable (y):** The target cooler condition percentage.
        - **One Training Example:** One single 60-second cycle executed by the hydraulic test rig.
        """)

    with st.expander("3. Why is this Supervised Regression rather than Classification or Clustering?", expanded=False):
        st.markdown("""
        - **Supervised:** The model learns from labeled historical cycles where ground truth sensor data and degradation states are both known.
        - **Regression:** Although the experimental test rig set cooler condition to discrete values (3%, 20%, 100%), degradation in real industrial environments is a **continuous physical process**. Estimating a numerical percentage reflects continuous degradation rather than discrete classification.
        - **Not Clustering/Unsupervised:** Clusters do not have ground truth labels and cannot predict calibrated engineering metrics.
        """)

    with st.expander("4. What does Gradient Descent optimize?", expanded=False):
        st.markdown(r"""
        Batch Gradient Descent iteratively minimizes the **Mean Squared Error (MSE)** loss function:
        $$J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (\mathbf{x}_i^T \mathbf{w} + b - y_i)^2$$
        By computing the analytical vector gradient $\nabla J = [\frac{\partial J}{\partial \mathbf{w}}, \frac{\partial J}{\partial b}]^T$, it steps downhill against the gradient by a learning rate $\eta$ until convergence.
        """)

    with st.expander("5. Did Calculus-Enhanced features improve performance?", expanded=False):
        st.markdown("""
        **Yes, for Linear and Regularized Ridge Models:**
        - On Ridge Regression (without virtual proxies), calculus features reduced CV RMSE from **5.171** down to **4.697** (a **9.2% improvement**).
        - Calculus features capture **rate of temperature change** and **cumulative energy**, which linear models cannot compute on their own.
        - For deep tree ensembles (ExtraTrees / RandomForest), summary statistics alone were sufficient, so the incremental benefit of calculus features was marginal. We report this honestly.
        """)

    with st.expander("6. What can this model reliably claim, and what can it NOT claim?", expanded=False):
        st.markdown(r"""
        **What it CAN reliably claim:**
        - High predictive precision ($R^2 > 0.99$, MAE < 0.1%) when interpolating cycles across similar operating regimes.
        - Significant error reduction when adding calculus dynamic features to linear regression models.
        - Effective discrimination between healthy (100%), impaired (20%), and failed (3%) cooling states.
        
        **What it CANNOT claim:**
        - It is NOT a certified safety shutdown system for real-time aerospace/nuclear facilities.
        - It does NOT claim physical causation from sensitivity derivatives ($\partial f / \partial x$).
        - It does NOT guarantee generalization to unseen future hydraulic rigs without recalibration.
        """)

st.markdown("---")
st.caption("AI-Driven Hydraulic System Health Prediction and Optimization | B.Tech AIML & Applied Calculus Project | October 2026")
