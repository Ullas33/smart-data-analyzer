import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Data Analyzer", page_icon="🧠", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #e8c98b !important;
    font-family: 'Outfit', sans-serif !important;
    color: #1a1a2e !important;
}
[data-testid="stSidebar"] {
    background-color: #1a1a2e !important;
    border-right: 3px solid #e8c547 !important;
}
[data-testid="stSidebar"] * { color: #f2d18f !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #e8c547 !important; font-size: 0.8rem !important; }

h1 { font-family: 'Outfit', sans-serif !important; font-weight: 800 !important;
     font-size: 2.2rem !important; color: #1a1a2e !important; }
h2, h3 { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; color: #1a1a2e !important; }

[data-testid="metric-container"] {
    background: #1a1a2e !important;
    border-radius: 10px !important;
    padding: 14px !important;
    border-bottom: 3px solid #e8c547 !important;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8c547 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1.3rem !important; }

.section-box {
    background: #ffffff;
    border: 1px solid #ddd8cc;
    border-left: 5px solid #e8c547;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.insight-box {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #cbd5e1;
    line-height: 1.7;
}
.insight-box b { color: #e8c547; }
.step-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.stButton > button {
    background: #e8c547 !important;
    color: #1a1a2e !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 24px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def statistical_summary(df):
    numeric = df.select_dtypes(include=np.number)
    summary = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        mode_val = stats.mode(s, keepdims=True).mode[0]
        summary.append({
            "Column":   col,
            "Mean":     round(s.mean(), 3),
            "Median":   round(s.median(), 3),
            "Mode":     round(mode_val, 3),
            "Std Dev":  round(s.std(), 3),
            "Variance": round(s.var(), 3),
            "Min":      round(s.min(), 3),
            "Max":      round(s.max(), 3),
            "Missing":  int(s.isna().sum()),
        })
    return pd.DataFrame(summary)


def probability_in_range(series, low, high):
    count_in_range = ((series >= low) & (series <= high)).sum()
    prob = count_in_range / len(series)
    return int(count_in_range), round(prob, 4)


def dot_product_similarity(row_a, row_b):
    dot = np.dot(row_a, row_b)
    norm_a = np.linalg.norm(row_a)
    norm_b = np.linalg.norm(row_b)
    cosine = dot / (norm_a * norm_b) if (norm_a * norm_b) != 0 else 0
    return round(dot, 4), round(cosine, 4)


def make_bar_chart(series, col_name):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#f4f1eb")
    ax.set_facecolor("#f4f1eb")
    n_bins = min(30, max(10, int(np.sqrt(len(series)))))
    ax.hist(series.dropna(), bins=n_bins, color="#1a1a2e", edgecolor="#f4f1eb", linewidth=0.5)
    ax.axvline(series.mean(),   color="#e8c547", linestyle="--", linewidth=2, label=f"Mean {series.mean():.2f}")
    ax.axvline(series.median(), color="#f97316", linestyle="-",  linewidth=2, label=f"Median {series.median():.2f}")
    ax.set_title(f"Distribution: {col_name}", fontsize=11, color="#1a1a2e", fontweight="bold")
    ax.set_xlabel(col_name, color="#64748b", fontsize=9)
    ax.set_ylabel("Frequency", color="#64748b", fontsize=9)
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ddd8cc")
    ax.legend(fontsize=8, facecolor="#f4f1eb", edgecolor="#ddd8cc")
    ax.grid(axis="y", color="#ddd8cc", linewidth=0.6)
    fig.tight_layout()
    return fig


def make_row_comparison_chart(row_a, row_b, columns, label_a, label_b):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#f4f1eb")
    ax.set_facecolor("#f4f1eb")
    x = np.arange(len(columns))
    ax.bar(x - 0.2, row_a, width=0.4, label=f"Row {label_a}", color="#1a1a2e")
    ax.bar(x + 0.2, row_b, width=0.4, label=f"Row {label_b}", color="#e8c547")
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=35, ha="right", fontsize=8, color="#1a1a2e")
    ax.set_ylabel("Value", color="#64748b", fontsize=9)
    ax.set_title("Row-by-Row Value Comparison", fontsize=11, color="#1a1a2e", fontweight="bold")
    ax.legend(fontsize=9, facecolor="#f4f1eb", edgecolor="#ddd8cc")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ddd8cc")
    ax.grid(axis="y", color="#ddd8cc", linewidth=0.6)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Smart Data Analyzer")
    st.markdown("---")
    uploaded = st.file_uploader("📥 Upload CSV", type=["csv"])
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.75rem;color:#94a3b8;font-family:IBM Plex Mono,monospace'>"
        "Weekly Project · Statistics + Probability + Vector Similarity"
        "</span>", unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 Smart Data Analyzer")
st.markdown(
    "<p style='color:#64748b;font-family:IBM Plex Mono,monospace;font-size:0.8rem;margin-top:-12px'>"
    "Statistics · Probability · Vector Similarity · Insights</p>",
    unsafe_allow_html=True
)

if uploaded is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(uploaded)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) == 0:
    st.error("No numeric columns found. Please upload a dataset with numeric data.")
    st.stop()

st.markdown(
    f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#64748b'>"
    f"✅ Loaded: <b>{df.shape[0]} rows × {df.shape[1]} columns</b> · "
    f"{len(numeric_cols)} numeric columns</span>",
    unsafe_allow_html=True
)

with st.expander("📋 Preview Dataset (first 10 rows)"):
    st.dataframe(df.head(10), width=1400)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – STATISTICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📐 1. Statistical Summary")
summary_df = statistical_summary(df)
st.dataframe(summary_df, width=1400)

# Metric highlight for selected column
st.markdown("<div class='step-label'>Quick Stats – Select a column</div>", unsafe_allow_html=True)
sel_col = st.selectbox("Column", numeric_cols, key="stat_col")
s = df[sel_col].dropna()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Mean",     round(s.mean(), 3))
c2.metric("Median",   round(s.median(), 3))
c3.metric("Std Dev",  round(s.std(), 3))
c4.metric("Variance", round(s.var(), 3))
c5.metric("Min",      round(s.min(), 3))
c6.metric("Max",      round(s.max(), 3))

# Distribution chart
st.pyplot(make_bar_chart(s, sel_col))

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – PROBABILITY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎲 2. Probability Insights")

prob_col = st.selectbox("Select column for probability analysis", numeric_cols, key="prob_col")
ps = df[prob_col].dropna()

col_min = float(ps.min())
col_max = float(ps.max())

st.markdown("<div class='step-label'>Define a value range</div>", unsafe_allow_html=True)
low, high = st.slider(
    "Range",
    min_value=col_min, max_value=col_max,
    value=(col_min, col_min + (col_max - col_min) / 2),
    key="prob_slider"
)

count_in, prob = probability_in_range(ps, low, high)
total = len(ps)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Values in Range", count_in)
col_b.metric("Total Values",    total)
col_c.metric("Probability P(x)", prob)

# Highlight range on histogram
fig_p, ax_p = plt.subplots(figsize=(8, 3.5))
fig_p.patch.set_facecolor("#f4f1eb")
ax_p.set_facecolor("#f4f1eb")
n_bins = min(30, max(10, int(np.sqrt(len(ps)))))
ax_p.hist(ps, bins=n_bins, color="#1a1a2e", edgecolor="#f4f1eb", linewidth=0.5, label="All values")
in_range = ps[(ps >= low) & (ps <= high)]
ax_p.hist(in_range, bins=n_bins, color="#e8c547", edgecolor="#f4f1eb", linewidth=0.5, label=f"In range [{low:.2f}, {high:.2f}]")
ax_p.set_title(f"Probability Range Highlight: {prob_col}", fontsize=11, color="#1a1a2e", fontweight="bold")
ax_p.set_xlabel(prob_col, color="#64748b", fontsize=9)
ax_p.set_ylabel("Frequency", color="#64748b", fontsize=9)
ax_p.tick_params(colors="#64748b")
for spine in ax_p.spines.values(): spine.set_edgecolor("#ddd8cc")
ax_p.legend(fontsize=8, facecolor="#f4f1eb", edgecolor="#ddd8cc")
ax_p.grid(axis="y", color="#ddd8cc", linewidth=0.6)
fig_p.tight_layout()
st.pyplot(fig_p)

# Normal distribution fit
st.markdown("<div class='step-label'>Normal Distribution Fit</div>", unsafe_allow_html=True)
mu, sigma = ps.mean(), ps.std()
x_range = np.linspace(col_min, col_max, 300)
pdf_vals = stats.norm.pdf(x_range, mu, sigma)

fig_n, ax_n = plt.subplots(figsize=(8, 3))
fig_n.patch.set_facecolor("#f4f1eb")
ax_n.set_facecolor("#f4f1eb")
ax_n.hist(ps, bins=n_bins, density=True, color="#1a1a2e", edgecolor="#f4f1eb", linewidth=0.5, alpha=0.7, label="Actual")
ax_n.plot(x_range, pdf_vals, color="#e8c547", linewidth=2.5, label=f"Normal fit (μ={mu:.2f}, σ={sigma:.2f})")
ax_n.set_title("Normal Distribution Fit", fontsize=11, color="#1a1a2e", fontweight="bold")
ax_n.set_xlabel(prob_col, color="#64748b", fontsize=9)
ax_n.set_ylabel("Density", color="#64748b", fontsize=9)
ax_n.tick_params(colors="#64748b")
for spine in ax_n.spines.values(): spine.set_edgecolor("#ddd8cc")
ax_n.legend(fontsize=8, facecolor="#f4f1eb", edgecolor="#ddd8cc")
ax_n.grid(axis="y", color="#ddd8cc", linewidth=0.6)
fig_n.tight_layout()
st.pyplot(fig_n)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – VECTOR SIMILARITY (DOT PRODUCT)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔢 3. Row Similarity – Dot Product")
st.write("Select two rows. Their numeric values form vectors and we compute similarity.")

numeric_df = df[numeric_cols].dropna().reset_index(drop=True)

row_a_idx = st.number_input("Row A (index)", min_value=0, max_value=len(numeric_df)-1, value=0, step=1)
row_b_idx = st.number_input("Row B (index)", min_value=0, max_value=len(numeric_df)-1, value=1, step=1)

row_a = numeric_df.iloc[row_a_idx].values.astype(float)
row_b = numeric_df.iloc[row_b_idx].values.astype(float)

dot, cosine = dot_product_similarity(row_a, row_b)

col_d1, col_d2, col_d3 = st.columns(3)
col_d1.metric("Dot Product",        dot)
col_d2.metric("Cosine Similarity",  cosine)
col_d3.metric("Columns Used",       len(numeric_cols))

# Side by side table
comparison_df = pd.DataFrame({
    "Column":  numeric_cols,
    f"Row {row_a_idx}": row_a,
    f"Row {row_b_idx}": row_b,
    "Difference": np.round(row_a - row_b, 4),
})
st.dataframe(comparison_df, width=1400)

# Visual comparison chart
st.pyplot(make_row_comparison_chart(row_a, row_b, numeric_cols, row_a_idx, row_b_idx))

# Radar chart (if ≤ 10 columns)
if len(numeric_cols) <= 10:
    st.markdown("<div class='step-label'>Radar / Spider Chart</div>", unsafe_allow_html=True)
    angles = np.linspace(0, 2 * np.pi, len(numeric_cols), endpoint=False).tolist()
    angles += angles[:1]

    norm_a = (row_a - row_a.min()) / ((row_a.max() - row_a.min()) + 1e-9)
    norm_b = (row_b - row_b.min()) / ((row_b.max() - row_b.min()) + 1e-9)
    vals_a = norm_a.tolist() + [norm_a[0]]
    vals_b = norm_b.tolist() + [norm_b[0]]

    fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig_r.patch.set_facecolor("#f4f1eb")
    ax_r.set_facecolor("#f4f1eb")
    ax_r.plot(angles, vals_a, color="#1a1a2e", linewidth=2, label=f"Row {row_a_idx}")
    ax_r.fill(angles, vals_a, color="#1a1a2e", alpha=0.25)
    ax_r.plot(angles, vals_b, color="#e8c547", linewidth=2, label=f"Row {row_b_idx}")
    ax_r.fill(angles, vals_b, color="#e8c547", alpha=0.25)
    ax_r.set_thetagrids(np.degrees(angles[:-1]), numeric_cols, fontsize=8, color="#1a1a2e")
    ax_r.set_title("Normalised Radar Comparison", fontsize=11, color="#1a1a2e", fontweight="bold", pad=15)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8, facecolor="#f4f1eb", edgecolor="#ddd8cc")
    ax_r.grid(color="#ddd8cc")
    st.pyplot(fig_r)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 💡 4. Auto-Generated Insights")

insights = []

# Skewness per column
for col in numeric_cols:
    sk = df[col].dropna().skew()
    if abs(sk) > 1:
        direction = "right" if sk > 0 else "left"
        insights.append(f"<b>{col}</b> is heavily {direction}-skewed (skew = {sk:.2f}). Consider log-transformation before modelling.")
    elif abs(sk) > 0.5:
        direction = "right" if sk > 0 else "left"
        insights.append(f"<b>{col}</b> shows moderate {direction}-skewness (skew = {sk:.2f}).")

# Missing values
total_missing = df.isnull().sum().sum()
if total_missing > 0:
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    insights.append(f"<b>{total_missing} missing value(s)</b> found across columns: {', '.join(cols_with_missing)}. Imputation or removal recommended.")
else:
    insights.append("<b>No missing values</b> found in the dataset. ✓")

# Probability range insight
insights.append(
    f"<b>Probability insight:</b> {count_in} out of {total} values in <b>{prob_col}</b> fall "
    f"within [{low:.2f}, {high:.2f}] — P(x) = <b>{prob}</b>."
)

# Similarity insight
if cosine > 0.95:
    insights.append(f"<b>Rows {row_a_idx} & {row_b_idx}</b> are highly similar (cosine = {cosine}). They may represent near-duplicate or closely related records.")
elif cosine > 0.7:
    insights.append(f"<b>Rows {row_a_idx} & {row_b_idx}</b> show moderate similarity (cosine = {cosine}).")
else:
    insights.append(f"<b>Rows {row_a_idx} & {row_b_idx}</b> are quite different (cosine = {cosine}). These records have distinct numeric profiles.")

# High variance columns
high_var_cols = [col for col in numeric_cols if df[col].std() > df[col].mean() * 0.5 and df[col].mean() != 0]
if high_var_cols:
    insights.append(f"<b>High variability</b> detected in: {', '.join(high_var_cols)}. These columns may contain outliers or represent diverse sub-groups.")

# Correlation (if multiple numeric cols)
if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr()
    high_corr = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.8:
                high_corr.append(f"{numeric_cols[i]} & {numeric_cols[j]} (r={val:.2f})")
    if high_corr:
        insights.append(f"<b>Strong correlation found:</b> {'; '.join(high_corr)}. These features may be redundant.")

for insight in insights:
    st.markdown(f"<div class='insight-box'>💡 {insight}</div>", unsafe_allow_html=True)