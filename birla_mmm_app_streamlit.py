
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Birla Opus - Guided MMM Simulator", layout="wide")

st.title("Birla Opus - Guided MMM Simulator (Regression + ROI)")
st.caption("Upload -> Auto bucket detection -> Guided defaults -> Bucket tests -> Final model -> ROI simulator -> Downloads")

# -----------------------
# Helpers
# -----------------------
MEDIA_KEYS = ["tv", "digital", "youtube", "video", "ooh", "print", "impressions", "reach", "grps", "sites"]
TRADE_KEYS = ["dealer", "in-bill", "inbill", "cn", "contractor", "painter", "loyalty", "token", "coverage"]
EXTERNAL_KEYS = ["ofr", "pdo", "pco", "sov", "inflation", "rain", "festival", "housing", "price"]
DEP_KEYS = ["secondary", "volume", "sales", "kl", "litre"]

def classify_columns(df):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    buckets = {"dependent_candidates": [], "media": [], "trade": [], "external": [], "other_numeric": []}
    for c in num_cols:
        lc = c.lower()
        if any(k in lc for k in DEP_KEYS):
            buckets["dependent_candidates"].append(c)
            continue
        if any(k in lc for k in MEDIA_KEYS):
            buckets["media"].append(c); continue
        if any(k in lc for k in TRADE_KEYS):
            buckets["trade"].append(c); continue
        if any(k in lc for k in EXTERNAL_KEYS):
            buckets["external"].append(c); continue
        buckets["other_numeric"].append(c)
    return buckets

def make_lag(df, group_keys, value_cols, lag=1):
    out = df.copy()
    sort_keys = [c for c in group_keys if c in out.columns]
    if "Month" in out.columns:
        sort_keys = sort_keys + ["Month"]
    if sort_keys:
        out = out.sort_values(sort_keys)
    for v in value_cols:
        if v in out.columns:
            out[v + f"_Lag{lag}"] = out.groupby(group_keys, dropna=False)[v].shift(lag) if group_keys else out[v].shift(lag)
    return out

def run_ols(df, y_col, x_cols, add_const=True):
    work = df[[y_col] + x_cols].dropna()
    if work.empty:
        return None, None, None, None
    X = work[x_cols]
    if add_const:
        X = sm.add_constant(X, has_constant="add")
    y = work[y_col]
    model = sm.OLS(y, X).fit()
    work = work.copy()
    work["Predicted"] = model.predict(X)
    coef = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "StdErr": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values
    })
    fit = pd.DataFrame({"Metric": ["N","R_squared","Adj_R_squared","AIC","BIC"],
                        "Value": [int(model.nobs), model.rsquared, model.rsquared_adj, model.aic, model.bic]})
    return model, work, coef, fit

def executive_summary(coef_df, fit_df, y_name="Sales"):
    if coef_df is None or fit_df is None:
        return "No model results available."
    adjr = float(fit_df.loc[fit_df["Metric"]=="Adj_R_squared","Value"].values[0])
    sig = coef_df[(coef_df["Variable"]!="const") & (coef_df["p_value"]<=0.10)].copy()
    sig_sorted = sig.sort_values("Coefficient", ascending=False)

    bullets = []
    bullets.append(f"Model explains {adjr:.0%} of variation in {y_name} (Adj. R^2).")
    if not sig_sorted.empty:
        top_pos = sig_sorted[sig_sorted["Coefficient"]>0]["Variable"].tolist()[:3]
        top_neg = sig_sorted[sig_sorted["Coefficient"]<0]["Variable"].tolist()[:2]
        if top_pos:
            bullets.append("Key positive drivers: " + ", ".join(top_pos) + ".")
        if top_neg:
            bullets.append("Headwinds (negative): " + ", ".join(top_neg) + ".")
    else:
        bullets.append("No strongly significant drivers at p<=0.10. Consider fewer variables or more months.")
    return "\\n".join([f"- {b}" for b in bullets])

def download_excel(dfs: dict, name="MMM_Results.xlsx"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, index=False, sheet_name=sheet[:31])
    output.seek(0)
    st.download_button("Download Results (Excel)", data=output.getvalue(),
                       file_name=name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------
# Upload
# -----------------------
st.sidebar.header("1) Upload Data")
f = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
if f is None:
    st.info("Upload your 16x10 panel. Columns like 'Secondary Vol. (KL)', 'TV Impressions (Lacs)', 'Dealer Input CN', etc.")
    st.stop()

df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)

# Month parse (best effort)
if "Month" in df.columns:
    try:
        df["Month"] = pd.to_datetime(df["Month"])
    except Exception:
        pass

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# -----------------------
# Auto bucket detection & guidance
# -----------------------
buckets = classify_columns(df)

st.markdown("### Auto-detected buckets (from your column names)")
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Dependent candidates", len(buckets["dependent_candidates"]))
with col2: st.metric("Media drivers", len(buckets["media"]))
with col3: st.metric("Trade drivers", len(buckets["trade"]))
with col4: st.metric("External drivers", len(buckets["external"]))

with st.expander("See detected variables"):
    st.write("Dependent candidates:", buckets["dependent_candidates"] or "—")
    st.write("Media:", buckets["media"] or "—")
    st.write("Trade:", buckets["trade"] or "—")
    st.write("External:", buckets["external"] or "—")

# Defaults
y_default = "Secondary Vol. (KL)" if "Secondary Vol. (KL)" in buckets["dependent_candidates"] else (buckets["dependent_candidates"][0] if buckets["dependent_candidates"] else None)
media_default = buckets["media"]
trade_default = buckets["trade"]
external_default = buckets["external"]

# -----------------------
# Guided selector
# -----------------------
st.markdown("### Guided selections")
left, right = st.columns([1.2, 0.8])

with left:
    grouping_hint = [c for c in ["State","SubCategory","Region"] if c in df.columns]
    group_keys = st.multiselect("Grouping keys (for lag creation & filtering)", options=list(df.columns), default=grouping_hint)
    if not y_default:
        st.warning("Could not detect a clear dependent variable. Pick one below.")
    y_col = st.selectbox("Dependent (Y)", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
                         index=([i for i,c in enumerate(df.columns) if c==y_default][0] if y_default in df.columns else 0))
    st.caption("Tip: Use 'Secondary Vol. (KL)' if available.")

    st.write("Bucket picks (you can edit):")
    media_picks = st.multiselect("Media X", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], default=media_default)
    trade_picks = st.multiselect("Trade X", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], default=trade_default)
    external_picks = st.multiselect("External X", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], default=external_default)

with right:
    st.write("Lag suggestions")
    lag_suggest = [c for c in media_picks + trade_picks if any(k in c.lower() for k in ["tv","digital","youtube","video","ooh","print","in-bill","inbill","cn"])]
    lag_vars = st.multiselect("Create Lag-1 for:", options=list(set(media_picks + trade_picks + external_picks)), default=lag_suggest)
    use_only_lag = st.checkbox("Use only lagged versions (replace with _Lag1)", value=True)

    st.write("One-click presets")
    preset = st.radio("Preset:", ["Bucket tests -> Final (recommended)", "Single model only"], index=0)

# Build working frame with lags
x_all = media_picks + trade_picks + external_picks
df_work = df.copy()
if lag_vars:
    df_work = make_lag(df_work, group_keys, lag_vars, lag=1)
    if use_only_lag:
        x_all = [x for x in x_all if x not in lag_vars] + [x + "_Lag1" for x in lag_vars]
    else:
        x_all = x_all + [x + "_Lag1" for x in lag_vars]

# -----------------------
# Run bucket tests (if preset selected)
# -----------------------
def bucket_run(title, x_cols):
    st.subheader(title)
    if not x_cols:
        st.info("No variables selected for this bucket.")
        return None, None, None, None
    model, work, coef_df, fit_df = run_ols(df_work, y_col, x_cols, add_const=True)
    if model is None:
        st.warning("Model could not run (missing data after NA drop).")
        return None, None, None, None
    st.table(fit_df)
    st.dataframe(coef_df)
    return model, work, coef_df, fit_df

keepers = []
if preset.startswith("Bucket"):
    st.markdown("## Bucket tests")
    m_model, m_work, m_coef, m_fit = bucket_run("Media-only model", [c for c in x_all if any(k in c.lower() for k in MEDIA_KEYS)])
    t_model, t_work, t_coef, t_fit = bucket_run("Trade-only model", [c for c in x_all if any(k in c.lower() for k in TRADE_KEYS)])
    e_model, e_work, e_coef, e_fit = bucket_run("External-only model", [c for c in x_all if any(k in c.lower() for k in EXTERNAL_KEYS)])

    def add_keepers(cdf, logic="pos"):
        if cdf is None: return
        for _, r in cdf.iterrows():
            v = r["Variable"]
            if v == "const": continue
            if r["p_value"] <= 0.15:
                if logic == "pos" and r["Coefficient"] > 0:
                    keepers.append(v)
                if logic == "neg" and r["Coefficient"] < 0:
                    keepers.append(v)
    add_keepers(m_coef, "pos")
    add_keepers(t_coef, "pos")
    if e_coef is not None:
        for _, r in e_coef.iterrows():
            v=r["Variable"]; lc=v.lower()
            if v=="const": continue
            if r["p_value"]<=0.15 and (("sov" in lc and r["Coefficient"]<0) or ("ofr" in lc and r["Coefficient"]>0) or ("pdo" in lc and r["Coefficient"]>0) or ("pco" in lc and r["Coefficient"]>0)):
                keepers.append(v)

    keepers = list(dict.fromkeys(keepers))[:12]
    st.success(f"Selected keepers for final: {keepers or '—'}")

# -----------------------
# Final model
# -----------------------
st.markdown("## Final model")
X_final = (keepers if preset.startswith('Bucket') and keepers else x_all)
st.caption("Variables used: " + (", ".join(X_final) if X_final else "None"))
f_model, f_work, f_coef, f_fit = run_ols(df_work, y_col, X_final, add_const=True)
if f_model is None:
    st.error("Final model failed (likely no usable rows after NA drop). Try fewer variables or different lags.")
    st.stop()

left2, right2 = st.columns([1.1, 0.9])
with left2:
    st.subheader("Executive Summary")
    st.markdown(executive_summary(f_coef, f_fit, y_name=y_col))
    st.subheader("Model Fit")
    st.table(f_fit)
    st.subheader("Coefficients")
    st.dataframe(f_coef)
with right2:
    st.subheader("Charts")
    fig1, ax1 = plt.subplots()
    ax1.plot(f_work.index, f_work[y_col], label="Actual")
    ax1.plot(f_work.index, f_work["Predicted"], label="Predicted")
    ax1.set_title("Actual vs Predicted")
    ax1.legend()
    st.pyplot(fig1)
    bar = f_coef[f_coef["Variable"]!="const"].sort_values("Coefficient")
    fig2, ax2 = plt.subplots()
    ax2.barh(bar["Variable"], bar["Coefficient"])
    ax2.set_title("Final model coefficients (impact/unit)")
    st.pyplot(fig2)

# -----------------------
# ROI simulator
# -----------------------
st.markdown("## ROI Simulator")
asp = st.number_input("Average Selling Price (₹ per litre) - optional", min_value=0.0, value=0.0, step=10.0)
roi_var = st.selectbox("Select driver", options=[v for v in f_coef["Variable"] if v!="const"])
pct_change = st.slider("Percent change", min_value=-50, max_value=200, value=10, step=5)

used_vars = [v for v in f_coef["Variable"] if v!="const"]
base_vals = f_work[used_vars].mean()
coef_map = dict(zip(f_coef["Variable"], f_coef["Coefficient"]))
baseline = float(base_vals.get(roi_var, np.nan))
coef = float(coef_map.get(roi_var, 0.0))
delta_units = baseline * (pct_change/100.0)
incr_kl = coef * delta_units

st.write(f"{roi_var} baseline (avg): {baseline:.3f} -> change of {delta_units:.3f} units for {pct_change}% delta.")
st.write(f"Predicted incremental volume: {incr_kl:.2f} KL")
if asp > 0:
    incr_rev = incr_kl * asp * 1000.0
    st.write(f"Predicted incremental revenue: ₹ {incr_rev:,.0f}")

# -----------------------
# Downloads
# -----------------------
st.markdown("## Downloads")
preds = f_work[[y_col, "Predicted"]].copy()
dfs = {"Final_Coefficients": f_coef, "Final_Fit": f_fit, "Predictions": preds}
download_excel(dfs, name="BirlaOpus_Guided_MMM_Results.xlsx")

html = f"""

<h2>Birla Opus - MMM Guided Summary</h2>
<p><b>Dependent:</b> {y_col}</p>
<h3>Executive Summary</h3>
<pre>{executive_summary(f_coef, f_fit, y_name=y_col)}</pre>
<h3>Coefficients</h3>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Variable</th><th>Coefficient</th><th>p-value</th></tr>
{''.join([f"<tr><td>{r.Variable}</td><td>{r.Coefficient:.3f}</td><td>{r.p_value:.3f}</td></tr>" for r in f_coef.itertuples()])}
</table>
<h3>ROI Simulation</h3>
<p>Driver: {roi_var} | Delta%: {pct_change}% | Incremental Volume: {incr_kl:.2f} KL{(' | Incremental Revenue: ₹ ' + format(incr_rev, ',.0f')) if asp>0 else ''}</p>

"""
st.download_button("Download HTML Summary", data=html.encode("utf-8"), file_name="BirlaOpus_Guided_MMM_Summary.html", mime="text/html")

# -----------------------
# Guidance block (FAQ style)
# -----------------------
st.markdown("---")
st.markdown("### What do these sections mean?")
st.markdown("""
- Grouping keys: columns like State/SubCategory/Region. Used to create lags within each group so you don't mix timings across markets.
- Bucket tests: we first test Media-only, Trade-only, and External-only to identify clean, logical drivers. Then we build a combined final model using only the keepers.
- Lag-1: aligns last month's driver to this month's sales (e.g., Jan TV -> Feb sales).
- Executive Summary: quick model strength and which levers matter most.
- ROI simulator: choose a driver and change %, we compute incremental KL (and ₹ if you enter ASP).
""")
