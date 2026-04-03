import io
import json
import os
import tempfile
import warnings
import zipfile
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
  page_title="GlycoForge",
  page_icon="⚒️",
  layout="wide",
  initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _pca_fig(df_clr, bio_groups, batch_groups=None, title=""):
  """Return a matplotlib Figure with 1 or 2 PCA panels."""
  X = df_clr.values.T
  pca = PCA(n_components=2)
  coords = pca.fit_transform(X)
  sample_names = df_clr.columns.tolist()
  n_panels = 1 + (batch_groups is not None)
  fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
  if n_panels == 1:
    axes = [axes]
  bio_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
  for g_idx, (gname, gcols) in enumerate(bio_groups.items()):
    idxs = [sample_names.index(c) for c in gcols if c in sample_names]
    axes[0].scatter(coords[idxs, 0], coords[idxs, 1],
                    c=bio_colors[g_idx % len(bio_colors)],
                    label=gname, alpha=0.8, s=60, edgecolors="white", linewidths=0.5)
  axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
  axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
  axes[0].set_title(f"{title}\n(by group)")
  axes[0].legend(fontsize=8)
  axes[0].grid(alpha=0.25)
  if batch_groups is not None:
    batch_colors = ["#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#FF6B35"]
    for b_idx, (bid, bcols) in enumerate(sorted(batch_groups.items())):
      idxs = [sample_names.index(c) for c in bcols if c in sample_names]
      axes[1].scatter(coords[idxs, 0], coords[idxs, 1],
                      c=batch_colors[b_idx % len(batch_colors)],
                      label=f"Batch {bid}", alpha=0.8, s=60,
                      edgecolors="white", linewidths=0.5)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].set_title(f"{title}\n(by batch)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
  fig.tight_layout()
  return fig


def _pvca_bar(pvca_dict):
  """Return a small horizontal bar chart for PVCA results."""
  labels = ["Batch", "Biological", "Residual"]
  vals = [pvca_dict["batch_variance_pct"],
          pvca_dict["bio_variance_pct"],
          pvca_dict["residual_variance_pct"]]
  colors = ["#d62728", "#1f77b4", "#7f7f7f"]
  fig, ax = plt.subplots(figsize=(5, 1.8))
  bars = ax.barh(labels, vals, color=colors, edgecolor="white", height=0.5)
  ax.set_xlim(0, 100)
  ax.set_xlabel("% variance")
  ax.set_title("PVCA variance decomposition")
  for bar, val in zip(bars, vals):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9)
  ax.grid(axis="x", alpha=0.25)
  fig.tight_layout()
  return fig


def _zip_dir(directory):
  """Return bytes of a zip archive of all files in directory."""
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(directory):
      fpath = os.path.join(directory, fname)
      if os.path.isfile(fpath):
        zf.write(fpath, arcname=fname)
  return buf.getvalue()


def _run_simulate(params, out_dir):
  """Call glycoforge.pipeline.simulate, capturing stdout."""
  from glycoforge.pipeline import simulate
  stdout_buf = io.StringIO()
  with contextlib.redirect_stdout(stdout_buf):
    result = simulate(**params, output_dir=out_dir, show_pca_plots=False, save_csv=True)
  log = stdout_buf.getvalue()
  return result, log


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚒️ GlycoForge")
st.sidebar.markdown("Configure your simulation below, then click **Run**.")

mode = st.sidebar.radio("Data source", ["Simulated", "Real"], horizontal=True)
data_source = mode.lower()

# Upload (only shown in real mode)
uploaded_file = None
healthy_prefix = "R7"
unhealthy_prefix = "BM"
use_real_es = False
if data_source == "real":
  uploaded_file = st.sidebar.file_uploader("Glycomics CSV", type=["csv"])
  col_a, col_b = st.sidebar.columns(2)
  healthy_prefix = col_a.text_input("Healthy prefix", value="R7")
  unhealthy_prefix = col_b.text_input("Unhealthy prefix", value="BM")
  use_real_es = st.sidebar.checkbox("Use real effect sizes", value=True)

st.sidebar.divider()
st.sidebar.subheader("Sample structure")
if data_source == "simulated":
  n_glycans = st.sidebar.number_input("# glycans", min_value=5, max_value=500, value=50, step=5)
else:
  n_glycans = None  # determined by data
col1, col2 = st.sidebar.columns(2)
n_H = col1.number_input("n healthy", min_value=3, max_value=200, value=15)
n_U = col2.number_input("n unhealthy", min_value=3, max_value=200, value=15)

st.sidebar.subheader("Biological signal")
bio_strength = st.sidebar.slider("bio_strength (λ)", 0.0, 5.0, 1.5, 0.1,
  help="Effect injection magnitude in CLR space")
k_dir = st.sidebar.slider("k_dir (concentration)", 10, 500, 100, 10,
  help="Dirichlet concentration; higher = less within-group variance")
variance_ratio = st.sidebar.slider("variance_ratio", 0.5, 5.0, 1.5, 0.1,
  help="Unhealthy / Healthy variance ratio")
diff_mask_mode = st.sidebar.selectbox("Differential mask", ["significant", "all", "null"],
  help="Which glycans receive effect injection")

st.sidebar.subheader("Batch effects")
n_batches = st.sidebar.slider("# batches", 2, 10, 3)
kappa_mu = st.sidebar.slider("kappa_mu (mean shift)", 0.0, 5.0, 1.0, 0.1)
var_b = st.sidebar.slider("var_b (variance inflation)", 0.0, 3.0, 0.5, 0.1)

st.sidebar.subheader("Missingness")
missing_fraction = st.sidebar.slider("missing_fraction", 0.0, 0.8, 0.0, 0.05)
mnar_bias = st.sidebar.slider("mnar_bias", 0.5, 5.0, 1.0, 0.1,
  help="Steepness of intensity-dependent missingness")

st.sidebar.subheader("Reproducibility")
seeds_str = st.sidebar.text_input("Random seeds (comma-separated)", value="42")
try:
  random_seeds = [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
except ValueError:
  random_seeds = [42]
  st.sidebar.warning("Could not parse seeds; using [42]")

run_btn = st.sidebar.button("▶  Run simulation", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("GlycoForge simulation")
st.caption("Realistic glycomics data simulation with controllable batch effects and MNAR missingness.")

if not run_btn:
  st.info("Configure parameters in the sidebar and click **Run simulation** to start.")
  st.stop()

# Validate real-mode inputs
if data_source == "real" and uploaded_file is None:
  st.error("Please upload a glycomics CSV file for real-data mode.")
  st.stop()

# ── Execute ───────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp_dir:
  data_file_path = None
  if data_source == "real":
    data_file_path = os.path.join(tmp_dir, "input.csv")
    with open(data_file_path, "wb") as f:
      f.write(uploaded_file.read())

  params = dict(
    data_source=data_source,
    n_H=int(n_H),
    n_U=int(n_U),
    bio_strength=float(bio_strength),
    k_dir=float(k_dir),
    variance_ratio=float(variance_ratio),
    differential_mask=diff_mask_mode,
    n_batches=int(n_batches),
    kappa_mu=float(kappa_mu),
    var_b=float(var_b),
    missing_fraction=float(missing_fraction),
    mnar_bias=float(mnar_bias),
    random_seeds=random_seeds
  )
  if data_source == "simulated":
    params["n_glycans"] = int(n_glycans)
  else:
    params["data_file"] = data_file_path
    params["column_prefix"] = {"healthy": healthy_prefix, "unhealthy": unhealthy_prefix}
    params["use_real_effect_sizes"] = use_real_es

  with st.spinner("Running GlycoForge simulation…"):
    try:
      result, log = _run_simulate(params, tmp_dir)
    except Exception as e:
      st.error(f"Simulation failed: {e}")
      with st.expander("Full traceback"):
        import traceback
        st.code(traceback.format_exc())
      st.stop()

  # ── Build display objects from saved CSVs ─────────────────────────────────
  seed = random_seeds[0]  # display first seed's results

  def _load(pattern):
    path = os.path.join(tmp_dir, pattern)
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None

  Y_clean_clr = _load(f"1_Y_clean_clr_seed{seed}.csv")
  Y_batch_clr = _load(f"2_Y_with_batch_clr_seed{seed}.csv")
  Y_missing_clr = _load(f"3_Y_with_batch_and_missing_clr_seed{seed}.csv")
  Y_clean = _load(f"1_Y_clean_seed{seed}.csv")
  Y_batch = _load(f"2_Y_with_batch_seed{seed}.csv")

  meta = result["metadata"][0]
  bio_groups = meta["sample_info"]["bio_groups"]
  batch_groups_raw = meta["sample_info"]["batch_groups"]
  batch_groups = {int(k): v for k, v in batch_groups_raw.items()}

  pvca = meta["quality_checks"].get("Y_with_batch", {}).get("pvca", None)
  overall = meta["quality_checks"].get("Y_with_batch", {}).get("overall_quality", {})
  bio_check_clean = meta["quality_checks"].get("Y_clean", {})

  # ── Tabs ──────────────────────────────────────────────────────────────────
  tab_pca, tab_metrics, tab_data, tab_log = st.tabs(
    ["📊 PCA", "📈 Metrics", "💾 Data & Download", "🪵 Log"]
  )

  with tab_pca:
    st.subheader("Clean data (before batch effects)")
    if Y_clean_clr is not None:
      st.pyplot(_pca_fig(Y_clean_clr, bio_groups, title="Clean CLR"), use_container_width=False)
    st.subheader("After batch effect injection")
    if Y_batch_clr is not None:
      st.pyplot(_pca_fig(Y_batch_clr, bio_groups, batch_groups, title="Batch CLR"),
                use_container_width=False)
    if Y_missing_clr is not None:
      st.subheader("After batch + missingness")
      st.pyplot(_pca_fig(Y_missing_clr, bio_groups, batch_groups, title="Missing CLR"),
                use_container_width=False)

  with tab_metrics:
    st.subheader("Biological signal quality")
    bio_eff = bio_check_clean.get("bio_effect", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("η² (clean)", f"{bio_eff.get('effect_size_eta2', 0):.1%}")
    c2.metric("Centroid dist.", f"{bio_eff.get('centroid_distance', 0):.2f}")
    c3.metric("Signal strength", bio_eff.get("strength", "—"))

    st.subheader("Batch effect (PVCA)")
    if pvca:
      col_a, col_b = st.columns([1.2, 1])
      with col_a:
        st.pyplot(_pvca_bar(pvca), use_container_width=False)
      with col_b:
        severity = overall.get("severity", "—")
        colour = {"NONE":"🟢","GOOD":"🟢","MILD":"🟡",
                  "WARNING":"🟠","MODERATE":"🟠","CRITICAL":"🔴"}.get(severity, "⚪")
        st.markdown(f"### {colour} {severity}")
        st.caption(overall.get("severity_description", ""))
        st.metric("Batch variance", f"{pvca['batch_variance_pct']:.1f}%")
        st.metric("Bio variance", f"{pvca['bio_variance_pct']:.1f}%")
        st.metric("Residual variance", f"{pvca['residual_variance_pct']:.1f}%")
    else:
      st.info("PVCA not available (requires bio_labels).")

    if missing_fraction > 0:
      st.subheader("Missingness diagnostics")
      miss_diag = meta["quality_checks"].get("missingness", {})
      mc1, mc2, mc3 = st.columns(3)
      mc1.metric("Target fraction", f"{miss_diag.get('missing_fraction_target',0):.1%}")
      mc2.metric("Actual fraction", f"{miss_diag.get('missing_fraction_actual',0):.1%}")
      mc3.metric("Total missing", miss_diag.get("total_missing", "—"))
      mnar_by_int = miss_diag.get("missing_rate_by_intensity", {})
      if mnar_by_int:
        st.markdown("**Missing rate by intensity bin:**")
        mnar_df = pd.DataFrame({"Intensity bin": list(mnar_by_int.keys()),
                                "Missing rate": [f"{v:.1%}" for v in mnar_by_int.values()]})
        st.dataframe(mnar_df, hide_index=True, use_container_width=False)

  with tab_data:
    st.subheader("Preview: clean data (first seed)")
    if Y_clean is not None:
      st.dataframe(Y_clean.head(10).round(4), use_container_width=True)
    st.subheader("Preview: post-batch data")
    if Y_batch is not None:
      st.dataframe(Y_batch.head(10).round(4), use_container_width=True)

    st.subheader("Download all outputs")
    zip_bytes = _zip_dir(tmp_dir)
    st.download_button(
      label="⬇ Download ZIP (all CSVs + metadata)",
      data=zip_bytes,
      file_name=f"glycoforge_seed{seed}.zip",
      mime="application/zip",
      use_container_width=True,
    )
    st.subheader("Metadata JSON")
    with st.expander("Show metadata"):
      st.json(meta)

  with tab_log:
    st.subheader("Simulation log")
    st.code(log if log.strip() else "(verbose=True output will appear here)")
