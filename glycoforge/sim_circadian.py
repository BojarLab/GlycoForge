import re
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from glycowork.glycan_data.stats import replace_outliers_winsorization, clr_transformation, impute_and_normalize, bh_adjust, cosinor_fit
from glycowork.glycan_data.data_entry import read_abundances

PERIOD = 24.0


def sample_columns(df):
    """Return the sample columns of a circadian dataframe (those matching the T{n}_... naming)."""
    return [c for c in df.columns if re.match(r"^T\d+_", str(c))]


def parse_time(cols, zt_seq=[12, 18, 0, 6, 12, 18, 0, 6, 12], reps=None, cum_seq=None):
    """Map T{n}_ZT{zt}_R{rep} sample columns to cumulative-hour and ZT-phase vectors.
    The T-index gives sample order; reps blocks it into timepoints, cum_seq supplies the
    cumulative hours per timepoint and zt_seq the ZT phase. Robust to missing timepoints
    or short replicate blocks. Returns (cumulative_hours, zt_hours), both length len(cols)."""
    reps = reps if reps is not None else [5] * len(zt_seq)
    cum_seq = cum_seq if cum_seq is not None else [i * 6.0 for i in range(len(zt_seq))]
    tn = np.array([int(c.split("_")[0][1:]) for c in cols])
    tp_sorted = np.concatenate([np.full(r, i) for i, r in enumerate(reps)])
    idx = np.empty(len(cols), dtype=int)
    idx[np.argsort(tn)] = tp_sorted
    return np.array([cum_seq[i] for i in idx], dtype=float), np.array([zt_seq[i] for i in idx], dtype=float)


def cosinor_table(mat, cum):
    """Run cosinor_fit on every feature of mat (features x samples) against cumulative hours cum.
    Returns a per-feature table of amplitude, acrophase_h, F, p, acrophase_ZT, and two-stage
    adaptive BH q-values; zero-variance features are skipped. acrophase_ZT shifts acrophase_h
    by 12 h so that cumulative-hour 0 (the ZT12 start of the course) maps back onto the ZT clock."""
    rows = [(name, (r := cosinor_fit(y, cum, period=PERIOD))["amplitude"], r["acrophase"], r["f_stat"], r["p_value"]) for name, y in zip(mat.index, mat.values) if np.std(y) > 0]
    out = pd.DataFrame(rows, columns=["feature", "amplitude", "acrophase_h", "F", "p"]).set_index("feature")
    out["acrophase_ZT"] = (out["acrophase_h"] + 12) % PERIOD
    out["q"] = bh_adjust(out["p"].to_numpy(), 0.05)
    return out


def prep_compositional(data_file, zt_seq=[12, 18, 0, 6, 12, 18, 0, 6, 12], reps=None, cum_seq=None, drop_cum=None):
    """Load a circadian glycan x sample dataset and return its imputed CLR matrix.
    data_file is a path to a glycan x sample CSV or a glycowork dataset name. Applies row-wise
    Winsorization, rhythm-test imputation (impute_and_normalize with circadian=True, which adds a
    draw of the prediction error to every imputed value), and gamma=0 CLR. drop_cum optionally
    removes whole timepoints by their cumulative hour. Returns (clr_mat, cum, df): clr_mat is
    features x samples with duplicate glycans averaged, cum is the per-sample cumulative-hour vector, df the imputed wide table."""
    df = pd.DataFrame(read_abundances(data_file))
    feat = df.columns[0]
    sc = sample_columns(df)
    df = df[[feat] + sc].copy()
    df[feat] = df[feat].astype(str)
    df[sc] = df[sc].apply(pd.to_numeric, errors="coerce").fillna(0)
    cum, _ = parse_time(sc, zt_seq=zt_seq, reps=reps, cum_seq=cum_seq)
    if drop_cum is not None:
        keep = ~np.isin(cum, drop_cum)
        sc = [c for c, k in zip(sc, keep) if k]
        cum = cum[keep]
        df = df[[feat] + sc]
    df = df.loc[~(df[sc] == 0).all(axis=1)].reset_index(drop=True)
    df = replace_outliers_winsorization(df)
    df = impute_and_normalize(df, [df.columns[1:].tolist()], impute=True, min_samples=0.25, circadian=True)
    data = df.iloc[:, 1:].astype(float)
    data.index = df.iloc[:, 0].values
    clr_mat = clr_transformation(data, data.columns.tolist(), [], gamma=0).groupby(level=0).mean()
    return clr_mat, cum, df


def define_circadian_injection_from_real_data(data_file, zt_seq=[12, 18, 0, 6, 12, 18, 0, 6, 12], reps=None, cum_seq=None, q_thresh=0.05):
    """Fit per-glycan circadian parameters and the residual backbone used to simulate new data.
    Preprocesses data_file, fits a weighted cosinor per glycan, and flags rhythmic glycans at
    fdr_tsbh q < q_thresh. The cosinor coefficients are zeroed for non-rhythmic glycans so the
    injected signal is clean ground truth, and per-sample residuals (CLR minus fitted cosinor)
    plus their Ledoit-Wolf covariance define the arrhythmic noise backbone for the copula.
    Returns a params dict consumed by pipeline.simulate_circadian, holding glycans, period,
    mesor/cos_coef/sin_coef, the rhythmic mask, amplitude/acrophase_ZT, residuals, cov, cum,
    the full cosinor table, and the fitted design (zt_seq, reps, cum_seq)."""
    clr_mat, cum, _ = prep_compositional(data_file, zt_seq=zt_seq, reps=reps, cum_seq=cum_seq)
    tab = cosinor_table(clr_mat, cum)
    glycans = clr_mat.index.to_numpy()
    omega = 2 * np.pi / PERIOD
    mesor = np.empty(len(glycans))
    cos_coef = np.empty(len(glycans))
    sin_coef = np.empty(len(glycans))
    resid = np.empty((clr_mat.shape[1], len(glycans)))
    for i, name in enumerate(glycans):
        y = clr_mat.loc[name].to_numpy()
        mesor[i], cos_coef[i], sin_coef[i] = cosinor_fit(y, cum)["beta"]
        resid[:, i] = y - (mesor[i] + cos_coef[i] * np.cos(omega * cum) + sin_coef[i] * np.sin(omega * cum))
    rhythmic = np.nan_to_num(tab.reindex(glycans)["q"].to_numpy(), nan=1.0) < q_thresh
    return {"glycans": glycans, "period": PERIOD, "mesor": mesor, "cos_coef": np.where(rhythmic, cos_coef, 0.0), "sin_coef": np.where(rhythmic, sin_coef, 0.0), "rhythmic": rhythmic,
            "amplitude": tab.reindex(glycans)["amplitude"].to_numpy(), "acrophase_ZT": tab.reindex(glycans)["acrophase_ZT"].to_numpy(), "residuals": resid, "cov": LedoitWolf().fit(resid).covariance_,
            "cum": cum, "real_tab": tab, "zt_seq": zt_seq, "reps": reps, "cum_seq": cum_seq}