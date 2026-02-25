# -*- coding: utf-8 -*-
"""
Streaming implementation of the Datathon v8 model (fixed).

Fixes vs previous version:
1) Prevent train/test leakage: h(l) and prior GLM are fit on TRAIN users only.
2) Prior GLM is fit to per-trial success probability (grouped binomial), not
   the "any correct in session" indicator, so p_prior is consistent with the
   Beta-Binomial update used later.
3) Safer handling of edge cases (language scaling denominator, regression denom,
   invalid count rows).

Pipeline:
1) Pass 1: collect unique users -> train/test split
2) Pass 2: estimate h(l) = beta0 + beta1*l via streaming regression on TRAIN users
3) Pass 3: sample low-history TRAIN rows, fit logistic prior coefficients (b0,bL,bT,bl)
   using grouped binomial GLM on session_correct/session_seen
4) Pass 4: precompute numeric feature file (all rows; evaluation still split by users)
5) Optimize z and half-life h on a subsample of train rows
6) Compute final NLL on full train & test (streaming)
"""

import numpy as np
import pandas as pd
import re
from scipy.special import expit
from scipy.optimize import minimize
import statsmodels.api as sm

# ================== CONFIG ==================

BIGFILE = r"D:\OneDrive - KU Leuven\Datathon\Data_set_A\Data Set A_ Spaced Repetition\learning_traces.13m.csv"
SEP = ","          # change to "\t" if your file is actually TSV
CHUNK_SIZE = 200_000
RANDOM_SEED = 42

FEATURE_FILE = r"D:\OneDrive - KU Leuven\Datathon\Data_set_A\precomputed_features.csv"

EPS = 1e-9        # to avoid log(0)
EPS_F = 1e-3      # epsilon in f(L)

# Training speed controls
USE_SUBSAMPLE_FOR_TRAINING = True
TRAIN_SUBSAMPLE_ROWS = 1_000_000   # rows used per NLL eval while fitting
MAXITER = 50

# Prior fitting controls
PRIOR_SAMPLE_MAX = 1_000_000
LOW_HISTORY_MAX = 3  # rows with history_seen <= LOW_HISTORY_MAX are used for h(l) and prior GLM

# If TRUE, the prior GLM uses a grouped-binomial target (session_correct/session_seen),
# which matches the model derivation where p_prior is a per-trial recall probability.
USE_GROUPED_BINOMIAL_PRIOR = True

# ==================  FEATURE FUNCTIONS  ==================

# ---- 1) Language difficulty f(L) from FSI-like hours ----
HOURS = {
    "fr": 675,
    "en": 600,
    "es": 675,
    "de": 750,
    "pt": 675,
    "it": 675,
}
DEFAULT_HOURS = 700

# Compute scaling bounds robustly from ALL language hour values considered by the map.
H_vals = list(HOURS.values())
H_min = min(H_vals)
H_max = max(H_vals)
_H_RANGE = max(H_max - H_min, 1e-12)


def f_language(lang_code: str) -> float:
    H = HOURS.get(lang_code, DEFAULT_HOURS)
    return EPS_F + (1 - 2 * EPS_F) * (1 - (H - H_min) / _H_RANGE)


# ---- 2) POS-based difficulty g(T) ----
def extract_pos_score(cell: str):
    """
    Map lexeme_string tags to difficulty g(T).
    Handles cases like:
      "hats/hat<n><pl>"
      "<*sf>/student<n><*numb>"
    """
    if not isinstance(cell, str):
        return np.nan

    tags = re.findall(r"<([^>]+)>", cell)
    if not tags:
        return 0.6  # neutral default

    tags = [t.lstrip("*").lower() for t in tags]

    if any(t.startswith("v") or t == "sep" for t in tags):
        return 0.89
    if any(t in ["n", "ant", "cog", "np"] for t in tags):
        return 0.92
    if any(t in ["num"] for t in tags):
        return 0.95
    if any(t in ["acr", "adj", "atn", "comp", "dem", "det", "detnt", "enc", "itg",
                 "obj", "ord", "pos", "pro", "qnt", "ref", "rel", "sint", "sup", "tn"]
           for t in tags):
        return 0.65
    if any(t in ["prn", "pron", "prpers"] for t in tags):
        return 0.531
    if any(t == "pr" for t in tags):
        return 0.469
    if any(t in ["adv", "cnjadv", "preadv"] for t in tags):
        return 0.45

    return 0.6


# ---- 3) Word length (for h(l)) ----
def extract_word_length(lexeme_string: str) -> float:
    """
    Length of longest surface form in lexeme_string.
    Example:
      "hats/hat<n><pl>"          -> len("hats") = 4
      "<*sf>/student<n><*numb>"  -> len("student") = 7
    """
    if not isinstance(lexeme_string, str):
        return np.nan
    without_tags = re.sub(r"<[^>]*>", "", lexeme_string)
    forms = without_tags.split("/")
    cleaned = ["".join(ch for ch in f if ch.isalpha()) for f in forms]
    cleaned = [f for f in cleaned if f]
    if not cleaned:
        return np.nan
    return max(len(f) for f in cleaned)


# ================== PASS 1: COLLECT UNIQUE USERS ==================

print("Pass 1: collecting unique users...")
unique_users = set()

for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE, usecols=["user_id"]):
    unique_users.update(chunk["user_id"].dropna().unique())

unique_users = list(unique_users)
print(f"Total unique users: {len(unique_users)}")

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(unique_users)

n_train_users = int(0.8 * len(unique_users))
train_users = set(unique_users[:n_train_users])
test_users = set(unique_users[n_train_users:])

print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")


# ================== PASS 2: ESTIMATE h(l) ON TRAIN USERS ==================

print("Pass 2: estimating h(l) = beta0 + beta1*l via streaming regression (TRAIN users only)...")

Sx = Sy = Sxx = Sxy = 0.0
N_reg = 0

for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):
    chunk = chunk[chunk["user_id"].isin(train_users)]
    if chunk.empty:
        continue

    df_h = chunk[(chunk["history_seen"] <= LOW_HISTORY_MAX) & (chunk["session_seen"] > 0)].copy()
    if df_h.empty:
        continue

    df_h["word_len"] = df_h["lexeme_string"].apply(extract_word_length)
    df_h = df_h[df_h["word_len"].notna()]
    if df_h.empty:
        continue

    # Per-row empirical success probability (used only to estimate h(l) feature map)
    df_h["acc"] = df_h["session_correct"] / df_h["session_seen"]
    df_h = df_h.replace([np.inf, -np.inf], np.nan).dropna(subset=["acc", "word_len"])
    if df_h.empty:
        continue

    x = df_h["word_len"].values.astype(float)
    y = df_h["acc"].values.astype(float)

    Sx += x.sum()
    Sy += y.sum()
    Sxx += np.dot(x, x)
    Sxy += np.dot(x, y)
    N_reg += len(df_h)

if N_reg < 2:
    raise RuntimeError("Not enough data to fit h(l).")

reg_denom = (N_reg * Sxx - Sx * Sx)
if abs(reg_denom) < 1e-12:
    raise RuntimeError("Degenerate regression for h(l): word_len has near-zero variance.")

beta1 = (N_reg * Sxy - Sx * Sy) / reg_denom
beta0 = (Sy - beta1 * Sx) / N_reg

print(f"h(l): beta0 = {beta0:.4f}, beta1 = {beta1:.4f}  (from {N_reg} TRAIN rows)")


# ================== PASS 3: FIT PRIOR COEFFICIENTS ON TRAIN USERS ==================

print("Pass 3: sampling low-history TRAIN rows to fit prior coefficients (b0, bL, bT, bl)...")

rows_list = []
rows_kept = 0

for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):
    chunk = chunk[chunk["user_id"].isin(train_users)]
    if chunk.empty:
        continue

    df_p = chunk[(chunk["history_seen"] <= LOW_HISTORY_MAX) & (chunk["session_seen"] > 0)].copy()
    if df_p.empty:
        continue

    # Compute features
    df_p["f_L"] = df_p["learning_language"].apply(f_language)
    df_p["g_T"] = df_p["lexeme_string"].apply(extract_pos_score)
    df_p["word_len"] = df_p["lexeme_string"].apply(extract_word_length)
    df_p["h_l"] = beta0 + beta1 * df_p["word_len"]

    # Keep counts for grouped-binomial likelihood of per-trial success probability
    keep_cols = ["f_L", "g_T", "h_l", "session_correct", "session_seen"]
    df_p = df_p[keep_cols]
    df_p = df_p.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols)
    if df_p.empty:
        continue

    # Sanity filter invalid count rows
    df_p = df_p[(df_p["session_seen"] > 0) & (df_p["session_correct"] >= 0) &
                (df_p["session_correct"] <= df_p["session_seen"])]
    if df_p.empty:
        continue

    # Cap sample size (simple truncation to keep memory bounded). For an unbiased sample,
    # reservoir sampling can be added later.
    remaining = PRIOR_SAMPLE_MAX - rows_kept
    if remaining <= 0:
        break
    if len(df_p) > remaining:
        df_p = df_p.iloc[:remaining].copy()

    rows_list.append(df_p)
    rows_kept += len(df_p)

    if rows_kept >= PRIOR_SAMPLE_MAX:
        break

if not rows_list:
    raise RuntimeError("No low-history TRAIN rows found for prior fitting.")

prior_df = pd.concat(rows_list, ignore_index=True)
print(f"Prior GLM sample size (after cleaning): {len(prior_df)}")

if len(prior_df) == 0:
    raise RuntimeError("No valid rows left for prior fitting after cleaning.")

X_prior = prior_df[["f_L", "g_T", "h_l"]].values.astype(float)
X_prior = sm.add_constant(X_prior)  # [1, f_L, g_T, h_l]

if USE_GROUPED_BINOMIAL_PRIOR:
    # Fit p_prior as PER-TRIAL recall probability using grouped binomial GLM.
    y_prop = (prior_df["session_correct"].values.astype(float) /
              prior_df["session_seen"].values.astype(float))
    n_trials = prior_df["session_seen"].values.astype(float)

    prior_model = sm.GLM(
        y_prop,
        X_prior,
        family=sm.families.Binomial(),
        var_weights=n_trials,
    )
else:
    # Backward-compatible option (not faithful to the derivation): any success in session.
    y_prior = (prior_df["session_correct"].values > 0).astype(int)
    prior_model = sm.GLM(y_prior, X_prior, family=sm.families.Binomial())

prior_res = prior_model.fit()
b0, bL, bT, bl = prior_res.params

print("Prior coefficients (from GLM):")
print(f"b0 = {b0:.4f}, bL = {bL:.4f}, bT = {bT:.4f}, bl = {bl:.4f}")
print(f"Prior GLM mode: {'grouped binomial (per-trial p)' if USE_GROUPED_BINOMIAL_PRIOR else 'any-success Bernoulli'}")


# ================== PASS 4: PRECOMPUTE NUMERIC FEATURE FILE ==================

print("Pass 4: precomputing numeric feature file (one-time cost)...")

with open(FEATURE_FILE, "w", newline="", encoding="utf-8") as f_out:
    first = True

    for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):
        # Keep only valid count rows
        chunk = chunk[(chunk["session_seen"] > 0) &
                      (chunk["session_correct"] >= 0) &
                      (chunk["session_correct"] <= chunk["session_seen"])]
        if chunk.empty:
            continue

        chunk = chunk.copy()
        chunk["f_L"] = chunk["learning_language"].apply(f_language)
        chunk["g_T"] = chunk["lexeme_string"].apply(extract_pos_score)
        chunk["word_len"] = chunk["lexeme_string"].apply(extract_word_length)
        chunk["h_l"] = beta0 + beta1 * chunk["word_len"]
        chunk["t_hours"] = chunk["delta"].astype(float) / 3600.0

        cols = [
            "user_id", "f_L", "g_T", "h_l", "t_hours",
            "history_seen", "history_correct",
            "session_seen", "session_correct",
        ]
        chunk = chunk[cols]
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        if chunk.empty:
            continue

        if first:
            chunk.to_csv(f_out, index=False)
            first = False
        else:
            chunk.to_csv(f_out, index=False, header=False)

print(f"Feature file written to: {FEATURE_FILE}")


# ================== MODEL ON FEATURE FILE ==================

def model_probs_on_features(df_sub: pd.DataFrame, theta):
    """
    Compute P_t for rows in df_sub, using precomputed numeric features.

    theta = [z, h_half]
    where z controls Beta prior concentration and h_half is half-life in hours.
    """
    z, h_half = theta
    z = max(float(z), 1e-6)
    h_half = max(float(h_half), 1e-6)

    f_L_vals = df_sub["f_L"].values.astype(float)
    g_T_vals = df_sub["g_T"].values.astype(float)
    h_l_vals = df_sub["h_l"].values.astype(float)
    t_vals = df_sub["t_hours"].values.astype(float)

    linear_prior = b0 + bL * f_L_vals + bT * g_T_vals + bl * h_l_vals
    p_prior = expit(linear_prior)  # per-trial recall probability prior

    alpha_p = z * p_prior
    beta_p = z * (1.0 - p_prior)

    hist_seen = df_sub["history_seen"].values.astype(float)
    hist_correct = df_sub["history_correct"].values.astype(float)

    # Posterior Beta parameters after incorporating history counts
    alpha = alpha_p + hist_correct
    beta_ = beta_p + (hist_seen - hist_correct)

    P = alpha / (alpha + beta_)

    # Approximate decay outside the Beta update (as in the derivation)
    d = 2.0 ** (-t_vals / h_half)
    P_t = np.clip(d * P, EPS, 1.0 - EPS)
    return P_t


def nll_over_features(theta, user_set, max_rows=None):
    """
    Average row-level negative log-likelihood over rows in FEATURE_FILE for users in user_set.
    Uses Binomial(row) log-likelihood with (session_seen, session_correct).
    """
    LL_sum = 0.0
    N_total = 0

    for chunk in pd.read_csv(FEATURE_FILE, chunksize=CHUNK_SIZE):
        df_sub = chunk[chunk["user_id"].isin(user_set)].copy()
        if df_sub.empty:
            continue

        P_t = model_probs_on_features(df_sub, theta)
        n = df_sub["session_seen"].values.astype(float)
        m = df_sub["session_correct"].values.astype(float)

        logY = m * np.log(P_t) + (n - m) * np.log(1.0 - P_t)
        LL_sum += float(logY.sum())
        N_total += len(df_sub)

        if max_rows is not None and N_total >= max_rows:
            break

    if N_total == 0:
        raise RuntimeError("No rows found for given user_set in feature file.")

    return -LL_sum / N_total


# ================== FIT z AND HALF-LIFE h ==================

print("Fitting z and half-life h on training users (subsampled rows for speed)...")

theta0 = np.array([5.0, 24.0])
bounds = [(1e-3, 100.0), (1e-3, 20000.0)]  # [z, h_half]


def objective(theta):
    if USE_SUBSAMPLE_FOR_TRAINING:
        return nll_over_features(theta, train_users, max_rows=TRAIN_SUBSAMPLE_ROWS)
    return nll_over_features(theta, train_users, max_rows=None)


opt_res = minimize(
    objective,
    theta0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": MAXITER},
)

z_hat, h_hat = opt_res.x

print("\nOptimization result for z and h:")
print(opt_res)
print(f"\nEstimated z  = {z_hat:.4f}")
print(f"Estimated h  = {h_hat:.4f} hours (half-life)")


# ================== FINAL TRAIN / TEST NLL ==================

print("\nComputing final NLL on full train and test sets...")

theta_hat = np.array([z_hat, h_hat])

nll_train = nll_over_features(theta_hat, train_users, max_rows=None)
nll_test = nll_over_features(theta_hat, test_users, max_rows=None)

print(f"\nFinal NLL (train): {nll_train:.6f}")
print(f"Final NLL (test) : {nll_test:.6f}")
