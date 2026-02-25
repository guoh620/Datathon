# -*- coding: utf-8 -*-
"""
Sped-up streaming implementation of your Datathon v7 model.

Pipeline:
1) Pass 1: collect unique users -> train/test split
2) Pass 2: estimate h(l) = beta0 + beta1*l via streaming linear regression
3) Pass 3: sample low-history rows, fit logistic prior coefficients (b0, bL, bT, bl)
4) Pass 4: precompute numeric feature file (no regex during training)
5) Optimize z and half-life h on a subsample of train rows
6) Compute final NLL on full train & test (streaming)

This script never loads the full 13M-row raw file into RAM at once.
"""

import numpy as np
import pandas as pd
import re
from scipy.special import expit
from scipy.optimize import minimize

# ================== CONFIG ==================

BIGFILE = r"D:\OneDrive - KU Leuven\Datathon\Data_set_A\Data Set A_ Spaced Repetition\learning_traces.13m.csv"
SEP = ","          # change to "\t" if your file is actually TSV
CHUNK_SIZE = 200_000
RANDOM_SEED = 42

FEATURE_FILE = r"D:\OneDrive - KU Leuven\Datathon\Data_set_A\precomputed_features.csv"

EPS = 1e-9        # to avoid log(0)
EPS_F = 1e-3      # epsilon in f(L)

# --- Helper: logistic regression for grouped binomial data, no statsmodels needed ---
def fit_logistic_prior(X, successes, trials, eps=1e-9):
    """Fit logistic regression where successes[i] ~ Binomial(trials[i], p_i),
    and logit(p_i) = X[i] @ beta. Returns beta (1D array)."""
    import numpy as _np
    from scipy.special import expit as _expit
    from scipy.optimize import minimize as _minimize

    X = _np.asarray(X, float)
    successes = _np.asarray(successes, float)
    trials = _np.asarray(trials, float)

    mask = trials > 0
    X = X[mask]
    successes = successes[mask]
    trials = trials[mask]

    def nll(beta):
        z = X @ beta
        p = _expit(z)
        p = _np.clip(p, eps, 1.0 - eps)
        return -_np.sum(successes * _np.log(p) + (trials - successes) * _np.log(1.0 - p))

    def grad(beta):
        z = X @ beta
        p = _expit(z)
        p = _np.clip(p, eps, 1.0 - eps)
        return X.T @ (trials * p - successes)

    beta0 = _np.zeros(X.shape[1], dtype=float)
    res = _minimize(nll, beta0, jac=grad, method="L-BFGS-B")
    if not res.success:
        print("WARNING: logistic prior fit did not fully converge:", res.message)
    return res.x

# Training speed controls
USE_SUBSAMPLE_FOR_TRAINING = True
TRAIN_SUBSAMPLE_ROWS = 1_000_000   # rows used per NLL eval while fitting
MAXITER = 50                     # optimizer iterations


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

H_vals = [HOURS.get(lang, DEFAULT_HOURS) for lang in HOURS]
H_min = min(H_vals)
H_max = max(H_vals)

def f_language(lang_code: str) -> float:
    H = HOURS.get(lang_code, DEFAULT_HOURS)
    return EPS_F + (1 - 2 * EPS_F) * (1 - (H - H_min) / (H_max - H_min))


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
    if any(t == "num" for t in tags):
        return 0.95
    if any(t in ["acr", "adj", "atn", "comp", "dem", "det", "detnt", "enc", "itg",
                 "ord", "pos", "pro", "qnt", "ref",
                 "rel", "sint", "sup", "tn"] for t in tags):
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

    parts = lexeme_string.split("/")
    if not parts:
        return np.nan

    max_len = 0
    for part in parts:
        # ignore pure tag chunks like "<*sf>"
        if "<" in part and ">" in part and not part.strip("<>").isalpha():
            continue
        token = re.sub(r"<.*?>", "", part)
        token = token.strip()
        if token:
            max_len = max(max_len, len(token))

    return float(max_len) if max_len > 0 else np.nan


# ================== PASS 1: COLLECT UNIQUE USERS ==================

print("Pass 1: collecting unique users...")
unique_users = set()

for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):
    unique_users.update(chunk["user_id"].unique())

unique_users = list(unique_users)
print(f"Total unique users: {len(unique_users)}")

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(unique_users)

n_train_users = int(0.8 * len(unique_users))
train_users = set(unique_users[:n_train_users])
test_users  = set(unique_users[n_train_users:])

print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")


# ================== PASS 2: ESTIMATE h(l) ==================

print("Pass 2: estimating h(l) = beta0 + beta1*l via streaming regression...")

Sx = Sy = Sxx = Sxy = 0.0
N_reg = 0

for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):

    # use only training users for h(l) regression to avoid leakage
    chunk = chunk[chunk["user_id"].isin(train_users)]
    df_h = chunk[chunk["history_seen"] <= 3].copy()
    if df_h.empty:
        continue

    df_h["word_len"] = df_h["lexeme_string"].apply(extract_word_length)
    df_h = df_h[df_h["word_len"].notna()]

    if df_h.empty:
        continue

    df_h["acc"] = df_h["session_correct"] / df_h["session_seen"]

    x = df_h["word_len"].values.astype(float)
    y = df_h["acc"].values.astype(float)

    Sx  += x.sum()
    Sy  += y.sum()
    Sxx += (x * x).sum()
    Sxy += (x * y).sum()
    N_reg += len(df_h)

if N_reg < 2:
    raise RuntimeError("Not enough data to fit h(l).")

den = (N_reg * Sxx - Sx * Sx)
if abs(den) < 1e-12:
    raise RuntimeError("Degenerate regression denominator when fitting h(l).")

beta1 = (N_reg * Sxy - Sx * Sy) / den
beta0 = (Sy - beta1 * Sx) / N_reg

print(f"h(l): beta0 = {beta0:.4f}, beta1 = {beta1:.4f}  (from {N_reg} rows)")


# ================== PASS 3: SAMPLE LOW-HISTORY ROWS & FIT PRIOR COEFFS ==================

print("Pass 3: sampling low-history rows to fit prior coefficients (b0, bL, bT, bl)...")

PRIOR_SAMPLE_MAX = 1_000_000
rows_list = []

for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):

    # only use training users when fitting the prior to avoid leakage
    chunk = chunk[chunk["user_id"].isin(train_users)]

    df_p = chunk[chunk["history_seen"] <= 3].copy()
    df_p = df_p[df_p["session_seen"] > 0]

    if df_p.empty:
        continue

    df_p["f_L"] = df_p["learning_language"].apply(f_language)
    df_p["g_T"] = df_p["lexeme_string"].apply(extract_pos_score)
    df_p["word_len"] = df_p["lexeme_string"].apply(extract_word_length)
    df_p["h_l"] = beta0 + beta1 * df_p["word_len"]

    # store raw counts for grouped-binomial logistic regression
    rows_list.append(df_p[["f_L", "g_T", "h_l", "session_seen", "session_correct"]])

    total_rows = sum(len(r) for r in rows_list)
    if total_rows >= PRIOR_SAMPLE_MAX:
        break

if not rows_list:
    raise RuntimeError("No low-history rows found for prior fitting.")

prior_df = pd.concat(rows_list, ignore_index=True)
print(f"Prior logistic regression sample size (before cleaning): {len(prior_df)}")

# Drop rows with NaN or inf in features or targets
prior_df = prior_df.replace([np.inf, -np.inf], np.nan)
prior_df = prior_df.dropna(subset=["f_L", "g_T", "h_l", "session_seen", "session_correct"])

print(f"Prior logistic regression sample size (after cleaning): {len(prior_df)}")

if len(prior_df) == 0:
    raise RuntimeError("No valid rows left for prior fitting after cleaning.")

X_prior = prior_df[["f_L", "g_T", "h_l"]].values.astype(float)
# add intercept: [1, f_L, g_T, h_l]
X_prior = np.column_stack([np.ones(len(X_prior)), X_prior])

successes = prior_df["session_correct"].values.astype(float)
trials = prior_df["session_seen"].values.astype(float)

beta_prior = fit_logistic_prior(X_prior, successes, trials)
b0, bL, bT, bl = beta_prior
print("Prior coefficients (from grouped-binomial logistic regression):")
print(f"b0 = {b0:.4f}, bL = {bL:.4f}, bT = {bT:.4f}, bl = {bl:.4f}")


# ================== PASS 4: PRECOMPUTE NUMERIC FEATURE FILE ==================

print("Pass 4: precomputing numeric feature file (this is a one-time cost)...")

with open(FEATURE_FILE, "w", newline="", encoding="utf-8") as f_out:
    first = True
    for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):

        # Keep only rows with valid session_seen
        chunk = chunk[chunk["session_seen"] > 0]
        if chunk.empty:
            continue

        # Compute features
        chunk["f_L"] = chunk["learning_language"].apply(f_language)
        chunk["g_T"] = chunk["lexeme_string"].apply(extract_pos_score)
        chunk["word_len"] = chunk["lexeme_string"].apply(extract_word_length)
        chunk["h_l"] = beta0 + beta1 * chunk["word_len"]
        chunk["t_hours"] = chunk["delta"] / 3600.0

        # Columns we want to keep
        cols = [
            "user_id",
            "f_L", "g_T", "h_l", "t_hours",
            "history_seen", "history_correct",
            "session_seen", "session_correct",
        ]

        # Select only these columns
        chunk = chunk[cols]
        
        # Drop any rows with NaN / inf in the numeric features
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=[
            "user_id",
            "f_L", "g_T", "h_l", "t_hours",
            "history_seen", "history_correct",
            "session_seen", "session_correct",
        ])

        if first:
            chunk.to_csv(f_out, index=False)
            first = False
        else:
            chunk.to_csv(f_out, index=False, header=False)

print(f"Numeric feature file written to {FEATURE_FILE}")


# ================== MODEL: PROBS ON FEATURES ==================

def model_probs_on_features(df_sub: pd.DataFrame, theta):
    """
    Compute P_t for rows in df_sub, using precomputed numeric features.

    theta = [z, h_half]   (we keep b0, bL, bT, bl fixed from prior GLM)
    """
    z, h_half = theta
    z = max(z, 1e-6)
    h_half = max(h_half, 1e-6)

    f_L_vals = df_sub["f_L"].values.astype(float)
    g_T_vals = df_sub["g_T"].values.astype(float)
    h_l_vals = df_sub["h_l"].values.astype(float)
    t_vals   = df_sub["t_hours"].values.astype(float)

    linear_prior = b0 + bL * f_L_vals + bT * g_T_vals + bl * h_l_vals
    p_prior = expit(linear_prior)

    alpha_p = z * p_prior
    beta_p  = z * (1 - p_prior)

    hist_seen    = df_sub["history_seen"].values.astype(float)
    hist_correct = df_sub["history_correct"].values.astype(float)

    alpha = alpha_p + hist_correct
    beta_ = beta_p + (hist_seen - hist_correct)

    P = alpha / (alpha + beta_)

    d = 2.0 ** (-t_vals / h_half)
    P_t = np.clip(d * P, EPS, 1 - EPS)
    return P_t


def nll_over_features(theta, user_set, max_rows=None):
    LL_sum = 0.0
    total_trials = 0.0     # <-- count trials, not rows

    for chunk in pd.read_csv(FEATURE_FILE, chunksize=CHUNK_SIZE):

        df_sub = chunk[chunk["user_id"].isin(user_set)].copy()
        if df_sub.empty:
            continue

        # Clean possible NaNs / infs
        df_sub = df_sub.replace([np.inf, -np.inf], np.nan)
        df_sub = df_sub.dropna(subset=[
            "f_L", "g_T", "h_l", "t_hours",
            "history_seen", "history_correct",
            "session_seen", "session_correct",
        ])
        if df_sub.empty:
            continue

        P_t = model_probs_on_features(df_sub, theta)
        n = df_sub["session_seen"].values.astype(float)
        m = df_sub["session_correct"].values.astype(float)

        # Drop any rows where P_t became nan/inf for any reason
        mask = np.isfinite(P_t)
        if not np.all(mask):
            P_t = P_t[mask]
            n = n[mask]
            m = m[mask]
            if len(P_t) == 0:
                continue

        # Binomial log-likelihood for each row
        logY = m * np.log(P_t) + (n - m) * np.log(1 - P_t)

        LL_sum += logY.sum()
        total_trials += n.sum()       # <-- accumulate trials, not rows

        # Stopping criterion for faster training
        if max_rows is not None and total_trials >= max_rows:
            break

    if total_trials == 0:
        raise RuntimeError("No trials found for given user_set in feature file.")

    # Return NLL PER TRIAL
    return -LL_sum / total_trials


# ================== FIT z AND HALF-LIFE h ==================

print("Fitting z and half-life h on training users (using subsample for speed)...")

theta0 = np.array([5.0, 24.0])   # [z, h_half]
bounds = [(1e-3, 100.0), (1e-3, 50000.0)]

def objective(theta):
    if USE_SUBSAMPLE_FOR_TRAINING:
        return nll_over_features(theta, train_users, max_rows=TRAIN_SUBSAMPLE_ROWS)
    else:
        return nll_over_features(theta, train_users, max_rows=None)

res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
               options={"maxiter": MAXITER})

if not res.success:
    print("WARNING: optimization did not converge:", res.message)

z_hat, h_hat = res.x
print(f"\nEstimated z  = {z_hat:.4f}")
print(f"Estimated h  = {h_hat:.4f} hours (half-life)")


# ================== FINAL TRAIN / TEST NLL ==================

print("\nComputing final NLL on full train and test sets (this may take a bit)...")

theta_hat = np.array([z_hat, h_hat])

nll_train = nll_over_features(theta_hat, train_users, max_rows=None)
nll_test  = nll_over_features(theta_hat, test_users,  max_rows=None)

print(f"\nFinal NLL (train): {nll_train:.6f}")
print(f"Final NLL (test) : {nll_test:.6f}")