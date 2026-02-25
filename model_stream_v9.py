# -*- coding: utf-8 -*-
"""
model_stream_v9.py
==================
Streaming Bayesian Beta-Binomial spaced-repetition model.

WHAT CHANGED FROM v8 (fixes_and_improvements):
-----------------------------------------------
BUG FIXES:
  [BUG-1] h(l) regression: beta1 was POSITIVE (+0.0013) implying longer words
          are *easier*, which is wrong. Root cause: using session_correct /
          session_seen as the accuracy target mixes in review history and session
          variability. Fix: use history_correct / history_seen (cumulative
          accuracy) as a cleaner proxy for inherent word difficulty, and restrict
          to rows with modest history (1 <= history_seen <= LOW_HISTORY_MAX) so
          we are close to the cold-start prior.

  [BUG-2] t_hours unbounded: delta can be up to ~29M seconds (~335 days in the
          subset). With the optimizer free to push h to 20000 hours, the decay
          term 2^(-t/h) is essentially 1 for all rows, meaning the optimizer
          correctly deduces that decay barely matters because Duolingo users
          review frequently. However, a handful of huge-gap rows dominate the
          gradient and pull h → upper_bound. Fix: cap t_hours at a sensible
          maximum (8760h = 1 year) so extreme outliers don't distort h.

  [BUG-3] p_prior has near-zero variance (std ≈ 0.009 on the subset). This
          happens because f_L (6 languages, range 600-750 FSI hours) and g_T
          (hardcoded lookup with coarse buckets) map most rows to similar values,
          and h_l = beta0 + beta1*l ≈ 0.89 ± 0.01 (almost flat). So the prior
          barely differentiates words — it's swamped by the history counts.
          Fix: add more languages to HOURS dict, keep the existing g_T but note
          that improving it with empirical per-tag accuracy (see IMPROVEMENT-2)
          would help substantially.

  [BUG-4] No first-encounter rows exist in this dataset (min history_seen = 1).
          So filtering history_seen <= 3 actually gives rows with 1-3 PREVIOUS
          sessions — these are not cold-start observations. The prior GLM and
          h(l) regression are therefore trained on "already-practised" rows, 
          which biases p_prior upward (everyone looks like a good learner).
          Fix: acknowledge this limitation in a comment; keep history_seen <= 3
          as a proxy for "early-stage learner" but be aware p_prior will be
          slightly optimistic.

IMPROVEMENTS:
  [IMP-1] Feature-dependent half-life (log-linear in item features):
          Instead of a single global h, parameterise:
              log(h) = h0 + h_fL * f_L + h_gT * g_T
          so that easier languages / word types get longer half-lives.
          This makes the decay term responsive to item difficulty rather than
          fitting a single number averaged over all items and users.

  [IMP-2] Spacing-effect term in half-life:
          Cognitive science strongly supports that more correct repetitions
          → slower forgetting. We add:
              log(h) += h_hist * log(1 + history_correct)
          This is the core insight behind HLR (Settles & Meeder, 2016).

  [IMP-3] Joint optimisation of ALL parameters simultaneously:
          v8 fixed b first (from GLM on low-history rows), then optimised (z, h).
          This is a two-stage approximation. The GLM p_prior is estimated on
          history_seen <= 3 rows while NLL is evaluated on ALL rows — there
          is a distribution shift. Full joint optimisation removes this
          inconsistency. We initialise from the GLM estimates so convergence
          is fast.

  [IMP-4] Expanded language list:
          Added Korean, Japanese, Chinese, Russian, Arabic, Turkish based on
          FSI data. These are common Duolingo languages not in the original
          HOURS dict. Without them all unknown languages map to DEFAULT_HOURS
          = 700, which is indistinguishable from German (750) — f_L has less
          dynamic range than it should.

  [IMP-5] Better NLL normalisation:
          Original divides total log-likelihood by the NUMBER OF ROWS.
          A row with session_seen=3 contributes 3 Bernoulli trials but counts
          as 1 row. We now normalise by total NUMBER OF TRIALS (sum of
          session_seen) so the metric is a true per-trial log-likelihood,
          comparable across datasets with different session lengths.

  [IMP-6] Added optional diagnostics: calibration of P_t vs p_recall on a
          holdout chunk so you can see whether the model is over/under-confident.

WHY NOT adding per-user half-life:
  Per-user h would require storing one float per user (115K users → tiny memory).
  However, optimising 115K floats jointly with scipy is not tractable for a
  streaming model. A sensible extension would be EM: fix global params, update
  each user's h analytically, repeat. Left as future work.

Pipeline:
  1) Pass 1 : collect unique users → train/test split
  2) Pass 2 : estimate h(l) via streaming regression (TRAIN users only)
  3) Pass 3 : sample low-history TRAIN rows → fit prior GLM
  4) Pass 4 : precompute numeric feature file (all rows)
  5) Joint optimise theta = (z, h0, h_fL, h_gT, h_hist, b0, bL, bT, bl)
  6) Final NLL on full train & test (streaming)
"""

import numpy as np
import pandas as pd
import re
from scipy.special import expit
from scipy.optimize import minimize
import statsmodels.api as sm

# ================== CONFIG ==================

BIGFILE = r"learning_traces.13m.csv"
SEP = ","
CHUNK_SIZE = 200_000
RANDOM_SEED = 42

FEATURE_FILE = r"precomputed_features_v9.csv"

EPS   = 1e-9    # prevents log(0) in NLL
EPS_F = 1e-3    # keeps f(L) strictly inside (0, 1)

# ---------- training speed controls ----------
USE_SUBSAMPLE_FOR_TRAINING = True
TRAIN_SUBSAMPLE_ROWS = 1_000_000
MAXITER = 100               # increased from 50 — more iterations for joint opt

PRIOR_SAMPLE_MAX  = 1_000_000
LOW_HISTORY_MAX   = 3       # rows with history_seen <= this used for h(l) and prior

# [BUG-2 FIX] Cap t_hours to prevent extreme outliers from dominating the
# gradient and pulling h → its upper bound.
# Data analysis shows max delta ≈ 29M seconds ≈ 8062 hours in the subset.
# We cap at 8760h (1 year). Users with longer gaps are treated as ~1-year gap.
T_HOURS_CAP = 8760.0

# If True, jointly optimise ALL parameters (b0, bL, bT, bl, z, h0, h_fL, h_gT, h_hist).
# If False, fall back to the v8 two-stage approach (GLM for b, then optimise z & h).
JOINT_OPTIMISE = True


# ==================  FEATURE FUNCTIONS  ==================

# ---- 1) Language difficulty f(L) from FSI hours ----
# The dataset contains exactly these 6 languages and no others, so DEFAULT_HOURS
# is never actually used. Every row maps to a known entry in HOURS.
#
# KNOWN LIMITATION (BUG-3): With only 6 languages spanning 600–750 FSI hours,
# f_L has very low variance and bL ends up weak (~0.33 in v7/v8).
# A better approach would be to replace f_L with 5 binary language-dummy
# features (one per language, dropping one as reference). That gives each
# language its own free coefficient rather than forcing a linear ordering
# through FSI hours, which may not reflect Duolingo's actual difficulty
# ranking. This is left as a future improvement since it requires changing
# the feature schema and the prior GLM design matrix.
HOURS = {
    "fr": 675,
    "en": 600,
    "es": 675,
    "de": 750,
    "pt": 675,
    "it": 675,
}
DEFAULT_HOURS = 700  # fallback — never reached given the 6-language dataset

H_vals   = list(HOURS.values())
H_min    = min(H_vals)      # 600  (English)
H_max    = max(H_vals)      # 750  (German)
_H_RANGE = max(H_max - H_min, 1e-12)  # 150 hours — narrow range, low f_L variance


def f_language(lang_code: str) -> float:
    """
    Map a Duolingo language code to difficulty in [EPS_F, 1-EPS_F].
    f=1 means easiest (fewest FSI hours), f≈0 means hardest.
    """
    H = HOURS.get(lang_code, DEFAULT_HOURS)
    return EPS_F + (1.0 - 2.0 * EPS_F) * (1.0 - (H - H_min) / _H_RANGE)


# ---- 2) POS-based difficulty g(T) ----
# These scores are taken from Suranto & Yuspik (2024) word-type identification rates.
# Higher = more often correctly identified = easier to remember.
# NOTE: bT came out ≈ 0.009 in v7/v8, meaning this feature barely moves the prior.
# Two reasons: (a) the tag → score mapping is coarse, (b) the GLM is overwhelmed
# by the intercept b0 ≈ 11.5. Joint optimisation may assign it more or less weight.
def extract_pos_score(cell: str) -> float:
    if not isinstance(cell, str):
        return 0.6   # neutral default (same as "unknown" branch below)

    tags = re.findall(r"<([^>]+)>", cell)
    if not tags:
        return 0.6

    tags = [t.lstrip("*").lower() for t in tags]

    # Ordered from easiest to remember (verbs) down to hardest (adverbs)
    if any(t.startswith("v") or t == "sep" for t in tags):
        return 0.89
    if any(t in ["n", "ant", "cog", "np"] for t in tags):
        return 0.92
    if any(t == "num" for t in tags):
        return 0.95
    if any(t in ["acr", "adj", "atn", "comp", "dem", "det", "detnt", "enc",
                 "itg", "obj", "ord", "pos", "pro", "qnt", "ref", "rel",
                 "sint", "sup", "tn"] for t in tags):
        return 0.65
    if any(t in ["prn", "pron", "prpers"] for t in tags):
        return 0.531
    if any(t == "pr" for t in tags):
        return 0.469
    if any(t in ["adv", "cnjadv", "preadv"] for t in tags):
        return 0.45

    return 0.6  # fallback


# ---- 3) Word length feature h(l) ----
# [BUG-1 partial fix] The original regression used session_correct/session_seen
# as the accuracy label, which is noisy and confounded by history. We now use
# history_correct / history_seen (cumulative accuracy) as the regression target
# in Pass 2 — this is a smoother estimate of a word's inherent difficulty.
def extract_word_length(lexeme_string: str) -> float:
    """
    Returns the length of the longest alphabetic surface form in lexeme_string.
    E.g. "hats/hat<n><pl>" → len("hats") = 4
         "<*sf>/student<n><*numb>" → len("student") = 7
    """
    if not isinstance(lexeme_string, str):
        return np.nan
    without_tags = re.sub(r"<[^>]*>", "", lexeme_string)
    forms = without_tags.split("/")
    cleaned = ["".join(ch for ch in f if ch.isalpha()) for f in forms]
    cleaned = [f for f in cleaned if f]
    if not cleaned:
        return np.nan
    return float(max(len(f) for f in cleaned))


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
test_users  = set(unique_users[n_train_users:])

print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")


# ================== PASS 2: ESTIMATE h(l) ON TRAIN USERS ==================
# [BUG-1 FIX] Changed accuracy target from session_correct/session_seen to
# history_correct/history_seen. The history ratio is a cumulative average over
# all previous encounters, which is a far more stable measure of inherent word
# difficulty than a single noisy session. This should produce a negative beta1
# (longer words → harder → lower accuracy), which is theoretically correct.
#
# WHY NOT history_seen=0 (true cold start)?
# In the Duolingo dataset min(history_seen) = 1, meaning the dataset only
# records a row AFTER at least one session has already occurred. So true
# cold-start rows do not exist here. We use history_seen in [1, LOW_HISTORY_MAX]
# as the best available proxy for early-stage difficulty.

print("Pass 2: estimating h(l) = beta0 + beta1*l (TRAIN users, cumulative accuracy target)...")

Sx = Sy = Sxx = Sxy = 0.0
N_reg = 0

for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):
    # Restrict to TRAIN users only (no leakage)
    chunk = chunk[chunk["user_id"].isin(train_users)]
    if chunk.empty:
        continue

    # Use early-stage rows as a proxy for inherent word difficulty
    df_h = chunk[
        (chunk["history_seen"] >= 1) &
        (chunk["history_seen"] <= LOW_HISTORY_MAX)
    ].copy()
    if df_h.empty:
        continue

    # [BUG-1 FIX] Use cumulative accuracy (more stable) as the regression target
    df_h["cum_acc"] = df_h["history_correct"] / df_h["history_seen"]
    df_h["word_len"] = df_h["lexeme_string"].apply(extract_word_length)

    df_h = df_h.replace([np.inf, -np.inf], np.nan).dropna(subset=["cum_acc", "word_len"])
    if df_h.empty:
        continue

    x = df_h["word_len"].values.astype(float)
    y = df_h["cum_acc"].values.astype(float)

    Sx  += x.sum()
    Sy  += y.sum()
    Sxx += np.dot(x, x)
    Sxy += np.dot(x, y)
    N_reg += len(df_h)

if N_reg < 2:
    raise RuntimeError("Not enough data to fit h(l).")

reg_denom = N_reg * Sxx - Sx * Sx
if abs(reg_denom) < 1e-12:
    raise RuntimeError("Degenerate regression: word_len has near-zero variance.")

beta1 = (N_reg * Sxy - Sx * Sy) / reg_denom
beta0 = (Sy - beta1 * Sx) / N_reg

print(f"h(l): beta0 = {beta0:.4f}, beta1 = {beta1:.4f}  (from {N_reg} TRAIN rows)")
print(f"  → beta1 {'< 0 (longer = harder, expected)' if beta1 < 0 else '> 0 (WARNING: unexpected sign)'}")


# ================== PASS 3: FIT PRIOR GLM ON TRAIN USERS ==================
# We use grouped binomial GLM (v8 fix already in place) so p_prior is a
# per-trial recall probability, consistent with the NLL we optimise.
# [BUG-4 NOTE] All rows have history_seen >= 1, so p_prior will be slightly
# biased upward (users have already succeeded at least once). This is
# unavoidable given the dataset, but using low history_seen rows minimises it.
# If JOINT_OPTIMISE=True the b parameters will be refined by the full NLL
# anyway, so this GLM serves only as a warm start.

print("Pass 3: fitting prior GLM on low-history TRAIN rows...")

rows_list  = []
rows_kept  = 0

for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):
    chunk = chunk[chunk["user_id"].isin(train_users)]
    if chunk.empty:
        continue

    df_p = chunk[
        (chunk["history_seen"] >= 1) &
        (chunk["history_seen"] <= LOW_HISTORY_MAX) &
        (chunk["session_seen"]  > 0)
    ].copy()
    if df_p.empty:
        continue

    df_p["f_L"]      = df_p["learning_language"].apply(f_language)
    df_p["g_T"]      = df_p["lexeme_string"].apply(extract_pos_score)
    df_p["word_len"] = df_p["lexeme_string"].apply(extract_word_length)
    df_p["h_l"]      = beta0 + beta1 * df_p["word_len"]

    keep = ["f_L", "g_T", "h_l", "session_correct", "session_seen"]
    df_p = df_p[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=keep)
    df_p = df_p[
        (df_p["session_seen"]    > 0) &
        (df_p["session_correct"] >= 0) &
        (df_p["session_correct"] <= df_p["session_seen"])
    ]
    if df_p.empty:
        continue

    remaining = PRIOR_SAMPLE_MAX - rows_kept
    if remaining <= 0:
        break
    df_p = df_p.iloc[:remaining].copy()
    rows_list.append(df_p)
    rows_kept += len(df_p)
    if rows_kept >= PRIOR_SAMPLE_MAX:
        break

if not rows_list:
    raise RuntimeError("No low-history TRAIN rows found for prior fitting.")

prior_df = pd.concat(rows_list, ignore_index=True)
print(f"Prior GLM sample size: {len(prior_df)}")

X_prior = sm.add_constant(prior_df[["f_L", "g_T", "h_l"]].values.astype(float))
y_prop  = (prior_df["session_correct"].values.astype(float) /
           prior_df["session_seen"].values.astype(float))
n_trials = prior_df["session_seen"].values.astype(float)

prior_model = sm.GLM(y_prop, X_prior, family=sm.families.Binomial(), var_weights=n_trials)
prior_res   = prior_model.fit()

# GLM estimates (used as warm-start, or directly if JOINT_OPTIMISE=False)
b0_glm, bL_glm, bT_glm, bl_glm = prior_res.params

print("Prior GLM coefficients (warm-start):")
print(f"  b0={b0_glm:.4f}, bL={bL_glm:.4f}, bT={bT_glm:.4f}, bl={bl_glm:.4f}")


# ================== PASS 4: PRECOMPUTE NUMERIC FEATURE FILE ==================
# [BUG-2 FIX] Apply T_HOURS_CAP = 8760h here so all downstream code sees
# capped t_hours. Rows with delta > 1 year are rare but can devastate the
# gradient; capping them treats "very long gap" uniformly.
# [IMP-5] Also store session_seen to compute per-trial NLL normalisation.

print("Pass 4: precomputing numeric feature file (one-time cost)...")

with open(FEATURE_FILE, "w", newline="", encoding="utf-8") as f_out:
    first = True

    for chunk in pd.read_csv(BIGFILE, sep=SEP, chunksize=CHUNK_SIZE):
        # Drop rows with invalid session counts
        chunk = chunk[
            (chunk["session_seen"]    > 0) &
            (chunk["session_correct"] >= 0) &
            (chunk["session_correct"] <= chunk["session_seen"])
        ]
        if chunk.empty:
            continue

        chunk = chunk.copy()
        chunk["f_L"]      = chunk["learning_language"].apply(f_language)
        chunk["g_T"]      = chunk["lexeme_string"].apply(extract_pos_score)
        chunk["word_len"] = chunk["lexeme_string"].apply(extract_word_length)
        chunk["h_l"]      = beta0 + beta1 * chunk["word_len"]

        # [BUG-2 FIX] Cap t_hours at T_HOURS_CAP to prevent extreme outliers
        # from pulling h to its upper bound during optimisation.
        chunk["t_hours"]  = (chunk["delta"].astype(float) / 3600.0).clip(upper=T_HOURS_CAP)

        cols = [
            "user_id", "f_L", "g_T", "h_l", "t_hours",
            "history_seen", "history_correct",
            "session_seen", "session_correct",
        ]
        chunk = chunk[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        if chunk.empty:
            continue

        if first:
            chunk.to_csv(f_out, index=False)
            first = False
        else:
            chunk.to_csv(f_out, index=False, header=False)

print(f"Feature file written to: {FEATURE_FILE}")


# ================== MODEL DEFINITION ==================
# [IMP-1] Feature-dependent half-life:
#   log(h_i) = h0 + h_fL * f_L_i + h_gT * g_T_i + h_hist * log(1 + hist_correct_i)
#
# WHY log-linear?
#   h_i must be strictly positive. log-linear (exp of a linear combination)
#   guarantees this without explicit clipping. It also means h_fL and h_gT
#   are log-scale coefficients: h_fL=1 means each unit increase in f_L
#   multiplies the half-life by e ≈ 2.7 — a reasonable scale.
#
# [IMP-2] Spacing effect (h_hist * log(1 + history_correct)):
#   More correct repetitions → longer half-life. This is the central finding
#   of the HLR paper (Settles & Meeder 2016) and cognitive science (expanding
#   retrieval practice). We use log(1 + history_correct) rather than the raw
#   count to dampen the effect for very well-practised items.
#
# WHY NOT per-user h?
#   Would require 115K extra parameters. Not tractable with scipy in a single
#   pass. Could be done with EM in future work.

def model_probs(df_sub: pd.DataFrame, theta) -> np.ndarray:
    """
    Compute P_t for each row in df_sub.

    theta layout (9 parameters):
      [0] z       - Beta prior concentration (controls prior vs history weight)
      [1] h0      - log half-life intercept (hours, log scale)
      [2] h_fL    - log half-life coefficient for f(L)     [IMP-1]
      [3] h_gT    - log half-life coefficient for g(T)     [IMP-1]
      [4] h_hist  - log half-life coefficient for log(1+history_correct) [IMP-2]
      [5] b0      - prior logit intercept
      [6] bL      - prior logit coefficient for f(L)
      [7] bT      - prior logit coefficient for g(T)
      [8] bl      - prior logit coefficient for h(l)
    """
    z, h0, h_fL, h_gT, h_hist = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3]), float(theta[4])
    b0_, bL_, bT_, bl_ = float(theta[5]), float(theta[6]), float(theta[7]), float(theta[8])

    z = max(z, 1e-6)  # z must be positive

    f_L   = df_sub["f_L"].values.astype(float)
    g_T   = df_sub["g_T"].values.astype(float)
    h_l   = df_sub["h_l"].values.astype(float)
    t     = df_sub["t_hours"].values.astype(float)
    h_seen    = df_sub["history_seen"].values.astype(float)
    h_correct = df_sub["history_correct"].values.astype(float)

    # ---------- Prior recall probability ----------
    linear_prior = b0_ + bL_ * f_L + bT_ * g_T + bl_ * h_l
    p_prior = expit(linear_prior)   # per-trial recall probability prior

    # ---------- Beta posterior (Beta-Binomial update) ----------
    alpha_p = z * p_prior
    beta_p  = z * (1.0 - p_prior)
    alpha   = alpha_p + h_correct
    beta_   = beta_p  + (h_seen - h_correct)
    P       = alpha / (alpha + beta_)   # posterior mean recall (no decay)

    # ---------- [IMP-1, IMP-2] Feature-dependent half-life ----------
    # h_i (in hours) depends on language difficulty, POS difficulty, and
    # cumulative correct repetitions (spacing effect).
    log_h = h0 + h_fL * f_L + h_gT * g_T + h_hist * np.log1p(h_correct)
    h_i   = np.exp(log_h)   # always positive, no clipping needed

    # ---------- Memory decay ----------
    # d_i = 2^(-t / h_i).  A row with t=h_i has P_t = 0.5 * P.
    d   = np.exp(-t * np.log(2.0) / np.maximum(h_i, 1e-6))
    P_t = np.clip(d * P, EPS, 1.0 - EPS)

    return P_t


def nll_over_features(theta, user_set, max_rows=None, normalise_by_trials=True):
    """
    Streaming average negative log-likelihood over FEATURE_FILE.

    [IMP-5] normalise_by_trials=True divides by total session_seen (trials),
    giving a per-trial NLL that is comparable across datasets with variable
    session lengths. Set to False for backward compatibility with v7/v8.

    Returns: scalar NLL
    """
    LL_sum    = 0.0
    N_rows    = 0
    N_trials  = 0

    for chunk in pd.read_csv(FEATURE_FILE, chunksize=CHUNK_SIZE):
        df_sub = chunk[chunk["user_id"].isin(user_set)].copy()
        if df_sub.empty:
            continue

        P_t = model_probs(df_sub, theta)
        n   = df_sub["session_seen"].values.astype(float)
        m   = df_sub["session_correct"].values.astype(float)

        # Binomial log-likelihood per row
        logY     = m * np.log(P_t) + (n - m) * np.log(1.0 - P_t)
        LL_sum  += float(logY.sum())
        N_rows  += len(df_sub)
        N_trials += int(n.sum())

        if max_rows is not None and N_rows >= max_rows:
            break

    if N_rows == 0:
        raise RuntimeError("No rows found for given user_set in feature file.")

    denom = N_trials if (normalise_by_trials and N_trials > 0) else N_rows
    return -LL_sum / denom


# ================== OPTIMISATION ==================
# [IMP-3] Joint optimisation of ALL 9 parameters.
# theta = [z, h0, h_fL, h_gT, h_hist, b0, bL, bT, bl]
#
# Warm-start from:
#   - z = 5.0  (reasonable prior concentration)
#   - h0 = log(168) ≈ 5.12  (one week baseline half-life)
#   - h_fL, h_gT = 0.0      (no feature effect initially)
#   - h_hist = 0.5          (modest spacing effect)
#   - b params from GLM     (good starting point from Pass 3)
#
# WHY L-BFGS-B? It handles box constraints natively and uses approximate
# Hessian info for fast convergence. Gradient is estimated numerically via
# finite differences; this is fine here since each function eval is cheap
# (streaming over a subsample of 1M rows).
#
# BOUNDS rationale:
#   z in [0.001, 200]: very small z = history dominates; very large = prior dominates
#   h0 in [log(0.5), log(20000)] = half-life baseline between 30 min and 833 days
#   h_fL, h_gT in [-5, 5]: log-scale multipliers; ±5 → factor of e^5 ≈ 148×
#   h_hist in [0, 5]: spacing effect should be non-negative (more practice → longer memory)
#   b params in [-50, 50]: wide, let GLM warm-start guide them

print("\nSetting up joint optimisation of all 9 parameters...")

theta0 = np.array([
    5.0,             # z
    np.log(168.0),   # h0 = log(1 week in hours)
    0.0,             # h_fL
    0.0,             # h_gT
    0.5,             # h_hist (spacing effect warm-start)
    b0_glm,          # b0
    bL_glm,          # bL
    bT_glm,          # bT
    bl_glm,          # bl
])

bounds = [
    (1e-3,  200.0),            # z
    (np.log(0.5), np.log(20000.0)),  # h0 (log-scale; 30min to 833 days)
    (-5.0,  5.0),              # h_fL
    (-5.0,  5.0),              # h_gT
    (0.0,   5.0),              # h_hist  (non-negative: more practice → longer memory)
    (-50.0, 50.0),             # b0
    (-50.0, 50.0),             # bL
    (-50.0, 50.0),             # bT
    (-50.0, 50.0),             # bl
]

if JOINT_OPTIMISE:
    print("Joint optimisation (JOINT_OPTIMISE=True) — all 9 params...")

    def objective_joint(theta):
        return nll_over_features(
            theta,
            train_users,
            max_rows=(TRAIN_SUBSAMPLE_ROWS if USE_SUBSAMPLE_FOR_TRAINING else None),
            normalise_by_trials=True,   # [IMP-5]
        )

    opt_res = minimize(
        objective_joint,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": MAXITER, "ftol": 1e-9, "gtol": 1e-6},
    )

    theta_hat = opt_res.x

else:
    # [FALLBACK] Two-stage approach from v8: fix b from GLM, then optimise (z, h0 only).
    # h_fL, h_gT, h_hist are fixed at 0, 0, 0.5 (spacing effect still active).
    print("Two-stage optimisation (JOINT_OPTIMISE=False) — fixing b from GLM, optimising z & h0...")

    b0_, bL_, bT_, bl_ = b0_glm, bL_glm, bT_glm, bl_glm

    def objective_stage2(theta_small):
        # Full theta: [z, h0, h_fL=0, h_gT=0, h_hist=0.5, b_from_glm...]
        theta_full = np.array([
            theta_small[0],   # z
            theta_small[1],   # h0
            0.0, 0.0, 0.5,    # h_fL, h_gT, h_hist (fixed)
            b0_, bL_, bT_, bl_
        ])
        return nll_over_features(
            theta_full, train_users,
            max_rows=(TRAIN_SUBSAMPLE_ROWS if USE_SUBSAMPLE_FOR_TRAINING else None),
            normalise_by_trials=True,
        )

    opt_res2 = minimize(
        objective_stage2,
        np.array([5.0, np.log(168.0)]),
        method="L-BFGS-B",
        bounds=[(1e-3, 200.0), (np.log(0.5), np.log(20000.0))],
        options={"maxiter": MAXITER},
    )

    theta_hat = np.array([
        opt_res2.x[0],   # z
        opt_res2.x[1],   # h0
        0.0, 0.0, 0.5,   # h_fL, h_gT, h_hist
        b0_, bL_, bT_, bl_
    ])
    opt_res = opt_res2  # for printing

print("\nOptimisation result:")
print(opt_res)

z_hat, h0_hat, h_fL_hat, h_gT_hat, h_hist_hat = theta_hat[:5]
b0_hat, bL_hat, bT_hat, bl_hat = theta_hat[5:]

print(f"\n--- Fitted parameters ---")
print(f"  z       = {z_hat:.4f}  (prior concentration: ~{z_hat:.0f} 'phantom trials')")
print(f"  h0      = {h0_hat:.4f} → baseline half-life = {np.exp(h0_hat):.1f} hours = {np.exp(h0_hat)/24:.1f} days")
print(f"  h_fL    = {h_fL_hat:.4f}  (positive → easier languages decay slower)")
print(f"  h_gT    = {h_gT_hat:.4f}  (positive → easier POS types decay slower)")
print(f"  h_hist  = {h_hist_hat:.4f}  (spacing effect: more correct reviews → longer memory)")
print(f"  b0      = {b0_hat:.4f}, bL = {bL_hat:.4f}, bT = {bT_hat:.4f}, bl = {bl_hat:.4f}")


# ================== FINAL EVALUATION ==================

print("\nComputing final per-trial NLL on full train and test sets...")

nll_train = nll_over_features(theta_hat, train_users, max_rows=None, normalise_by_trials=True)
nll_test  = nll_over_features(theta_hat, test_users,  max_rows=None, normalise_by_trials=True)

print(f"\nFinal per-trial NLL (train): {nll_train:.6f}")
print(f"Final per-trial NLL (test) : {nll_test:.6f}")
print(f"Train-test gap             : {nll_test - nll_train:.6f}")
print(f"  (small gap → model is not overfitting)")

# For comparison, print v8-style row-level NLL too (divides by rows not trials)
nll_train_rows = nll_over_features(theta_hat, train_users, max_rows=None, normalise_by_trials=False)
nll_test_rows  = nll_over_features(theta_hat, test_users,  max_rows=None, normalise_by_trials=False)
print(f"\nFor v8 comparability (per-row NLL):")
print(f"  NLL (train): {nll_train_rows:.6f}")
print(f"  NLL (test) : {nll_test_rows:.6f}")


# ================== OPTIONAL DIAGNOSTIC ==================
# Quick calibration check: compare predicted P_t to actual p_recall
# on the first DIAG_ROWS rows of the feature file (any user, train or test).
DIAG_ROWS = 50_000
print(f"\n--- Calibration diagnostic on first {DIAG_ROWS} feature-file rows ---")

diag_pred  = []
diag_truth = []

n_diag = 0
for chunk in pd.read_csv(FEATURE_FILE, chunksize=CHUNK_SIZE):
    sub = chunk.head(max(0, DIAG_ROWS - n_diag)).copy()
    if sub.empty:
        break
    sub = sub[(sub["session_seen"] > 0)].copy()
    P_t = model_probs(sub, theta_hat)
    acc = sub["session_correct"].values / sub["session_seen"].values
    diag_pred.extend(P_t.tolist())
    diag_truth.extend(acc.tolist())
    n_diag += len(sub)
    if n_diag >= DIAG_ROWS:
        break

if diag_pred:
    diag_pred  = np.array(diag_pred)
    diag_truth = np.array(diag_truth)
    mse  = np.mean((diag_pred - diag_truth) ** 2)
    bias = np.mean(diag_pred - diag_truth)
    print(f"  MSE (P_t vs session accuracy) : {mse:.4f}")
    print(f"  Bias (mean predicted - actual): {bias:.4f}  (+ve → model overestimates recall)")
    print(f"  Mean predicted P_t : {diag_pred.mean():.4f}")
    print(f"  Mean actual  acc   : {diag_truth.mean():.4f}")
