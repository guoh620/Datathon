# -*- coding: utf-8 -*-
"""
Half-Life Regression model (Duolingo Settles & Meeder, 2016)
adapted to your data + user-based train/test split + binomial NLL.

Expected input format (columns, tab or comma separated):
p_recall, timestamp, delta, user_id, learning_language, ui_language,
lexeme_id, lexeme_string, history_seen, history_correct,
session_seen, session_correct
"""

import csv
import math
import os
import random
from collections import defaultdict, namedtuple
from sys import intern

# ====================== CONFIG ======================

INPUT_FILE = r"D:\OneDrive - KU Leuven\Datathon\Data_set_A\Data Set A_ Spaced Repetition\learning_traces.13m.csv"
SEP = ","          # change to "\t" if it is actually TSV
MAX_LINES = None   # e.g. 2_000_000 for dev, or None for all
TRAIN_USER_FRAC = 0.8
RANDOM_SEED = 42

# ====================== CONSTANTS ======================

# constraints on half-life
MIN_HALF_LIFE = 15.0 / (24 * 60)   # 15 minutes in days
MAX_HALF_LIFE = 274.0              # 9 months in days
LN2 = math.log(2.0)

# data instance object (a is unused but kept for compatibility)
Instance = namedtuple(
    "Instance",
    "p t fv h a lang right wrong ts uid lexeme".split()
)

# ====================== UTILS ======================

def pclip(p: float) -> float:
    """Bound min/max model predictions (helps with loss optimization & log)."""
    return min(max(p, 0.0001), 0.9999)


def hclip(h: float) -> float:
    """Bound min/max half-life."""
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def mae(l1, l2):
    return sum(abs(l1[i] - l2[i]) for i in range(len(l1))) / float(len(l1))


def mean(lst):
    return float(sum(lst)) / len(lst)


def spearmanr(l1, l2):
    m1 = mean(l1)
    m2 = mean(l2)
    num = 0.0
    d1 = 0.0
    d2 = 0.0
    for i in range(len(l1)):
        num += (l1[i] - m1) * (l2[i] - m2)
        d1 += (l1[i] - m1) ** 2
        d2 += (l2[i] - m2) ** 2
    return num / math.sqrt(d1 * d2)


# ====================== MODEL ======================

class SpacedRepetitionModel(object):
    """
    Spaced repetition model. Implements:
      - 'hlr'  (half-life regression; trainable)
      - 'lr'   (logistic regression; trainable)
      - 'leitner' (fixed)
      - 'pimsleur' (fixed)
    Here we will use only 'hlr'.
    """

    def __init__(
        self,
        method="hlr",
        omit_h_term=False,
        initial_weights=None,
        lrate=0.001,
        hlwt=0.01,
        l2wt=0.1,
        sigma=1.0,
    ):
        self.method = method
        self.omit_h_term = omit_h_term
        self.weights = defaultdict(float)
        if initial_weights is not None:
            self.weights.update(initial_weights)
        self.fcounts = defaultdict(int)
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma

    def halflife(self, inst, base=2.0):
        try:
            dp = sum(self.weights[k] * x_k for (k, x_k) in inst.fv)
            return hclip(base ** dp)
        except Exception:
            return MAX_HALF_LIFE

    def predict(self, inst, base=2.0):
        """Return (p, h) given an Instance."""
        if self.method == "hlr":
            h = self.halflife(inst, base)
            p = 2.0 ** (-inst.t / h)
            return pclip(p), h
        elif self.method == "leitner":
            try:
                h = hclip(2.0 ** inst.fv[0][1])
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2.0 ** (-inst.t / h)
            return pclip(p), h
        elif self.method == "pimsleur":
            try:
                h = hclip(2.0 ** (2.35 * inst.fv[0][1] - 16.46))
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2.0 ** (-inst.t / h)
            return pclip(p), h
        elif self.method == "lr":
            dp = sum(self.weights[k] * x_k for (k, x_k) in inst.fv)
            p = 1.0 / (1.0 + math.exp(-dp))
            return pclip(p), random.random()
        else:
            raise ValueError("Unknown method: %s" % self.method)

    def train_update(self, inst):
        if self.method == "hlr":
            base = 2.0
            p, h = self.predict(inst, base)
            # derivatives of squared loss w.r.t. weights
            dlp_dw = 2.0 * (p - inst.p) * (LN2 ** 2) * p * (inst.t / h)
            dlh_dw = 2.0 * (h - inst.h) * LN2 * h
            for (k, x_k) in inst.fv:
                rate = (1.0 / (1.0 + inst.p)) * self.lrate / math.sqrt(
                    1 + self.fcounts[k]
                )
                # sl(p) update
                self.weights[k] -= rate * dlp_dw * x_k
                # sl(h) update
                if not self.omit_h_term:
                    self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
                # L2 regularization
                self.weights[k] -= rate * self.l2wt * self.weights[k] / (
                    self.sigma ** 2
                )
                self.fcounts[k] += 1
        elif self.method in ("leitner", "pimsleur"):
            return
        elif self.method == "lr":
            p, _ = self.predict(inst)
            err = p - inst.p
            for (k, x_k) in inst.fv:
                rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                self.weights[k] -= rate * err * x_k
                self.weights[k] -= rate * self.l2wt * self.weights[k] / (
                    self.sigma ** 2
                )
                self.fcounts[k] += 1

    def train(self, trainset):
        if self.method in ("leitner", "pimsleur"):
            return
        random.shuffle(trainset)
        for inst in trainset:
            self.train_update(inst)

    def losses(self, inst):
        p, h = self.predict(inst)
        slp = (inst.p - p) ** 2
        slh = (inst.h - h) ** 2
        return slp, slh, p, h

    def eval(self, testset, prefix=""):
        results = {"p": [], "h": [], "pp": [], "hh": [], "slp": [], "slh": []}
        for inst in testset:
            slp, slh, p, h = self.losses(inst)
            results["p"].append(inst.p)
            results["h"].append(inst.h)
            results["pp"].append(p)
            results["hh"].append(h)
            results["slp"].append(slp)
            results["slh"].append(slh)
        mae_p = mae(results["p"], results["pp"])
        mae_h = mae(results["h"], results["hh"])
        cor_p = spearmanr(results["p"], results["pp"])
        cor_h = spearmanr(results["h"], results["hh"])
        total_slp = sum(results["slp"]) / len(results["slp"])
        total_slh = sum(results["slh"]) / len(results["slh"])
        total_l2 = sum(v * v for v in self.weights.values())
        total_loss = total_slp + self.hlwt * total_slh + self.l2wt * total_l2
        if prefix:
            print(f"{prefix}\t", end="")
        print(
            f"{total_loss:.1f} (p={total_slp:.1f}, h={self.hlwt*total_slh:.1f}, "
            f"l2={self.l2wt*total_l2:.1f})\t"
            f"mae(p)={mae_p:.3f}\tcor(p)={cor_p:.3f}\t"
            f"mae(h)={mae_h:.3f}\tcor(h)={cor_h:.3f}"
        )


# ====================== DATA LOADING (USER SPLIT) ======================

def read_data_user_split(input_file, max_lines=None):
    """
    Read the Duolingo learning traces and return (trainset, testset),
    where the split is done by user_id (TRAIN_USER_FRAC train users).
    """
    print("Reading data...")
    instances = []

    # open file in text mode
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=SEP)
        for i, row in enumerate(reader):
            if max_lines is not None and i >= max_lines:
                break

            try:
                p = pclip(float(row["p_recall"]))
            except Exception:
                # skip rows with bad p_recall
                continue

            # time delta in days (original code uses days)
            t = float(row["delta"]) / (60 * 60 * 24.0)

            # implied half-life from p and t
            try:
                h = hclip(-t / math.log(p, 2))
            except (ValueError, ZeroDivisionError):
                # fallback if something weird happens
                h = MIN_HALF_LIFE

            lang = f"{row['ui_language']}->{row['learning_language']}"
            lexeme_string = row["lexeme_string"]
            timestamp = int(row["timestamp"])
            user_id = row["user_id"]

            seen = int(row["history_seen"])
            right = int(row["history_correct"])
            wrong = seen - right

            right_this = int(row["session_correct"])
            wrong_this = int(row["session_seen"]) - right_this

            # feature vector
            fv = []
            # core HLR features
            fv.append((intern("right"), math.sqrt(1.0 + right)))
            fv.append((intern("wrong"), math.sqrt(1.0 + wrong)))
            # bias
            fv.append((intern("bias"), 1.0))
            # lexeme feature
            fv.append(
                (intern(f"{row['learning_language']}:{lexeme_string}"), 1.0)
            )

            # smoothed accuracy a = (right + 2)/(seen + 4) (not actually used here)
            if seen > 0:
                a = (right + 2.0) / (seen + 4.0)
            else:
                a = 0.5

            inst = Instance(
                p=p,
                t=t,
                fv=fv,
                h=h,
                a=a,
                lang=lang,
                right=right_this,
                wrong=wrong_this,
                ts=timestamp,
                uid=user_id,
                lexeme=lexeme_string,
            )
            instances.append(inst)

            if (i + 1) % 1_000_000 == 0:
                print(f"{i+1} rows read...")

    print("Done reading.")
    print(f"Total instances: {len(instances)}")

    # user-based split
    user_ids = sorted({inst.uid for inst in instances})
    print(f"Total unique users: {len(user_ids)}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(user_ids)

    n_train_users = int(TRAIN_USER_FRAC * len(user_ids))
    train_users = set(user_ids[:n_train_users])
    test_users = set(user_ids[n_train_users:])

    trainset = [inst for inst in instances if inst.uid in train_users]
    testset = [inst for inst in instances if inst.uid in test_users]

    print(f"|train| = {len(trainset)} instances")
    print(f"|test|  = {len(testset)} instances")

    return trainset, testset


# ====================== NLL COMPUTATION ======================

def compute_nll(instances, model):
    """
    Compute average negative log-likelihood per row using
    session_seen/session_correct (inst.right, inst.wrong) as binomial counts.
    """
    loglik = 0.0
    N = 0
    for inst in instances:
        n = inst.right + inst.wrong  # session_seen
        if n <= 0:
            continue
        m = inst.right               # session_correct

        p_pred, _ = model.predict(inst)
        p_pred = pclip(p_pred)

        loglik += m * math.log(p_pred) + (n - m) * math.log(1.0 - p_pred)
        N += 1

    if N == 0:
        return float("nan")
    return -loglik / N


# ====================== MAIN ======================

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    trainset, testset = read_data_user_split(INPUT_FILE, max_lines=MAX_LINES)

    print("Training HLR model...")
    model = SpacedRepetitionModel(method="hlr", omit_h_term=False)
    model.train(trainset)

    print("Eval on train set:")
    model.eval(trainset, prefix="train")
    print("Eval on test set:")
    model.eval(testset, prefix="test")

    print("Computing binomial NLL (per row)...")
    nll_train = compute_nll(trainset, model)
    nll_test = compute_nll(testset, model)

    print(f"NLL (train): {nll_train:.6f}")
    print(f"NLL (test) : {nll_test:.6f}")