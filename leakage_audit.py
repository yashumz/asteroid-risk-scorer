"""
Leakage audit for asteroid-risk-scorer.

Uses ONLY numpy + pandas. No scipy, no scikit-learn.
AUC and a small greedy decision tree are implemented directly below.

    python leakage_audit.py

Reads asteroids.db. Changes nothing.
"""

import sqlite3
import numpy as np
import pandas as pd

AU_KM = 149_597_870.7
H_THRESHOLD = 22.0

FEATURES = [
    "diameter_km", "velocity_kps", "miss_dist_km", "magnitude_h",
    "kinetic_energy_proxy", "threat_score", "is_fast", "size_category",
    "log_miss_dist", "log_diameter",
]


# ---------------------------------------------------------------- metrics
def auc(y, score):
    """Area under ROC, via the rank-sum identity. No scipy needed."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 2 * p * (1 - p)


# ------------------------------------------------------------------- tree
def candidates(col, k=40):
    q = np.unique(np.quantile(col, np.linspace(0.02, 0.98, k)))
    return q


def best_split(X, y, feats):
    best = None
    parent = gini(y) * len(y)
    for fi, f in enumerate(feats):
        col = X[:, fi]
        for t in candidates(col):
            left = col <= t
            nl, nr = left.sum(), (~left).sum()
            if nl == 0 or nr == 0:
                continue
            score = gini(y[left]) * nl + gini(y[~left]) * nr
            if best is None or score < best[0]:
                best = (score, fi, f, t)
    if best is None or best[0] >= parent:
        return None
    return best


def fit_tree(X, y, feats, depth):
    if depth == 0 or len(np.unique(y)) == 1 or len(y) < 4:
        return {"leaf": y.mean() if len(y) else 0.0}
    s = best_split(X, y, feats)
    if s is None:
        return {"leaf": y.mean()}
    _, fi, fname, t = s
    left = X[:, fi] <= t
    return {
        "fi": fi, "name": fname, "t": t,
        "L": fit_tree(X[left], y[left], feats, depth - 1),
        "R": fit_tree(X[~left], y[~left], feats, depth - 1),
    }


def predict(node, X):
    if "leaf" in node:
        return np.full(len(X), node["leaf"])
    out = np.empty(len(X))
    left = X[:, node["fi"]] <= node["t"]
    if left.any():
        out[left] = predict(node["L"], X[left])
    if (~left).any():
        out[~left] = predict(node["R"], X[~left])
    return out


def show_tree(node, indent="      "):
    if "leaf" in node:
        print(f"{indent}--> hazardous fraction {node['leaf']:.2f}")
        return
    print(f"{indent}if {node['name']} <= {node['t']:.4g}:")
    show_tree(node["L"], indent + "   ")
    print(f"{indent}else:")
    show_tree(node["R"], indent + "   ")


def stratified_split(y, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        test[idx[: max(1, int(round(len(idx) * test_frac)))]] = True
    return ~test, test


def tree_auc(X, y, feats, depth, tr, te):
    t = fit_tree(X[tr], y[tr], feats, depth)
    return auc(y[te], predict(t, X[te])), t


# ------------------------------------------------------------------- data
def load_and_engineer():
    conn = sqlite3.connect("asteroids.db")
    df = pd.read_sql("SELECT * FROM neo_raw", conn)
    conn.close()
    df["is_hazardous"] = df["is_hazardous"].astype(bool)
    df["kinetic_energy_proxy"] = 0.5 * (df["diameter_km"] ** 3) * (df["velocity_kps"] ** 2)
    df["threat_score"] = df["diameter_km"] / (df["miss_dist_km"] / 1e6)
    df["is_fast"] = (df["velocity_kps"] > 20).astype(int)
    df["size_category"] = pd.cut(
        df["diameter_km"], bins=[0, 0.3, 1.0, float("inf")], labels=[0, 1, 2]
    ).astype(int)
    df["log_miss_dist"] = np.log1p(df["miss_dist_km"])
    df["log_diameter"] = np.log1p(df["diameter_km"])
    return df


# ------------------------------------------------------------------- main
def main():
    df = load_and_engineer()
    y = df["is_hazardous"].astype(int).values
    Xdf = df[FEATURES].astype(float)
    X = Xdf.values

    print("=" * 70)
    print(f"rows: {len(df)}   hazardous: {y.sum()} ({y.mean():.1%})")
    print("=" * 70)

    findings = []

    print("\n[1] Is magnitude_h alone a one-sided perfect rule?")
    print("    NASA: an asteroid with H > 22.0 can NEVER be a PHA.")
    above = df[df["magnitude_h"] > H_THRESHOLD]
    below = df[df["magnitude_h"] <= H_THRESHOLD]
    if len(above):
        viol = int(above["is_hazardous"].sum())
        print(f"    rows with H > 22  : {len(above)}  ({len(above)/len(df):.1%} of data)")
        print(f"    of those, hazardous: {viol}")
        if viol == 0:
            print(f"    >>> LEAK. magnitude_h alone settles all {len(above)} of them.")
            findings.append(
                f"magnitude_h alone settles {len(above)/len(df):.0%} of rows with certainty")
    if len(below):
        print(f"    rows with H <= 22 : {len(below)}  hazardous rate {below['is_hazardous'].mean():.1%}")

    print("\n[2] Full NASA rule, using miss_distance as a MOID stand-in")
    rule = ((df["magnitude_h"] <= H_THRESHOLD) &
            (df["miss_dist_km"] <= 0.05 * AU_KM)).astype(int).values
    agree = (rule == y).mean()
    print(f"    rule reproduces the label on {agree:.1%} of rows")
    if agree > 0.95:
        print("    >>> LEAK. The label is a threshold rule already in your features.")
        findings.append("the published rule reproduces the label directly")
    else:
        print("    (miss_distance is one close approach, not true MOID, so exact")
        print("     agreement is not expected - check [1] and [4] instead)")

    print("\n[3] Each feature's AUC ON ITS OWN (>0.98 = gives the answer away)")
    for i, f in enumerate(FEATURES):
        col = np.nan_to_num(X[:, i], nan=np.nanmedian(X[:, i]))
        if len(np.unique(col)) < 2:
            continue
        a = max(auc(y, col), auc(y, -col))
        flag = "   <-- LEAK" if a > 0.98 else ("   <-- very high" if a > 0.90 else "")
        print(f"    {f:24s} {a:.4f}{flag}")
        if a > 0.98:
            findings.append(f"{f} alone predicts the label")

    tr, te = stratified_split(y)
    print(f"\n    (train {tr.sum()} / test {te.sum()})")

    print("\n[4] Can a 2-question tree do the whole job?")
    a2, t2 = tree_auc(X, y, FEATURES, 2, tr, te)
    print(f"    depth-2 tree, held-out AUC = {a2:.4f}")
    print("    the questions it chose:")
    show_tree(t2)
    if a2 > 0.95:
        print("    >>> Two yes/no questions are essentially enough. Your 300-tree")
        print("    >>> XGBoost is not learning anything the definition did not")
        print("    >>> already hand it.")
        findings.append("a 2-question tree nearly matches the full model")

    print("\n[5] Drop one feature at a time (depth-4 tree)")
    base, _ = tree_auc(X, y, FEATURES, 4, tr, te)
    print(f"    baseline, all features    {base:.4f}")
    for i, f in enumerate(FEATURES):
        keep = [j for j in range(len(FEATURES)) if j != i]
        a3, _ = tree_auc(X[:, keep], y, [FEATURES[j] for j in keep], 4, tr, te)
        print(f"    without {f:24s} {a3:.4f}  ({a3 - base:+.4f})")

    print("\n[6] Drop the ENTIRE magnitude family")
    h_family = ["magnitude_h", "diameter_km", "log_diameter",
                "size_category", "kinetic_energy_proxy", "threat_score"]
    keep = [i for i, f in enumerate(FEATURES) if f not in h_family]
    rest = [FEATURES[i] for i in keep]
    a4, _ = tree_auc(X[:, keep], y, rest, 4, tr, te)
    print(f"    remaining: {rest}")
    print(f"    AUC = {a4:.4f}")
    print("    This is roughly what the model is worth with the definitional")
    print("    inputs removed.")

    print("\n" + "=" * 70)
    if findings:
        print("VERDICT: target leakage confirmed")
        for f in sorted(set(findings)):
            print(f"  - {f}")
    else:
        print("VERDICT: not confirmed by these checks. Investigate duplicates")
        print("and the close_approach_data[0] selection.")
    print("=" * 70)


if __name__ == "__main__":
    main()
