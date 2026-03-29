# ============================================================
#  CICIDS2017 — BASELINE RF + PSO + GENETIC ALGORITHM
#  Add these cells AFTER your dataProc.ipynb cells
#  Your data is already loaded as: X_train_scaled, X_test_scaled,
#  y_train (binary), y_test (binary), feature_cols
# ============================================================


# ┌─────────────────────────────────────────────────────────┐
# │  CELL A — Install & Import (add after dataProc cells)   │
# └─────────────────────────────────────────────────────────┘

!pip install pyswarms --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import json
import time
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

import pyswarms as ps

print(" All libraries ready!")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL B — Load Your Preprocessed Data                   │
# │  (Skip this if you're continuing in the same session)   │
# └─────────────────────────────────────────────────────────┘

# ── Option 1: Continuing in same session as dataProc.ipynb ──
# X_train_scaled, X_test_scaled, y_train, y_test are already
# in memory — just run Cell C directly.

# ── Option 2: Starting a fresh Colab session ──
# Uncomment and run the block below to reload from Drive:

# from google.colab import drive
# drive.mount('/content/drive')
# import numpy as np
#
# save_path = '/content/drive/MyDrive/AIMcw/preprocessed_data/'
#
# X_train_scaled = np.load(save_path + 'X_train_scaled.npy')
# X_test_scaled  = np.load(save_path + 'X_test_scaled.npy')
# y_train        = np.load(save_path + 'y_binary_full.npy')  # adjust name if needed
# y_test         = np.load(save_path + 'y_binary_full.npy')  # adjust name if needed
#
# with open(save_path + 'feature_names.txt') as f:
#     feature_cols = [line.strip() for line in f.readlines()]
#
# print(f"Loaded: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}")
# print(f"Features: {len(feature_cols)}")

print(" Data ready — proceeding with baseline model")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL C — Create Validation Split from Training Data    │
# └─────────────────────────────────────────────────────────┘
# dataProc.ipynb only created train/test.
# We carve out a validation set from training for PSO & GA.

print("Creating validation split from training data...")
print(f"Original training size: {X_train_scaled.shape[0]:,} rows")

# Use 10% of training data as validation
# (keeping 90% for training because CICIDS2017 is large)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.10,
    random_state=42,
    stratify=y_train
)

print(f" Split complete:")
print(f"   Final train : {X_tr.shape[0]:,} rows")
print(f"   Validation  : {X_val.shape[0]:,} rows")
print(f"   Test        : {X_test_scaled.shape[0]:,} rows")
print(f"   Features    : {X_tr.shape[1]}")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL D — BASELINE MODEL: Train on ALL 70 Features      │
# └─────────────────────────────────────────────────────────┘

print("\n" + "=" * 55)
print("  BASELINE: RANDOM FOREST — ALL 70 FEATURES")
print("=" * 55)

# NOTE: CICIDS2017 has ~2M rows — we use a sample for speed
# Remove the sample_size limit if you have time/GPU runtime
SAMPLE_SIZE = 200_000   # ← increase to 500_000 or remove limit for full data

if X_tr.shape[0] > SAMPLE_SIZE:
    idx = np.random.choice(X_tr.shape[0], SAMPLE_SIZE, replace=False)
    X_tr_sample = X_tr[idx]
    y_tr_sample = y_tr.iloc[idx] if hasattr(y_tr, 'iloc') else y_tr[idx]
    print(f" Using {SAMPLE_SIZE:,} sample rows for speed (full={X_tr.shape[0]:,})")
else:
    X_tr_sample = X_tr
    y_tr_sample = y_tr
    print(f"Using full training set: {X_tr.shape[0]:,} rows")

# Train baseline RF with ALL features
start = time.time()
baseline_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,           # cap depth for large dataset speed
    random_state=42,
    n_jobs=-1,
    class_weight='balanced' # handles CICIDS2017 class imbalance (83% benign)
)
baseline_rf.fit(X_tr_sample, y_tr_sample)
train_time = time.time() - start

print(f"\n Baseline RF trained in {train_time:.1f}s")
print(f"   Features used: {X_tr_sample.shape[1]} (ALL features)")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL E — Validate Baseline                             │
# └─────────────────────────────────────────────────────────┘

print("\n" + "=" * 55)
print("  VALIDATE ON VALIDATION SET")
print("=" * 55)

y_val_pred  = baseline_rf.predict(X_val)
y_val_proba = baseline_rf.predict_proba(X_val)[:, 1]

print(f" Predicted {len(y_val_pred):,} validation samples")
print(f"   Predicted attacks : {y_val_pred.sum():,}")
print(f"   Actual attacks    : {(y_val == 1).sum():,}")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL F — Baseline Metrics                              │
# └─────────────────────────────────────────────────────────┘

print("\n" + "=" * 55)
print("  BASELINE PERFORMANCE METRICS (Validation Set)")
print("=" * 55)

baseline_val_acc  = accuracy_score(y_val, y_val_pred)
baseline_val_prec = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
baseline_val_rec  = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
baseline_val_f1   = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
baseline_val_auc  = roc_auc_score(y_val, y_val_proba)

print(f"\n  Accuracy  : {baseline_val_acc:.4f}")
print(f"  Precision : {baseline_val_prec:.4f}")
print(f"  Recall    : {baseline_val_rec:.4f}")
print(f"  F1-Score  : {baseline_val_f1:.4f}")
print(f"  ROC-AUC   : {baseline_val_auc:.4f}")

print("\n Full Classification Report (Validation):")
print(classification_report(y_val, y_val_pred,
      target_names=["BENIGN (0)", "ATTACK (1)"], zero_division=0))


# ┌─────────────────────────────────────────────────────────┐
# │  CELL G — Confusion Matrix                              │
# └─────────────────────────────────────────────────────────┘

print("\n" + "=" * 55)
print("  CONFUSION MATRIX (Validation Set)")
print("=" * 55)

cm_val = confusion_matrix(y_val, y_val_pred)
print("\nRaw matrix:")
print(cm_val)
print(f"\n  TN (Correctly BENIGN) : {cm_val[0,0]:,}")
print(f"  FP (False Alarm)      : {cm_val[0,1]:,}  ← BENIGN flagged as attack")
print(f"  FN (Missed Attack)    : {cm_val[1,0]:,}  ← Attack missed  ")
print(f"  TP (Correctly ATTACK) : {cm_val[1,1]:,}")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm_val,
                       display_labels=["BENIGN", "ATTACK"]).plot(
    cmap='Blues', ax=ax, colorbar=False)
ax.set_title("Baseline RF — Confusion Matrix (Validation)", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig("confusion_matrix_baseline.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix_baseline.png")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL H — Final Test Evaluation                         │
# └─────────────────────────────────────────────────────────┘

print("\n" + "=" * 55)
print("  FINAL TEST EVALUATION (Unseen Test Set)")
print("=" * 55)

y_test_pred  = baseline_rf.predict(X_test_scaled)
y_test_proba = baseline_rf.predict_proba(X_test_scaled)[:, 1]

baseline_test_metrics = {
    "model":      "Random Forest (Baseline — All 70 Features)",
    "accuracy":   float(accuracy_score(y_test, y_test_pred)),
    "precision":  float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
    "recall":     float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
    "f1":         float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
    "roc_auc":    float(roc_auc_score(y_test, y_test_proba)),
    "n_features": int(X_test_scaled.shape[1])
}

print(f"\n  Test Accuracy  : {baseline_test_metrics['accuracy']:.4f}")
print(f"  Test Precision : {baseline_test_metrics['precision']:.4f}")
print(f"  Test Recall    : {baseline_test_metrics['recall']:.4f}")
print(f"  Test F1-Score  : {baseline_test_metrics['f1']:.4f}")
print(f"  Test ROC-AUC   : {baseline_test_metrics['roc_auc']:.4f}")
print(f"\n BASELINE F1 TO BEAT: {baseline_test_metrics['f1']:.4f}")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL I — Feature Importance Plot                       │
# └─────────────────────────────────────────────────────────┘

importances = baseline_rf.feature_importances_
feat_names  = feature_cols if 'feature_cols' in dir() else [f"F{i}" for i in range(len(importances))]

# Top 20 features
top_n = 20
indices = np.argsort(importances)[::-1][:top_n]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(top_n),
       importances[indices],
       color=['#2563eb' if i < 10 else '#7c3aed' for i in range(top_n)],
       edgecolor='white', linewidth=0.5)
ax.set_xticks(range(top_n))
ax.set_xticklabels([feat_names[i] for i in indices], rotation=60, ha='right', fontsize=8)
ax.set_title(f"Baseline RF — Top {top_n} Most Important Features (out of {len(importances)})",
             fontsize=12, pad=10)
ax.set_ylabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_baseline.png", dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: feature_importance_baseline.png")

# Show top 10 feature names
print("\nTop 10 most important features:")
for rank, i in enumerate(indices[:10], 1):
    print(f"  {rank:2}. {feat_names[i]:<40} {importances[i]:.4f}")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL J — Save Baseline                                 │
# └─────────────────────────────────────────────────────────┘

with open('baseline_rf_cicids.pkl', 'wb') as f:
    pickle.dump(baseline_rf, f)

with open('baseline_metrics.json', 'w') as f:
    json.dump(baseline_test_metrics, f, indent=2)

print(" baseline_rf_cicids.pkl  → saved")
print(" baseline_metrics.json  → saved")
print(f"\n BASELINE RECORDED:")
print(f"   Model    : {baseline_test_metrics['model']}")
print(f"   Features : {baseline_test_metrics['n_features']}")
print(f"   F1-Score : {baseline_test_metrics['f1']:.4f}  ← PSO/GA must beat this!")


# ═══════════════════════════════════════════════════════════
#   PSO FEATURE SELECTION
# ═══════════════════════════════════════════════════════════


# ┌─────────────────────────────────────────────────────────┐
# │  CELL K — PSO Setup                                     │
# └─────────────────────────────────────────────────────────┘
# PSO on 70 features is expensive — we use a subsample
# for the fitness evaluations inside PSO/GA

PSO_SAMPLE = 50_000   # rows used per fitness evaluation
                      # increase if you have more time

# Subsample for PSO/GA fitness evaluation
idx_pso = np.random.choice(X_tr.shape[0], min(PSO_SAMPLE, X_tr.shape[0]), replace=False)
X_pso_train = X_tr[idx_pso]
y_pso_train = y_tr.iloc[idx_pso] if hasattr(y_tr, 'iloc') else y_tr[idx_pso]

# Smaller validation sample for speed
idx_val = np.random.choice(X_val.shape[0], min(20_000, X_val.shape[0]), replace=False)
X_pso_val = X_val[idx_val]
y_pso_val = y_val.iloc[idx_val] if hasattr(y_val, 'iloc') else y_val[idx_val]

n_features = X_tr.shape[1]
print(f"PSO/GA setup:")
print(f"  Total features     : {n_features}")
print(f"  Train sample size  : {X_pso_train.shape[0]:,}")
print(f"  Val sample size    : {X_pso_val.shape[0]:,}")
print(f"\n  Note: Each PSO/GA fitness call trains a RF — this takes time.")
print(f"   ~30 particles × 50 iters = ~1500 RF trains for PSO")


def pso_fitness(particles):
    """
    PSO fitness function.
    Each particle = 70 floats. Value > 0.5 → use that feature.
    PSO minimises, so we return -F1.
    """
    scores = []
    for particle in particles:
        mask = particle > 0.5
        if mask.sum() == 0:
            scores.append(1.0)
            continue

        rf = RandomForestClassifier(
            n_estimators=30,       # small for speed inside PSO
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_pso_train[:, mask], y_pso_train)
        y_pred = rf.predict(X_pso_val[:, mask])
        f1 = f1_score(y_pso_val, y_pred, average='weighted', zero_division=0)

        # Small penalty per feature (rewards fewer features)
        penalty = 0.0005 * mask.sum()
        scores.append(-f1 + penalty)

    return np.array(scores)

print("\n PSO fitness function defined")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL L — Run PSO                                       │
# └─────────────────────────────────────────────────────────┘

pso_options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
bounds = (np.zeros(n_features), np.ones(n_features))

print(f"Running PSO: 30 particles × 50 iterations on {n_features} features")
print("(This will take several minutes on CICIDS2017...)\n")

optimizer = ps.single.GlobalBestPSO(
    n_particles=30,
    dimensions=n_features,
    options=pso_options,
    bounds=bounds
)

start_pso = time.time()
best_cost_pso, best_pos_pso = optimizer.optimize(pso_fitness, iters=50, verbose=True)
pso_time = time.time() - start_pso

print(f"\n PSO done in {pso_time/60:.1f} minutes")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL M — Evaluate PSO Model                            │
# └─────────────────────────────────────────────────────────┘

pso_mask     = best_pos_pso > 0.5
pso_feat_idx = np.where(pso_mask)[0]
pso_n_feats  = int(pso_mask.sum())
feat_names_list = feature_cols if 'feature_cols' in dir() else [f"F{i}" for i in range(n_features)]

print(f"\nPSO selected {pso_n_feats} / {n_features} features:")
print(f"  Kept    : {[feat_names_list[i] for i in pso_feat_idx]}")
print(f"  Dropped : {[feat_names_list[i] for i in range(n_features) if i not in pso_feat_idx]}")

# Train final PSO model on FULL training sample
pso_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
pso_rf.fit(X_tr_sample[:, pso_mask], y_tr_sample)

y_test_pred_pso  = pso_rf.predict(X_test_scaled[:, pso_mask])
y_test_proba_pso = pso_rf.predict_proba(X_test_scaled[:, pso_mask])[:, 1]

pso_metrics = {
    "model":      "Random Forest + PSO",
    "accuracy":   float(accuracy_score(y_test, y_test_pred_pso)),
    "precision":  float(precision_score(y_test, y_test_pred_pso, average='weighted', zero_division=0)),
    "recall":     float(recall_score(y_test, y_test_pred_pso, average='weighted', zero_division=0)),
    "f1":         float(f1_score(y_test, y_test_pred_pso, average='weighted', zero_division=0)),
    "roc_auc":    float(roc_auc_score(y_test, y_test_proba_pso)),
    "n_features": pso_n_feats
}

print(f"\n PSO Model — Test Results:")
print(f"  Accuracy  : {pso_metrics['accuracy']:.4f}")
print(f"  Precision : {pso_metrics['precision']:.4f}")
print(f"  Recall    : {pso_metrics['recall']:.4f}")
print(f"  F1-Score  : {pso_metrics['f1']:.4f}")
print(f"  ROC-AUC   : {pso_metrics['roc_auc']:.4f}")
print(f"  Features  : {pso_n_feats} (was {n_features})")

diff = pso_metrics['f1'] - baseline_test_metrics['f1']
if diff > 0:
    print(f"\n PSO BEAT BASELINE by +{diff:.4f} F1 using {n_features - pso_n_feats} fewer features!")
else:
    print(f"\n PSO vs baseline: Δ F1 = {diff:.4f}")


# ═══════════════════════════════════════════════════════════
#   GENETIC ALGORITHM FEATURE SELECTION
# ═══════════════════════════════════════════════════════════


# ┌─────────────────────────────────────────────────────────┐
# │  CELL N — GA Functions                                  │
# └─────────────────────────────────────────────────────────┘

def ga_fitness(chromosome):
    """chromosome: binary array of length 70. 1=use, 0=skip."""
    mask = chromosome == 1
    if mask.sum() == 0:
        return 0.0
    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_pso_train[:, mask], y_pso_train)
    y_pred = rf.predict(X_pso_val[:, mask])
    return float(f1_score(y_pso_val, y_pred, average='weighted', zero_division=0))


def tournament_selection(population, scores, k=3):
    idx = np.random.choice(len(population), k, replace=False)
    return population[idx[np.argmax(scores[idx])]].copy()


def crossover(p1, p2, rate=0.9):
    if np.random.rand() < rate:
        pt = np.random.randint(1, len(p1))
        return (np.concatenate([p1[:pt], p2[pt:]]),
                np.concatenate([p2[:pt], p1[pt:]]))
    return p1.copy(), p2.copy()


def mutate(chrom, rate=0.05):
    flip = np.random.rand(len(chrom)) < rate
    chrom[flip] = 1 - chrom[flip]
    return chrom

print(" GA functions defined")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL O — Run Genetic Algorithm                         │
# └─────────────────────────────────────────────────────────┘

GA_POP       = 30
GA_GENS      = 40
GA_MUTATION  = 0.05
GA_ELITES    = 2

print(f"Running GA: {GA_POP} chromosomes × {GA_GENS} generations")
print(f"Each chromosome = {n_features} bits (1=use feature, 0=drop)\n")

population = np.random.randint(0, 2, (GA_POP, n_features))
best_per_gen = []
avg_per_gen  = []

start_ga = time.time()

for gen in range(GA_GENS):
    scores = np.array([ga_fitness(c) for c in population])
    best_idx = np.argmax(scores)

    best_per_gen.append(scores[best_idx])
    avg_per_gen.append(scores.mean())

    print(f"  Gen {gen+1:02d}/{GA_GENS} | "
          f"Best F1: {scores[best_idx]:.4f} | "
          f"Avg F1: {scores.mean():.4f} | "
          f"Feats: {int(population[best_idx].sum())}")

    # Elitism
    top_idx = np.argsort(scores)[::-1][:GA_ELITES]
    new_pop = [population[i].copy() for i in top_idx]

    # Breed rest
    while len(new_pop) < GA_POP:
        p1 = tournament_selection(population, scores)
        p2 = tournament_selection(population, scores)
        c1, c2 = crossover(p1, p2)
        new_pop.append(mutate(c1, GA_MUTATION))
        if len(new_pop) < GA_POP:
            new_pop.append(mutate(c2, GA_MUTATION))

    population = np.array(new_pop)

ga_time = time.time() - start_ga
print(f"\n GA done in {ga_time/60:.1f} minutes")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL P — Evaluate GA Model                             │
# └─────────────────────────────────────────────────────────┘

final_scores  = np.array([ga_fitness(c) for c in population])
best_chrom    = population[np.argmax(final_scores)]
ga_mask       = best_chrom == 1
ga_feat_idx   = np.where(ga_mask)[0]
ga_n_feats    = int(ga_mask.sum())

print(f"\nGA selected {ga_n_feats} / {n_features} features:")
print(f"  Kept    : {[feat_names_list[i] for i in ga_feat_idx]}")
print(f"  Dropped : {[feat_names_list[i] for i in range(n_features) if i not in ga_feat_idx]}")

# Final model with GA features
ga_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
ga_rf.fit(X_tr_sample[:, ga_mask], y_tr_sample)

y_test_pred_ga  = ga_rf.predict(X_test_scaled[:, ga_mask])
y_test_proba_ga = ga_rf.predict_proba(X_test_scaled[:, ga_mask])[:, 1]

ga_metrics = {
    "model":      "Random Forest + GA",
    "accuracy":   float(accuracy_score(y_test, y_test_pred_ga)),
    "precision":  float(precision_score(y_test, y_test_pred_ga, average='weighted', zero_division=0)),
    "recall":     float(recall_score(y_test, y_test_pred_ga, average='weighted', zero_division=0)),
    "f1":         float(f1_score(y_test, y_test_pred_ga, average='weighted', zero_division=0)),
    "roc_auc":    float(roc_auc_score(y_test, y_test_proba_ga)),
    "n_features": ga_n_feats
}

print(f"\n GA Model — Test Results:")
print(f"  Accuracy  : {ga_metrics['accuracy']:.4f}")
print(f"  Precision : {ga_metrics['precision']:.4f}")
print(f"  Recall    : {ga_metrics['recall']:.4f}")
print(f"  F1-Score  : {ga_metrics['f1']:.4f}")
print(f"  ROC-AUC   : {ga_metrics['roc_auc']:.4f}")
print(f"  Features  : {ga_n_feats} (was {n_features})")

diff_ga = ga_metrics['f1'] - baseline_test_metrics['f1']
if diff_ga > 0:
    print(f"\n GA BEAT BASELINE by +{diff_ga:.4f} F1 using {n_features - ga_n_feats} fewer features!")
else:
    print(f"\n GA vs baseline: Δ F1 = {diff_ga:.4f}")


# ═══════════════════════════════════════════════════════════
#   FINAL RESULTS & CHARTS
# ═══════════════════════════════════════════════════════════


# ┌─────────────────────────────────────────────────────────┐
# │  CELL Q — Final Comparison Table                        │
# └─────────────────────────────────────────────────────────┘

all_results = [baseline_test_metrics, pso_metrics, ga_metrics]

print("\n" + "=" * 78)
print("  FINAL COMPARISON — CICIDS2017 INTRUSION DETECTION")
print("=" * 78)
print(f"\n{'Model':<38} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Feats':>6}")
print("-" * 78)

for r in all_results:
    marker = "★" if r == max(all_results, key=lambda x: x['f1']) else ""
    print(f"{r['model']:<38}"
          f"{r['accuracy']:>7.4f}"
          f"{r['precision']:>7.4f}"
          f"{r['recall']:>7.4f}"
          f"{r['f1']:>7.4f}"
          f"{r['roc_auc']:>7.4f}"
          f"{r['n_features']:>6}{marker}")

print("-" * 78)
best = max(all_results, key=lambda x: x['f1'])
print(f"\n Best model: {best['model']}  |  F1={best['f1']:.4f}  |  Features={best['n_features']}")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL R — Charts                                        │
# └─────────────────────────────────────────────────────────┘

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("CICIDS2017 — Baseline vs PSO vs GA Feature Selection",
             fontsize=13, fontweight='bold')

# Plot 1: Metrics grouped bar
ax1 = axes[0]
metric_keys  = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_label = ['Acc', 'Prec', 'Rec', 'F1', 'AUC']
x_pos   = np.arange(len(metric_keys))
w       = 0.26
colors  = ['#dc2626', '#2563eb', '#059669']
m_labels = ['Baseline', 'PSO', 'GA']
for i, (r, c, lbl) in enumerate(zip(all_results, colors, m_labels)):
    ax1.bar(x_pos + i*w, [r[k] for k in metric_keys], w,
            label=lbl, color=c, alpha=0.85)
ax1.set_xticks(x_pos + w)
ax1.set_xticklabels(metric_label)
ax1.set_ylim(min(r['f1'] for r in all_results) - 0.05, 1.01)
ax1.set_title("Performance Metrics")
ax1.set_ylabel("Score")
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Features used
ax2 = axes[1]
feat_counts = [r['n_features'] for r in all_results]
bars = ax2.bar(m_labels, feat_counts, color=colors, alpha=0.85, width=0.4)
for bar, cnt in zip(bars, feat_counts):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3, str(cnt),
             ha='center', fontweight='bold', fontsize=12)
ax2.axhline(n_features, color='grey', linestyle='--', linewidth=1, label=f'All {n_features}')
ax2.set_ylim(0, n_features + 10)
ax2.set_title("Features Used per Model")
ax2.set_ylabel("Number of Features")
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: GA convergence
ax3 = axes[2]
gens = range(1, GA_GENS + 1)
ax3.plot(gens, best_per_gen, color='#059669', linewidth=2, label='Best F1')
ax3.plot(gens, avg_per_gen,  color='#d97706', linewidth=1.5, linestyle='--', label='Avg F1')
ax3.axhline(baseline_test_metrics['f1'], color='#dc2626',
            linestyle=':', linewidth=1.5, label=f"Baseline F1 ({baseline_test_metrics['f1']:.3f})")
ax3.set_xlabel("Generation")
ax3.set_ylabel("F1 Score")
ax3.set_title("GA Convergence Curve")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("final_comparison_cicids.png", dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: final_comparison_cicids.png")


# ┌─────────────────────────────────────────────────────────┐
# │  CELL S — Save Everything                               │
# └─────────────────────────────────────────────────────────┘

save_path = '/content/drive/MyDrive/AIMcw/results/'
os.makedirs(save_path, exist_ok=True)

# Save models
with open(f'{save_path}baseline_rf.pkl', 'wb') as f: pickle.dump(baseline_rf, f)
with open(f'{save_path}pso_rf.pkl',      'wb') as f: pickle.dump(pso_rf, f)
with open(f'{save_path}ga_rf.pkl',       'wb') as f: pickle.dump(ga_rf, f)

# Save all metrics + selected features
final_save = {
    "baseline":          baseline_test_metrics,
    "pso":               {**pso_metrics,
                          "selected_feature_names": [feat_names_list[i] for i in pso_feat_idx],
                          "selected_feature_indices": pso_feat_idx.tolist()},
    "ga":                {**ga_metrics,
                          "selected_feature_names": [feat_names_list[i] for i in ga_feat_idx],
                          "selected_feature_indices": ga_feat_idx.tolist()},
    "best_model":        best['model']
}

with open(f'{save_path}all_results.json', 'w') as f:
    json.dump(final_save, f, indent=2)

print("All models + results saved to Drive!")
print(f"\n REPORT TABLE (copy this):\n")
print(f"{'Model':<40} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1-Score':>9} {'ROC-AUC':>8} {'Features':>8}")
print("-" * 97)
for r in all_results:
    print(f"{r['model']:<40}"
          f"{r['accuracy']:>9.4f}"
          f"{r['precision']:>9.4f}"
          f"{r['recall']:>9.4f}"
          f"{r['f1']:>9.4f}"
          f"{r['roc_auc']:>8.4f}"
          f"{r['n_features']:>8}")
print("-" * 97)