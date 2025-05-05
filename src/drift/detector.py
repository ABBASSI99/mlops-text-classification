import numpy as np
from scipy.stats import ks_2samp

# Simple KS drift detection based on predictions distributions
def detect_drift(preds_ref, preds_prod, threshold=0.1):
    stat, pval = ks_2samp(preds_ref, preds_prod)
    drift = pval < threshold
    return drift, stat, pval

if __name__ == "__main__":
    # Ex: on compare les prédictions historiques et récentes
    preds_ref = np.random.binomial(1, 0.7, 200)
    preds_prod = np.random.binomial(1, 0.5, 200)
    drift, stat, pval = detect_drift(preds_ref, preds_prod)
    print("Drift detected:", drift, "KS stat:", stat, "p-value:", pval)