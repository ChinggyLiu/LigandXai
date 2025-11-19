import numpy as np

MASS_TABLE = {
    "H": 1.008, "D": 2.014, "T": 3.016,
    "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06,
    "CL": 35.45, "BR": 79.904, "I": 126.90
}

def mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size > 0 else np.nan



def std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std()) if x.size > 0 else np.nan

def center_of_mass(coords, weights=None):

    coords = np.asarray(coords, float)
    if coords.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    if weights is None:
        return coords.mean(axis=0)

    weights = np.asarray(weights)

    if weights.dtype.kind in ("U", "S", "O"):
        masses = np.array([
            MASS_TABLE.get(str(el).strip().upper(), 12.0) for el in weights
        ], dtype=float)
    else:
        masses = weights.astype(float)

    if masses.shape[0] != coords.shape[0]:
        return coords.mean(axis=0)

    total = masses.sum()
    if total <= 0:
        return coords.mean(axis=0)

    return (coords * masses[:, None]).sum(axis=0) / total