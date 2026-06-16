"""Camera parameters used by the D4PED DFH-4 inference scripts."""

import numpy as np

D4PED_CAMERA_INTRINSIC = np.array([
    [10797.7175384583, 0.0, 544.276512067169],
    [0.0, 10808.4020025212, 384.961654616631],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

D4PED_DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
