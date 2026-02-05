from scipy.stats import norm
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator

def get_XRD_pattern(structure, 
    thetas = np.arange(0, 0.01, 90),
    σ = 0.2,
    wavelength='CuKa', 
    debye_waller_factors=None, 
    scaled=True, 
    ):
    two_theta_range = (np.min(thetas), np.max(thetas))
    xrd_peaks = XRDCalculator(
        wavelength = wavelength, 
        debye_waller_factors = debye_waller_factors
    ).get_pattern(
        structure, 
        scaled = scaled, 
        two_theta_range = two_theta_range
    )
    pattern = sum(map(lambda i: xrd_peaks.y[i] * 
        norm.pdf(thetas, xrd_peaks.x[i], σ), range(len(xrd_peaks.x))))
    return 100 * pattern / np.max(pattern)