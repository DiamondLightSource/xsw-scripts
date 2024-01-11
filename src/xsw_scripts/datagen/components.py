import math
import numpy as np
from numpy.typing import NDArray


def doniach_sunjic(
    energy_scale: NDArray,
    energy: float,
    width: float,
    asymmetry: float,
    intensity: float,
) -> NDArray:
    """Implementation of the Doniach-Sunjic lineshape (see REF) (intensity is defined here as the peak maximum).

    Args:
        energy_scale: The array of energy values over which the function is to be drawn (the x-axis).
        energy: The energy (centroid) of the peak.
        width: Parameter related to the width of the peak.
        asymmetry: Parameter describing how asymmetrical the peak is.
        intensity: The intensity of the peak (defined here as the maximum height).
    Returns:
        NDArray: linespace
    """
    arctan = np.arctan((energy_scale - energy) / width)
    numer = np.cos(0.5 * np.pi * asymmetry + (1 - asymmetry) * arctan)
    denom = (width**2 + (energy_scale - energy) ** 2) ** (0.5 * (1 - asymmetry))
    quotient = numer / denom
    return intensity * quotient / max(quotient)


def gaussian(
    energy_scale: NDArray, energy: float, width: float, intensity: float
) -> NDArray:
    """Implementation of the Gaussian lineshape (intensity is defined here as the peak maximum).

    Args:
        energy_scale: The array of energy values over which the function is to be drawn (the x-axis).
        energy: The energy (centroid) of the peak.
        width: Parameter related to the width of the peak.
        asymmetry: Parameter describing how asymmetrical the peak is.
        intensity: The intensity of the peak (defined here as the maximum height).
    Returns:
        NDArray: linespace
    """
    return intensity * np.exp(-((energy_scale - energy) ** 2) / (2 * width**2))


def step_func(
    energy_scale: NDArray, energy: float, width: float, intensity: float
) -> NDArray:
    """Implementation of the step function (based on the Gaussian error function)

    Args:
        energy_scale: The array of energy values over which the function is to be drawn (the x-axis).
        energy: The energy (centroid) of the peak.
        width: Parameter related to the width of the peak.
        asymmetry: Parameter describing how asymmetrical the peak is.
        intensity: The intensity of the peak (defined here as the maximum height).
    Returns:
        NDArray: linespace
    """
    return (
        0.5
        * intensity
        * (
            np.array(
                [math.erf((x - energy) / width) for x in energy_scale],
                dtype="float32",
            )
            + 1
        )
    )
