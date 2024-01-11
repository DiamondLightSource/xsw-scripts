import numpy as np
import random
from typing import Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass


class Peak:
    energy: float
    gaussian_width: float
    lorentzian_width: float
    asymmetry: float
    step_intensity_coefficient: float
    peak_intensity: float

    def __init__(self, edge_margin: float) -> None:
        random_params = np.random.random_sample(6)
        random_params[0] = random_params[0]* (1 - 2 * edge_margin) + edge_margin



@dataclass
class Spectra:
    num_peaks: int
    noise_std: float
    noise_seed: float
    binding_energy_range: float
    num_datapoints: int
    background_coefficients: NDArray[np.float32]
    peak_parameters: NDArray[np.float32]
    peaks: List[Peak]


def gen_cs(
    num_spectra: int,
    seed: Optional[float] = None,
    num_peaks: Optional[int] = None,
    max_num_peaks: Optional[int] = 6,
    edge_margin: float = 0.05,
    width_seperation: float = 0.75,
    min_allowed_intensity: float = 4,
    min_slope: float = -0.5,
    cutoff_higher: float = 0.95,
) -> NDArray:
    """Uses number generation to create a 2D Array of parameters which are used to build XP xp spectra

    Each parameter, after being appropriately scaled, will eventually be used to describe specific features of the spectra
    e.g. number of peaks, peak positions, widths, the noise intensity etc.

     Arguments:
    """
    if seed:
        random.seed(seed)

    num_cs = max_num_peaks * 6 + 10
    cs = np.array(
        [[random.uniform(0, 1) for m in range(num_spectra)] for n in range(num_cs)],
        dtype="float32",
    )
    print(cs.shape)

    cs = np.random.random_sample((max_num_peaks * 6, num_spectra)).astype(np.float32)
    print(cs.shape)

    my_peak = Peak(*)
    print(my_peak)


if __name__ == "__main__":
    gen_cs(7)
