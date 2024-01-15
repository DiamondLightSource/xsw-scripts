import numpy as np
import random
from typing import Optional, List
from numpy.typing import NDArray
from dataclasses import dataclass


# could use builder pattern here
# keep track to see how that works
class Peak:
    def __init__(self, energy_range: float, edge_marg: float) -> None:
        self.energy = random.random() * (1 - 2 * edge_marg) + edge_marg
        self.gaussian_width = random.random() * (energy_range / 4 - 0.1) + 0.1
        self.lorentzian_width = random.random() * (energy_range / 4 - 0.1) + 0.1
        asymmetry: random.random()
        step_intensity_coefficient: random.random()
        peak_intensity: random.random()

    def is_valid() -> bool:
        return True


"""
Generate a new peak
- Randomly intialise all properties with a number between 0 and 1
- Rescale energy to not apprioach the margins
- 

"""


"""
Steps to create a new spectra

    - There is a chance the spectrum has more peaks than expected (account for this)
    - check peak position relative to each other (ignore one peak spectra)
        - generate a binding energy range from [0,1] param
        - reorder peaks by energy
        - loop over pairs of peaks getting the energy of the peaks
        - Get gaussian and lorentzian widths of the 2 peaks
        - define energy threshold
        - if lower than threshold discard that peak and try again
        - get largest peak intensity
        - then for each peak if it's intesnity is maller than ithreshold chnage to random value between threshold and max int
        - reoder peaks interms of intensity
    - recale 1st order background polynomial coefficient
    -  Rescale binding energy range to the interval (5 eV, 50 eV).
    - Convert all peak energies in all spectra from fractional to (relative) binding energy in eV.
    - 

"""


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

    def get_energy_range(self):
        return self.binding_energy_range * 45 + 5
