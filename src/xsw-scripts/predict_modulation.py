from pathlib import Path
from typing import Any, Optional, Sequence

import constants
import numpy

# TODO I want to be more explict with there varibales for now i will just translate as prescribed
#
# emitter_z: Optional[float] = None,
# principle_q_num: Optional[int] = None,
# azimuthal_q_num: Optional[int] = None,
# total_spin: Optional[float] = None,
# hkl_index: Optional[Sequence[int]] = None,
# scattering_surface_angle: Optional[float] = None,
# lattice_unit_cell: Optional[Sequence[float]] = None,
# positions_of_atoms: Optional[Sequence[float]] = None,
# gaussian_broadening: Optional[float] = None


def predict_modulation(
    data_dir: Path,
    coherent_fraction: float,
    coherent_position: float,
    theta: float,
    plotter: bool,
    **kwargs,
) -> Any:
    # Bragg angle
    bragg_angle = 90 - 4

    if kwargs:
        sample = "Cu"
        atom_type = constants.Z.index(sample) + 1

    alphaB = 0
    energy = 2971
    a_lat = 3.6149
    hs = numpy.array([1, 1, 1], dtype=numpy.int32)

    lps0 = numpy.array(
        [a_lat, a_lat, a_lat, 90, 90, 90], dtype=numpy.float32
    )  # lattice unit cell = [a b c, alpha beta gamma]
    xyzs = numpy.array(
        [
            [atom_type, 0, 0, 0, 1],
            [atom_type, 0.5, 0.5, 0, 1],
            [atom_type, 0.0, 0.5, 0.5, 1],
            [atom_type, 0.5, 0.0, 0.5, 1],
        ],
        dtype=numpy.float32,
    )

    width = 0.2

    # run q_params function
