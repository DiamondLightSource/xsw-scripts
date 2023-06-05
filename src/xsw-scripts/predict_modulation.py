from pathlib import Path
from typing import Any, Optional, Sequence

import math
import constants
import numpy
from q_param import q_param
from calculate_energy import calculate_energy
from scipy.interpolate import interp1d

# Maybe use this maybe don't
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
    *args,
) -> Any:
    """calcuates XSW absorption profile from inputted parameters.

    Args:
        datadir: folder containing the structure factors (*.nff files) and f0_all_free_atoms.txt
        coherent_fraction: coherent fraction
        coherent_position: coherent position
        theta: angle between the measured emission direction and the photon
            polarisation (for most I09 stuff this is 18)
        plotter: if greater than zero plots figures and outputs text to the command line

        args[0]: Z value of the emitter atom (e.g. 8 for oxygen)
        args[1]: principle quantum number, n, for photoelectron emitting orbital
        args[2]: azimuthal quantum number, l, for photoelectron emitting orbital,
                e.g. 0 for s, 1 for l, 2 for d, 3 for f
        args[3]: total spin, js = l+ms, for photoelectron emitting orbital
                e.g. 3/2 for the Ti 2p 3/2 level. For an s orbital set to 1/2
        args[4]: angle between the scattering plane and the surface, usually 0
        args[5]: (h,k,l) index of the reflection, e.g. [1,1,1] for the (111)
        args[6]: lattice unit cell = [a b c, alpha beta gamma] in Å and °.
                e.g. for Cu: = [3.6149,3.6149,3.6149, 90,90,90]
        args[7]: position of atoms in the unit cell in fractional
                    coordinates and an occupational factor (usually 1) in the
                    format [Z, x, y, z, f], e.g. for fcc Cu:
                    = [29, 0.0, 0.0, 0.0, 1;
                        29, 0.5, 0.5, 0.0, 1;
                        29, 0.5, 0.0, 0.5, 1;
                        29, 0.0, 0.5, 0.5, 1;]
        args[8]: gaussian broadening, models the experimental
              broadening due to imperfections in the monochromator (pretty small)
              and the sample substrate (significantly larger). Numbers between
              0.1 and 0.5 eV are common, 0.3 is broadly the mean
    """
    bragg_angle = 90 - 4

    if not args:
        sample = "Cu"
        z_value = constants.Z.index(sample) + 1
        principal_qn = 1
        azimuthal_qn = 0
        spin_qn = 0.5
        scattering_plane_angle = 0
        hkl_index = numpy.array([1, 1, 1], dtype=numpy.float64)
        a_lat = 3.6149
        lattice_unit_cell = numpy.array(
            [a_lat, a_lat, a_lat, 90, 90, 90], dtype=numpy.float32
        )  # lattice unit cell = [a b c, alpha beta gamma]
        positions_of_atoms = numpy.array(
            [
                [z_value, 0, 0, 0, 1],
                [z_value, 0.5, 0.5, 0, 1],
                [z_value, 0.0, 0.5, 0.5, 1],
                [z_value, 0.5, 0.0, 0.5, 1],
            ],
            dtype=numpy.float64,
        )
        width = 0.2
    elif len(args) == 9:
        (
            z_value,
            principal_qn,
            azimuthal_qn,
            spin_qn,
            scattering_plane_angle,
            hkl_index,
            lattice_unit_cell,
            positions_of_atoms,
            width,
        ) = args
    else:
        raise Exception(
            "You must either include the Z, n, l and js of the emitter orbital, or you must edit the script to include this information"
        )
    energy = calculate_energy(lattice_unit_cell, hkl_index, bragg_angle)
    bettab, gamtab, deltab, Eb = q_param(
        data_dir / Path("q_param.txt"), z_value, principal_qn, azimuthal_qn, spin_qn, plotter
    )
