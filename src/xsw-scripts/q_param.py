from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import constants
import numpy
from numpy.typing import NDArray

azimuthal_2_spin_dict: Dict[str, Union[str, List[float]]] = {
    "s": "",
    "p": (0.5, 1.5),
    "d": (1.5, 2.5),
    "f": (2.5, 3.5),
}


def get_orbital(azimuthal_qn: int) -> Optional[str]:
    """Translates a intiger azimuthal quantum number to the correposing letter orbital.

    Args:
        azimuthal_qn: the azimuthal quntim number

    Returns:
        Optional[str]: the letter of that orbital
    """
    return list(azimuthal_2_spin_dict.keys())[azimuthal_qn]


def get_spin_as_string(azimuthal_qn: int, spin_qn: float) -> Optional[str]:
    """Translates a spin quantum number float into a string represing the coresponding fraction.

    Args:
        azimuthal_qn: the azimuthal quantum number
        spin_qn: the spin quantum number

    Returns:
        Optional[str]: string represing the fraction of the spin quantum number
    """
    orbital = get_orbital(azimuthal_qn)
    spins = azimuthal_2_spin_dict[orbital]

    if isinstance(spins, str):
        return spins
    elif spin_qn in spins:
        return Fraction(spin_qn).__str__()

    raise Exception(f"the {orbital} orbital cannot have a js of: {spin_qn}")


def q_param(
    data_file: Path,
    Z: int,
    principal_qn: int,
    azimuthal_qn: int,
    spin_qn: float,
    plotter: Optional[int] = 1,
) -> Optional[Tuple[NDArray]]:
    """
    Gets the q params from the data_file for a specific principle, azimuthal, and spin quantum number.

    Args:
        data_file: path to file with data in it
        Z:
        principal_qn: principal quantum number
        azimuthal_qn: azimuthal quantum number
        spin_qn: spin quantum number
        plotter: verbosity and plotting

    Returns:
        Optional[Tuple[List[NDArray]]]: params from file
    """
    if plotter:
        print(
            f"{constants.Z[Z]} {principal_qn}{get_orbital(azimuthal_qn)}{get_spin_as_string(azimuthal_qn,spin_qn)}"
        )

    with open(data_file, "r") as param_file:
        name = f"{Z} {principal_qn}{get_orbital(azimuthal_qn)}{get_spin_as_string(azimuthal_qn,spin_qn)}"
        for line in param_file:
            if line.strip() == name:
                Eb = numpy.fromstring(param_file.readline(), int, sep=" ")
                print(Eb)
                Erow = numpy.fromstring(param_file.readline(), int, sep=" ")

                sigma = numpy.stack(
                    [
                        Erow,
                        numpy.fromstring(param_file.readline(), numpy.float64, sep=" "),
                    ]
                )
                beta = numpy.stack(
                    [
                        Erow,
                        numpy.fromstring(param_file.readline(), numpy.float64, sep=" "),
                    ]
                )
                gamma = numpy.stack(
                    [
                        Erow,
                        numpy.fromstring(param_file.readline(), numpy.float64, sep=" "),
                    ]
                )
                delta = numpy.stack(
                    [
                        Erow,
                        numpy.fromstring(param_file.readline(), numpy.float64, sep=" "),
                    ]
                )
                return (beta, gamma, delta, Eb)
        raise NotImplementedError(f"{name} not found in params.txt")
