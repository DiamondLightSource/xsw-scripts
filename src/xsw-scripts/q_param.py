from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import constants
import numpy as np
from numpy.typing import DTypeLike, NDArray

azimuthal_2_spin_dict: Dict[str, Union[str, Tuple[float]]] = {
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
    """Translates a spin quantum number float into a string
    represing the coresponding fraction.

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
        numerator, denominator = spin_qn.as_integer_ratio()
        return f"{numerator}/{denominator}"

    raise KeyError(f"The {orbital} orbital cannot have a js of: {spin_qn}")


def q_param(
    data_file: Path,
    Z: int,
    principal_qn: int,
    azimuthal_qn: int,
    spin_qn: float,
    plotter: Optional[int] = 1,
    dtype: DTypeLike = np.float64,
) -> Optional[Tuple[NDArray]]:
    """
    Gets the q params from the data_file for a specific
    principal, azimuthal, and spin quantum number.

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
                Eb = np.fromstring(param_file.readline(), dtype, sep=" ")
                Erow = np.fromstring(param_file.readline(), dtype, sep=" ")
                # TODO is the sigma here needed
                _ = np.stack(
                    [
                        Erow,
                        np.fromstring(param_file.readline(), dtype, sep=" "),
                    ]
                )
                beta = np.stack(
                    [
                        Erow,
                        np.fromstring(param_file.readline(), dtype, sep=" "),
                    ]
                )
                gamma = np.stack(
                    [
                        Erow,
                        np.fromstring(param_file.readline(), dtype, sep=" "),
                    ]
                )
                delta = np.stack(
                    [
                        Erow,
                        np.fromstring(param_file.readline(), dtype, sep=" "),
                    ]
                )
                return (beta, gamma, delta, Eb)
        raise NotImplementedError(f"{name} not found in params.txt")
