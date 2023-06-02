""" This file is able to extract q params from the q_params.txt"""
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy

import constants

azimuthal_2_spin_dict: Dict[str, Union[str, List[float]]] = {
    "s": "",
    "p": [0.5, 1.5],
    "d": [1.5, 2.5],
    "f": [2.5, 3.5],
}


def get_orbital(azimuthal_qn: int) -> Optional[str]:
    return list(azimuthal_2_spin_dict.keys())[azimuthal_qn]


def get_spin_as_string(azimuthal_qn: int, spin_qn: float) -> Optional[str]:
    orbital = get_orbital(azimuthal_qn)
    spins = azimuthal_2_spin_dict[orbital]

    if isinstance(spins, str):
        return spins
    elif spin_qn in spins:
        return Fraction(spin_qn).__str__()

    raise Exception(f"the {orbital} orbital cannot have a js of: {spin_qn}")


def q_param(
    data_dir: Path,
    Z: int,
    principal_qn: int,
    azimuthal_qn: int,
    spin_qn: float,
    plotter: Optional[int] = 1,
) -> Optional[Tuple[List]]:
    """Gets q_params from the q_params .txt"""

    with open(data_dir / Path("q_param.txt"), "r") as param_file:
        if plotter:
            print(
                f"{constants.Z[Z]} {principal_qn}{get_orbital(azimuthal_qn)}{get_spin_as_string(azimuthal_qn,spin_qn)}"
            )

        name = f"{Z} {principal_qn}{get_orbital(azimuthal_qn)}{get_spin_as_string(azimuthal_qn,spin_qn)}"
        print(name)
        for line in param_file:
            if line.strip() == name:
                print(line.strip())
                Eb = param_file.readline()
                Erow = numpy.fromstring(param_file.readline(), int, sep=" ")
                sigma = [Erow, numpy.fromstring(param_file.readline(), float, sep=" ")]
                beta = [Erow, numpy.fromstring(param_file.readline(), float, sep=" ")]
                gamma = [Erow, numpy.fromstring(param_file.readline(), float, sep=" ")]
                delta = [Erow, numpy.fromstring(param_file.readline(), float, sep=" ")]
                print(f"beta:{beta} gamma:{gamma} delta:{delta} Eb:{Eb}")
                return (beta, gamma, delta, Eb)
        raise NotImplementedError(f"{name} not found in params.txt")


if __name__ == "__main__":
    q_param(Path("/scratch/nal89286/xsw-script/src/fpfpp/"), 7, 2, 1, 1.5)
