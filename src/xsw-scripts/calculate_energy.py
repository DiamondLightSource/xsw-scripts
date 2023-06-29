import numpy as np
from constants import ANG, CSPEED, EJ, H
from numpy.typing import NDArray


def calculate_energy(
    lps: NDArray[np.float64], hkl_plane: NDArray[np.float64], Bragg_angle: float
) -> float:
    """Calculates the energy of the lattice unit cell

    Args:
        lps: lattice unit cell
        hkl_plane: hkl plane
        bragg_angle: bragg angle
    Returns:
        float: energy calculated
    """
    lps[3:6] = lps[3:6] * np.pi / 180
    ucvs: float = (
        lps[0]
        * lps[1]
        * lps[2]
        * np.sqrt(
            1
            - np.cos(lps[3]) ** 2
            - np.cos(lps[4]) ** 2
            - np.cos(lps[5]) ** 2
            + 2 * np.cos(lps[3]) * np.cos(lps[4]) * np.cos(lps[5])
        )
    )  # Unit cell volume in A^3, sample.

    lvs = np.array(
        [
            [lps[0], 0, 0],
            [lps[1] * np.cos(lps[5]), lps[1] * np.sin(lps[5]), 0],
            [
                lps[2] * np.cos(lps[4]),
                lps[2]
                * (np.cos(lps[3]) - np.cos(lps[4]) * np.cos(lps[5]))
                / np.sin(lps[5]),
                lps[2]
                * np.sqrt(
                    1
                    - np.cos(lps[3]) ** 2
                    - np.cos(lps[4]) ** 2
                    - np.cos(lps[5]) ** 2
                    + 2 * np.cos(lps[3]) * np.cos(lps[4]) * np.cos(lps[5])
                )
                / np.sin(lps[5]),
            ],
        ]
    )
    # Real space lattice vectors a, b, and c in Cartesian coordinates
    # with a parallel to X and b in the XY plane

    rlvs = (
        np.array(
            [
                [
                    lvs[1, 1] * lvs[2, 2] - lvs[1, 2] * lvs[2, 1],
                    lvs[1, 2] * lvs[2, 0] - lvs[1, 0] * lvs[2, 2],
                    lvs[1, 0] * lvs[2, 1] - lvs[1, 1] * lvs[2, 0],
                ],
                [
                    lvs[2, 1] * lvs[0, 2] - lvs[2, 2] * lvs[0, 1],
                    lvs[2, 2] * lvs[0, 0] - lvs[2, 0] * lvs[0, 2],
                    lvs[2, 0] * lvs[0, 1] - lvs[2, 1] * lvs[0, 0],
                ],
                [0, 0, lps[0] * lps[1] * np.sin(lps[5])],
            ]
        )
        / ucvs
    )

    rlv: np.ndarray = np.dot(hkl_plane, rlvs)

    dhkl: float = np.sqrt(np.sum(rlv**2)) ** (-1)  # 1/sqrt(sum(abs(plane2*rlvs).^2))

    lam: float = 2 * dhkl * ANG * np.sin(np.deg2rad(Bragg_angle))

    return H * CSPEED / lam / EJ
