import math
from numpy import float64
from numpy.typing import NDArray

from typing import Tuple, List

def calculate_energy(lps0: List[float], hkl: Tuple[int], Bragg_angle: float) -> float:
    plane1 = hkl

    h: float = 6.63e-34
    cspeed: float = 3e8
    eJ: float = 1.6e-19
    ang: float = 1e-10

    lps = lps0
    lps[0][3:6] = [x * math.pi / 180 for x in lps[0][3:6]]
    ucvs: float = (
        lps[0]
        * lps[1]
        * lps[2]
        * math.sqrt(
            1
            - math.cos(lps[3]) ** 2
            - math.cos(lps[4]) ** 2
            - math.cos(lps[5]) ** 2
            + 2 * math.cos(lps[3]) * math.cos(lps[4]) * math.cos(lps[5])
        )
    )  # Unit cell volume in A^3, sample.

    lvs: list[list[float]] = [
        [lps[0], 0, 0],
        [lps[1] * math.cos(lps[5]), lps[1] * math.sin(lps[5]), 0],
        [
            lps[2] * math.cos(lps[4]),
            lps[2]
            * (math.cos(lps[3]) - math.cos(lps[4]) * math.cos(lps[5]))
            / math.sin(lps[5]),
            lps[2]
            * math.sqrt(
                1
                - math.cos(lps[3]) ** 2
                - math.cos(lps[4]) ** 2
                - math.cos(lps[5]) ** 2
                + 2 * math.cos(lps[3]) * math.cos(lps[4]) * math.cos(lps[5])
            )
            / math.sin(lps[5]),
        ],
    ]  # Real space lattice vectors a, b, and c in Cartesian coordinates with a parallel to X and b in the XY plane

    rlvs: list[list[float]] = [
        [
            lvs[1][1] * lvs[2][2] - lvs[1][2] * lvs[2][1],
            lvs[1][2] * lvs[2][0] - lvs[1][0] * lvs[2][2],
            lvs[1][0] * lvs[2][1] - lvs[1][1] * lvs[2][0],
        ],
        [
            lvs[2][1] * lvs[0][2] - lvs[2][2] * lvs[0][1],
            lvs[2][2] * lvs[0][0] - lvs[2][0] * lvs[0][2],
            lvs[2][0] * lvs[0][1] - lvs[2][1] * lvs[0][0],
        ],
        [0, 0, lps[0] * lps[1] * math.sin(lps[5])],
    ] / ucvs

    rlv: list[float] = [
        sum([plane1[i] * rlvs[i][j] for i in range(len(plane1))])
        for j in range(len(rlvs[0]))
    ]

    dhkl: float = math.sqrt(sum((rlv[i] ** 2 for i in range(len(rlv))))) ** (
        -1
    )  # 1/sqrt(sum(abs(plane2*rlvs).^2))

    lam: float = 2 * dhkl * ang * math.sin(math.radians(Bragg_angle))
    energy: float = h * cspeed / lam / eJ

    return energy
