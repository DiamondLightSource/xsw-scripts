from pathlib import Path
from typing import Any, Optional, Tuple

from . import constants
import matplotlib.pyplot as plt
import numpy as np
from .calculate_energy import calculate_energy
from numpy.typing import NDArray
from .q_param import q_param


def predict_modulation(
    data_dir: Path,
    coherent_fraction: float,
    coherent_position: float,
    theta: float,
    plotter: bool,
    *,
    atom_type: int = 8,
    principal_qn: int = 1,
    azimuthal_qn: int = 0,
    spin_qn: float = 0.5,
    alphaB: float = 0,
    hkl_index: Tuple[int, ...] = (1, 1, 1),
    lps0: Optional[NDArray] = None,
    xyzs: Optional[NDArray] = None,
    width: float = 0.2,
    thetaB: float = 90 - 4,
    qce: float = 0,
    dwb_sample: float = 0,
    lpm0: Optional[NDArray] = None,
    xyzm: Optional[NDArray] = None,
    dwb_mono: float = 0,
    hm: Tuple[int, ...] = (1, 1, 1),
) -> Tuple[NDArray, ...]:
    """calcuates XSW absorption profile from inputted parameters.

    Args:
        datadir: folder containing the structure factors (*.nff files)
            and f0_all_free_atoms.txt
        coherent_fraction: coherent fraction
        coherent_position: coherent position
        theta: angle between the measured emission direction and the photon
            polarisation (for most I09 stuff this is 18)
        plotter: If True plots figures and outputs text to the command line
        atom_type: Z value of the emitter atom (e.g. 8 for oxygen)
        principal_qn: principle quantum number, n, for photoelectron emitting orbital
        azimuthal_qn: azimuthal quantum number, l, for photoelectron emitting orbital
        spin_qn: total spin, js = l+ms, for photoelectron emitting orbital
        alphaB: angle between the scattering plane and the surface, usually 0
        hkl_index: (h,k,l) index of the reflection, e.g. [1,1,1] for the (111)
        lps0: lattice unit cell = [a b c, alpha beta gamma] in Å and °
        xyzs: position of atoms in the unit cell in fractional coordinates and an occupational factor (usually 1) in the format [Z, x, y, z, f],
        width: gaussian broadening, models the experimental
              broadening due to imperfections in the monochromator (pretty small)
              and the sample substrate (significantly larger). Numbers between
              0.1 and 0.5 eV are common, 0.3 is broadly the mean
        thetaB: Bragg angle
        qce: Analyser-beam geometry ?
        dwb_sample: Debye-Waller B factor, sample
        hm: Mono Si reflection,
        dwb_mono = Debye-Waller B factor, mono.
        lpm0: Lattice parameters for  Si monochromator
        xyzm: Atomic coordinates, for Si monochromator
        dwb_mono: float = 0,
    """
    if lps0 is None:
        a_lat = 3.6149
        lps0 = np.array(
            [a_lat, a_lat, a_lat, 90, 90, 90], dtype=np.float64
        )  # lattice unit cell = [a b c, alpha beta gamma]

    if xyzs is None:
        xyzs = np.array(
            [
                [atom_type, 0, 0, 0, 1],
                [atom_type, 0.5, 0.5, 0, 1],
                [atom_type, 0.0, 0.5, 0.5, 1],
                [atom_type, 0.5, 0.0, 0.5, 1],
            ],
            dtype=np.float64,
        )

    # Si monochromator
    if lpm0 is None:
        lpm0 = np.array(
            [5.431, 5.431, 5.431, 90, 90, 90], dtype=np.float64
        )  # Lattice parameters, DW factor and atomic coordinates, Si monochromator

    if xyzm is None:
        xyzm = np.array(
            [
                [14, 0, 0, 0, 1],
                [14, 0.5, 0.5, 0, 1],
                [14, 0.5, 0, 0.5, 1],
                [14, 0, 0.5, 0.5, 1],
                [14, 0.25, 0.25, 0.25, 1],
                [14, 0.75, 0.75, 0.25, 1],
                [14, 0.75, 0.25, 0.75, 1],
                [14, 0.25, 0.75, 0.75, 1],
            ],
            dtype=np.float64,
        )
        # Atomic number, x, y, z, occupancy (Mono Si)

    energy = calculate_energy(lps0.copy(), hkl_index, thetaB)
    bettab, gamtab, deltab, Eb = q_param(
        data_dir / Path("q_param.txt"),
        atom_type,
        principal_qn,
        azimuthal_qn,
        spin_qn,
        plotter,
        dtype=np.float64,
    )

    fwhmgaus = width

    # ###################################################################
    # Non-dipolar corrections, assuming delta=0
    #  C1=(1+Q)/(1-Q), C2=1/(1-Q), C3=phase shift
    #  Row1=Ek, Row2=No correction, Row3=C1s, Row4=N1s, Row5=O1s, Row6=Fe2p3/2
    #  Ref[1]: Trzhaskovskaya et al., Atomic Data and Nuclear Data Tables 77, 97 (2001)
    #
    #  Q is defined as the forward/backward assymetry factor and includes gamma,
    #  delta, beta (delta however is set to 0 as the effect is very small !! ( ompare Ref[1]))
    #  for Fe2p, compared to gamma, delta is one order or magnitude less big
    #
    #  Non dipolar corrections in the order: photon energy, 0 ,C1s, N1s, O1s,
    #  Fe2p3/2 (delta term newly inserted on 14.07.2014 to deal correctly with Fe2p)

    # Analyser-beam geometry
    the = theta * np.pi / 180  # Theta angle in FIG1 in Ref[1]
    phie = 0 * np.pi / 180  # Phi angle in FIG1 in Ref[1]

    test_E = np.arange(1500, 5001, 10)

    beta = np.interp(test_E, bettab[0, :], bettab[qce + 1, :])
    gamma = np.interp(test_E, gamtab[0, :], gamtab[qce + 1, :])
    delta = np.interp(test_E, deltab[0, :], deltab[qce + 1, :])

    energy_diff = test_E - energy + Eb[qce]
    min_index = np.argmin(np.abs(energy_diff))
    bet, gam, del_ = beta[min_index], gamma[min_index], delta[min_index]

    Q = (
        (del_ * np.sin(the) * np.cos(phie))
        + (gam * np.cos(phie) * np.sin(the) * np.cos(the) ** 2)
    ) / (
        1 + bet * 0.5 * (3 * np.cos(the) ** 2 - 1)
    )  # new Q-value with delta terms to cope with Fe2p properly

    C1 = (1 + Q) / (1 - Q)  # C 1s at 30 deg, 2.63keV
    C2 = 1 / (1 - Q)
    C3 = 0

    wavelength = 12398.54 / energy  # Wavelength in A

    nas = xyzs.shape[0]

    nam = xyzm.shape[0]
    xyzm[:, 1:4] = xyzm[:, 1:4] - np.ones((nam, 3)) * 0.125
    # Shift the origin to the inversion center

    # Unit cell volumes, Bragg plane spacings, and Bragg angles
    lps = lps0
    lps[3:6] = lps[3:6] * np.pi / 180  # Change deg to rad, sample
    lpm = lpm0
    lpm[3:6] = lpm[3:6] * np.pi / 180  # Change deg to rad, mono
    ucvs = (
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
    )
    # Unit cell volume in A^3, sample
    ucvm = lpm[0] * lpm[1] * lpm[2]  # Unit cell volume in A^3, mono Si

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
        ],
        dtype=np.float64,
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
            ],
            dtype=np.float64,
        )
        / ucvs
    )

    dhs = np.sqrt(np.sum((hkl_index @ rlvs) ** 2)) ** (
        -1
    )  # Bragg plane spacing for hkl reflection in A, sample.

    dhm = lpm[0] / np.sqrt(
        np.sum(hm @ np.array(hm).T)
    )  # Bragg plane spacing for hkl reflection in A, mono Si.

    thbs = np.arcsin(dhs ** (-1) * (wavelength / 2))  # Bragg angle in rad, sample.
    thbm = np.arcsin(dhm ** (-1) * (wavelength / 2))  # Bragg angle in rad, mono Si.

    ##########################################
    # Structure factors and chi values, sample
    ##########################################

    f0 = np.loadtxt(data_dir / Path("f0_all_free_atoms.txt"))
    fps = np.zeros(nas)
    fpps = np.zeros(nas)
    fs = np.zeros(nas, dtype=np.complex64)
    f0s = np.zeros(nas, dtype=np.complex64)

    for i in range(nas):
        fpfppdata = np.loadtxt(
            data_dir / Path(constants.Z[int(xyzs[i, 0]) - 1].lower() + ".nff"),
            delimiter="\t",
            skiprows=1,
            usecols=(0, 1, 2),
        )

        fps[i] = np.interp(energy, fpfppdata[:, 0], fpfppdata[:, 1]) - xyzs[i, 0]
        fpps[i] = np.interp(energy, fpfppdata[:, 0], fpfppdata[:, 2])
        fs[i] = (
            np.interp(0.5 * dhs ** (-1), f0[:, 0], f0[:, int(xyzs[i, 0]) - 2])
            + fps[i]
            + 1j * fpps[i]
        )
        f0s[i] = xyzs[i, 0] + fps[i] + 1j * fpps[i]

    hrs = xyzs[:, 1:4] @ np.array(hkl_index).T

    Fhs = (np.exp(2 * np.pi * 1j * hrs.T) @ (fs * xyzs[:, 4])) * np.exp(
        -dwb_sample / dhs**2 / 4
    )
    Fhbs = (np.exp(-2 * np.pi * 1j * hrs.T) @ (fs * xyzs[:, 4])) * np.exp(
        -dwb_sample / dhs**2 / 4
    )
    F0s = np.sum(f0s * xyzs[:, 4]) * np.exp(-dwb_sample / dhs**2 / 4)

    gams = 2.818e-5 * wavelength**2 / np.pi / ucvs
    chihs = -gams * Fhs
    chihbs = -gams * Fhbs
    chi0s = -gams * F0s

    #######################################
    # Structure factors and chi values, mono
    ########################################

    fpm = np.zeros(nam)
    fppm = np.zeros(nam)
    fm = np.zeros(nam, dtype=np.complex64)
    f0m = np.zeros(nam, dtype=np.complex64)

    for i in range(nam):
        fpfppdata = np.loadtxt(
            data_dir / Path(constants.Z[int(xyzm[i, 0]) - 1].lower() + ".nff"),
            delimiter="\t",
            skiprows=1,
            usecols=(0, 1, 2),
        )
        fpm[i] = np.interp(energy, fpfppdata[:, 0], fpfppdata[:, 1]) - xyzm[i, 0]
        fppm[i] = np.interp(energy, fpfppdata[:, 0], fpfppdata[:, 2])
        fm[i] = (
            np.interp(0.5 * dhm ** (-1), f0[:, 0], f0[:, int(xyzm[i, 0]) - 2])
            + fpm[i]
            + 1j * fppm[i]
        )
        f0m[i] = xyzm[i, 0] + fpm[i] + 1j * fppm[i]

    hrm = xyzm[:, 1:4] @ np.array(hm).T
    Fhm = (np.exp(2 * np.pi * 1j * hrm.T) @ (fm * xyzm[:, 4])) * np.exp(
        -dwb_mono / dhm**2 / 4
    )
    Fhbm = (np.exp(-2 * np.pi * 1j * hrm.T) @ (fm * xyzm[:, 4])) * np.exp(
        -dwb_mono / dhm**2 / 4
    )
    F0m = np.sum(f0m * xyzm[:, 4]) * np.exp(-dwb_mono / dhm**2 / 4)

    gamm = 2.818e-5 * wavelength**2 / np.pi / ucvm
    chihm = -gamm * Fhm
    chihbm = -gamm * Fhbm
    chi0m = -gamm * F0m

    ############################################
    # Gaussian (normalized to integrated area)
    ############################################

    areagaus = 1
    ngaus = float(1010)

    # if fwhmgaus > 0:
    rangegaus = 20  # (max(datar[:, 1]) - min(datar[:, 1])) * 2
    degaus = rangegaus / (ngaus - 1)
    egaus = (
        np.arange(-round((ngaus - 1) / 2) - 1, ngaus - round((ngaus - 1) / 2) - 1)
        * degaus
    )
    # else:
    # degaus = 0.005
    #########################
    # Sample rocking curve
    ############################

    bs = -np.sin(np.deg2rad(thetaB - alphaB)) / np.sin(np.deg2rad(thetaB + alphaB))
    Ps = 1.0
    ewidths = (
        energy * np.abs(np.real(chihs) * Ps) / np.sin(thbs) ** 2 / np.sqrt(np.abs(bs))
    )

    #########################################
    # These parameters can be played with
    ########################################
    des1 = -10 * ewidths
    des2 = 10 * ewidths
    ################################
    # End
    ################################

    nsteps = ((des2 - des1) / degaus).round(decimals=0)

    a1 = np.arange(1, nsteps + 1)
    des = des1 + (a1 - 1) * degaus
    etas = (
        (2 * bs * des * np.sin(thbs) ** 2 / energy - chi0s * (1 - bs) / 2)
        / Ps
        / np.sqrt(np.abs(bs) * chihs * chihbs)
    )

    xs = np.zeros(int(nsteps), dtype=np.complex64)
    mask_pos = np.real(etas) >= 0
    mask_neg = np.real(etas) < 0

    xs[mask_pos] = (
        np.sqrt(np.abs(bs))
        * (np.sqrt(chihs * chihbs) / chihbs)
        * (etas[mask_pos] - np.sqrt(etas[mask_pos] ** 2 - 1))
    )
    xs[mask_neg] = (
        np.sqrt(np.abs(bs))
        * (np.sqrt(chihs * chihbs) / chihbs)
        * (etas[mask_neg] + np.sqrt(etas[mask_neg] ** 2 - 1))
    )

    rs = np.abs(xs) ** 2

    ######################################################
    # Convolution 1 (sample rocking curve  gaussian)
    ########################################################

    ######################
    # Mono rocking curve
    ########################

    bm = -1
    Pm = 1.0
    ewidthm = (
        energy * np.abs(np.real(chihm) * Pm) / np.sin(thbm) ** 2 / np.sqrt(np.abs(bm))
    )

    dem1 = -4 * ewidthm
    dem2 = 10 * ewidthm
    nstepm = ((dem2 - dem1) / degaus).round()
    a1 = np.arange(1, nstepm + 1)
    dem = dem2 - (a1 - 1) * degaus  # Inverted for convolution
    etam = (
        (2 * bm * dem * np.sin(thbm) ** 2 / energy - chi0m * (1 - bm) / 2)
        / Pm
        / np.sqrt(np.abs(bm) * chihm * chihbm)
    )

    xm = np.zeros(int(nstepm), dtype=np.complex64)
    mask_pos = np.real(etam) >= 0
    mask_neg = np.real(etam) < 0

    xm[mask_pos] = (
        np.sqrt(np.abs(bm))
        * (np.sqrt(chihm * chihbm) / chihbm)
        * (etam[mask_pos] - np.sqrt(etam[mask_pos] ** 2 - 1))
    )
    xm[mask_neg] = (
        np.sqrt(np.abs(bm))
        * (np.sqrt(chihm * chihbm) / chihbm)
        * (etam[mask_neg] + np.sqrt(etam[mask_neg] ** 2 - 1))
    )

    rm = np.abs(xm) ** 2

    rmn = (rm * rm) / np.sum(rm * rm)
    cfwhmm = 0.5 * (
        dem[np.max(np.where(rm >= (np.max(rm) / 2)))]
        + dem[np.min(np.where(rm >= (np.max(rm) / 2)))]
    )

    ###########################################################################
    # Convolution 2 (mono rocking curve * (sample rocking curve * gaussian))
    #############################################################################

    rsgm = np.convolve(rmn, rs, mode="full")

    #########################
    # Fit rocking curve
    #########################
    escale = width
    eoffset = energy  # X0[np.argmax(Y)] - 0.6
    rscale = 1  # (np.max(datar[:,1]) - 0.5 * (datar[-1,1] + datar[0,1])) / 0.9 * 3
    rbgoffset = 0  # 0.5 * (datar[-1,1] + datar[0,1]) / rscale

    # TODO This closure is the first place to look for bugs
    def f1(p: NDArray) -> Tuple[NDArray, ...]:
        p = np.array(
            [
                p[0] / multiples_p[0],
                p[1] / multiples_p[1],
                p[2] / multiples_p[2],
                p[3] / multiples_p[3],
                p[4] / multiples_p[4],
            ],
            dtype=np.float64,
        )

        fwhmgaus = p[0]
        if fwhmgaus > 0:
            gaus = np.exp(-4 * np.log(2) * (egaus / fwhmgaus) ** 2)
            gaus = gaus / np.sum(gaus)
            gaus = gaus * areagaus
            a1 = np.arange(1, nsteps + ngaus)
            desg = des[0] - (rangegaus + egaus[0]) + a1 * degaus
            rsga = np.convolve(gaus, rsgm)
        else:
            desg = des
            rsga = rsgm

        a1 = np.arange(1, nstepm + desg.shape[0])
        desgm = desg[0] - (np.max(dem) - cfwhmm) + a1 * degaus

        e = desgm

        return e, a1, rsga

    # Define the scaling factors
    multiples_p = np.array([10, 1e-3, 1e-4, 1e3, 1e2], dtype=np.float64)

    # Define the initial parameter values
    p0 = np.array(
        [
            escale * multiples_p[0],
            eoffset * multiples_p[1],
            rscale * multiples_p[2],
            0 * multiples_p[3],
            rbgoffset * multiples_p[4],
        ],
        dtype=np.float64,
    )

    p = p0
    v, a1, rsga = f1(p0)
    p = np.array(
        [
            p[0] / multiples_p[0],
            p[1] / multiples_p[1],
            p[2] / multiples_p[2],
            p[3] / multiples_p[3],
            p[4] / multiples_p[4],
        ],
        dtype=np.float64,
    )
    dedth = energy * (np.pi / 180) * np.cos(thbs) / np.sin(thbs)

    # Screen output of fitting results for rocking curve

    if plotter > 0:
        print(
            f"                      dE/dth ={ dedth},eV/deg \n"
            f"                Least-square = {v} \n"
            f"fitted Gaussian width = p(1) = {p[0]} \n"
            f"        Energy offset = p(2) = {p[1]} \n"
            f"   Incident intensity = p(3) = {p[2]} \n"
            f"   R background slope = p(4) = {p[3]} \n"
            f"  R background offset = p(5) = {p[4]} \n"
            f"                        Rmax = {max(rsgm)} \n"
        )

    ##########################
    # Fit XSW yield curve
    ###########################

    q0 = np.array([1, coherent_fraction, coherent_position], dtype=np.float64)
    q = np.array([1, coherent_fraction, coherent_position], dtype=np.float64)

    ys = rs * C1 + 2 * C2 * q[1] * np.sqrt(rs) * np.cos(
        C3 + np.arctan2(np.imag(xs), np.real(xs)) - 2 * np.pi * q[2]
    )
    gaus = np.exp(-4 * np.log(2) * (egaus / fwhmgaus) ** 2)
    gaus = gaus / np.sum(gaus)
    gaus = gaus * areagaus

    if fwhmgaus > 0:
        ysg = np.convolve(gaus, ys, mode="full")
    else:
        ysg = ys
    ysgm = np.convolve(rmn, ysg, mode="full") + 1

    def f2(q: NDArray[np.float64]) -> NDArray[np.float64]:
        ys = rs * C1 + 2 * C2 * q[1] * np.sqrt(rs) * np.cos(
            C3 + np.arctan2(np.imag(xs), np.real(xs)) - 2 * np.pi * q[2]
        )
        if fwhmgaus > 0:
            ysg = np.convolve(gaus, ys)
        else:
            ysg = ys

        ysgm = np.convolve(rmn, ysg) + 1

        e = ysgm

        return e

    rsga = np.convolve(gaus, rsgm)

    v2 = f2(q0)

    # Screen output of fitting results for yield
    if plotter > 0:
        print(
            f"          Least-square = {v2} \n"
            f"          fH = q[1] = {q[1]} \n"
            f"          PH = q[2] = {q[2]} \n"
            f"Non-dipolar corrections used: \n"
            f"      C1 = (1+Q)/(1-Q) = {C1}! \n"
            f"      C2 =   1/(1-Q)   = {C2} \n"
            f"      C3 = phase shift = {C3}"
        )

    ##################################
    # Save the data, fit and figure
    ##################################
    # Calculate desg and desgm
    desg = des[0] - (rangegaus + egaus[0]) + a1 * degaus
    desgm = desg[0] - (np.max(dem) - cfwhmm) + a1 * degaus

    # ########################################
    # #######   Alternative Plotter   ########
    # ########################################

    if plotter == 1:
        plt.figure()

        # Plot XSW profiles (relative to Bragg reflection)
        # Plot fits (relative to Bragg reflection)
        plt.plot(desgm, rsga, color="k", linewidth=3)
        plt.plot(desgm, ysgm, color="r", linewidth=3)

        # Label plots
        plt.xlabel("E - E_Bragg (eV)", fontsize=20, fontweight="bold", color="k")
        plt.ylabel("Relative Absorption", fontsize=20, fontweight="bold", color="k")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.axis([-3, 5, -0.1, 3.5])
        plt.show()
        # plt.savefig(dataprefix + '.pdf')
        # plt.savefig(dataprefix + '.svg')

    the_out = np.column_stack((desgm, ysgm))
    the_nix_out = np.column_stack((desgm, rsga))
    fit_out = (the_out, the_nix_out)
    # q = q[1:]

    # #####################################
    # ####   write out data      ##########
    # #####################################

    # filename = input("Save Fit Data As: ")

    # # Coherent fraction
    # coherent_fraction = q[0]
    # np.savetxt(filename, ["CoherentFraction"], \
    # delimiter='', newline='\r\n', fmt='%s')
    # np.savetxt(filename, [coherent_fraction], \
    # delimiter='', newline='\r\n', fmt='%.10f', append=True)

    # # Coherent position
    # coherent_position = q[1]
    # np.savetxt(filename, ["CoherentPosition"], delimiter='', \
    # newline='\r\n', fmt='%s', append=True)
    # np.savetxt(filename, [coherent_position], delimiter='', \
    # newline='\r\n', fmt='%.10f', append=True)

    # # RelativePhotonEnergy XSWyield XSWreflection
    # np.savetxt(filename, ["RelativePhotonEnergy XSWyield XSWreflection"],\
    #  delimiter='', newline='\r\n', fmt='%s', append=True)
    # combine = np.column_stack((datarf[:,0], datayf[:,1], datarf[:,1]))
    # np.savetxt(filename, combine, delimiter='\t', newline='\r\n', \
    # fmt='%.10f', append=True)

    # # RelativePhotonEnergy XSWyieldFit
    # np.savetxt(filename, ["RelativePhotonEnergy XSWyieldFIT"], \
    # delimiter='', newline='\r\n', fmt='%s', append=True)
    # combine = np.column_stack((desgm, ysgm))
    # np.savetxt(filename, combine, delimiter='\t', newline='\r\n',\
    #  fmt='%.10f', append=True)

    # # RelativePhotonEnergy XSWreflectionFit
    # np.savetxt(filename, ["RelativePhotonEnergy XSWreflectionFIT"], \
    # delimiter='', newline='\r\n', fmt='%s', append=True)
    # combine = np.column_stack((desgm, rsgm))
    # np.savetxt(filename, combine, delimiter='\t', newline='\r\n', \
    # fmt='%.10f', append=True)

    return fit_out
