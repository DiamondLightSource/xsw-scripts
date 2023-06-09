from pathlib import Path
from typing import Any, Optional, Sequence

import math
import constants
import numpy as np
from q_param import q_param
from calculate_energy import calculate_energy
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

NUMPY_DTYPE = np.float32
THETAB = 90 - 4


def predict_modulation(
    data_dir: Path,
    fh0: float,
    ph0: float,
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

    if not args:
        sample = "Cu"
        atom_type = constants.Z.index(sample) + 1
        principal_qn = 1
        azimuthal_qn = 0
        spin_qn = 0.5
        alphaB = 0
        hs = np.array([1, 1, 1])
        a_lat = 3.6149
        lps0 = np.array(
            [a_lat, a_lat, a_lat, 90, 90, 90], dtype=np.float32
        )  # lattice unit cell = [a b c, alpha beta gamma]
        xyzs = np.array(
            [
                [atom_type, 0, 0, 0, 1],
                [atom_type, 0.5, 0.5, 0, 1],
                [atom_type, 0.0, 0.5, 0.5, 1],
                [atom_type, 0.5, 0.0, 0.5, 1],
            ],
            dtype=NUMPY_DTYPE,
        )
        width = 0.2
    elif len(args) == 9:
        (
            atom_type,
            principal_qn,
            azimuthal_qn,
            spin_qn,
            alphaB,
            hs,
            lps0,
            xyzs,
            width,
        ) = args
    else:
        raise Exception(
            "You must either include the Z, n, l and js of the emitter orbital, or you must edit the script to include this information"
        )

    energy = calculate_energy(lps0, hs, THETAB)
    bettab, gamtab, deltab, Eb = q_param(
        data_dir / Path("q_param.txt"),
        atom_type,
        principal_qn,
        azimuthal_qn,
        spin_qn,
        plotter,
    )

    fwhmgaus = width

    """
        Non-dipolar corrections, assuming delta=0
        %// C1=(1+Q)/(1-Q), C2=1/(1-Q), C3=phase shift
        %// Row1=Ek, Row2=No correction, Row3=C1s, Row4=N1s, Row5=O1s, Row6=Fe2p3/2
        %// Ref[1]: Trzhaskovskaya et al., Atomic Data and Nuclear Data Tables 77, 97 (2001)

        %Q is defined as the forward/backward assymetry factor and includes gamma,
        %delta, beta (delta however is set to 0 as the effect is very small !! ( ompare Ref[1]))
        %%%% for Fe2p, compared to gamma, delta is one order or magnitude less big

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %non dipolar corrections in the order: photon energy, 0 ,C1s, N1s, O1s,
        %Fe2p3/2 (delta term newly inserted on 14.07.2014 to deal correctly with Fe2p)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    QCE = 0
    the = theta * np.pi / 180  # Theta angle in FIG1 in Ref[1]
    phie = 0 * np.pi / 180  # Phi angle in FIG1 in Ref[1]

    test_E = np.arange(1500, 5001, 10)

    beta = np.interp(test_E, bettab[0, :], bettab[QCE + 1, :])
    gamma = np.interp(test_E, gamtab[0, :], gamtab[QCE + 1, :])
    delta = np.interp(test_E, deltab[0, :], deltab[QCE + 1, :])

    energy_diff = test_E - energy + Eb[QCE]
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

    hm = np.array([1, 1, 1])

    lambda_ = 12398.54 / energy  # Wavelength in A

    DWBs = 0.0  # Debye-Waller B factor, sample.
    nas = xyzs.shape[0]

    # Si monochromator
    lpm0 = np.array(
        [5.431, 5.431, 5.431, 90, 90, 90]
    )  # Lattice parameters, DW factor and atomic coordinates, Si monochromator
    DWBm = 0.0  # Debye-Waller B factor, mono.

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
        ]
    )
    # Atomic number, x, y, z, occupancy (Mono Si)

    nam = xyzm.shape[0]

    xyzm[:, 1:4] = xyzm[:, 1:4] - np.ones((nam, 1)) * 0.125
    # Shift the origin to the inversion center

    # Unit cell volumes, Bragg plane spacings, and Bragg angles
    lps = np.copy(lps0)
    lps[3:6] = lps[3:6] * np.pi / 180  # Change deg to rad, sample
    lpm = np.copy(lpm0)
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

    dhs = np.sqrt(np.sum((hs @ rlvs) ** 2)) ** (
        -1
    )  # Bragg plane spacing for hkl reflection in A, sample.

    dhm = lpm[0] / np.sqrt(
        np.sum(hm @ hm)
    )  # Bragg plane spacing for hkl reflection in A, mono Si.

    thbs = np.arcsin(dhs ** (-1) * (lambda_ / 2))  # Bragg angle in rad, sample.
    thbm = np.arcsin(dhm ** (-1) * (lambda_ / 2))  # Bragg angle in rad, mono Si.
    print(thbs)
    # z=['Ag' 'Cu' 'Si'];
    # Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No'];

    ##########################################
    # Structure factors and chi values, sample
    ##########################################

    f0 = np.loadtxt(data_dir / Path("f0_all_free_atoms.txt"))
    fps = np.zeros(nas)
    fpps = np.zeros(nas)
    fs = np.zeros(nas)
    f0s = np.zeros(nas)

    for i in range(nas):
        print(constants.Z[int(xyzs[i, 0])])
        fpfppdata = np.loadtxt(
            data_dir / Path(constants.Z[int(xyzs[i, 0])].lower() + ".nff"),
            delimiter="\t",
            skiprows=1,
            usecols=(0, 1, 2),
        )
        fps[i] = np.interp(energy, fpfppdata[:, 0], fpfppdata[:, 1]) - xyzs[i, 0]
        fpps[i] = np.interp(energy, fpfppdata[:, 0], fpfppdata[:, 2])
        fs[i] = (
            np.interp(0.5 * dhs ** (-1), f0[:, 0], f0[:, int(xyzs[i, 0]) - 1])
            + fps[i]
            + 1j * fpps[i]
        )
        f0s[i] = xyzs[i, 0] + fps[i] + 1j * fpps[i]

    hrs = xyzs[:, 1:4] @ hs
    print("//////////////////////////////////////////////////////")
    Fhs = (np.exp(2 * np.pi * 1j * hrs.T) @ (fs * xyzs[:, 4])) * np.exp(
        -DWBs / dhs**2 / 4
    )
    Fhbs = (np.exp(-2 * np.pi * 1j * hrs.T) @ (fs * xyzs[:, 4])) * np.exp(
        -DWBs / dhs**2 / 4
    )
    F0s = np.sum(f0s * xyzs[:, 4]) * np.exp(-DWBs / dhs**2 / 4)

    gams = 2.818e-5 * lambda_**2 / np.pi / ucvs
    chihs = -gams * Fhs
    chihbs = -gams * Fhbs
    chi0s = -gams * F0s

    #######################################
    # Structure factors and chi values, mono
    ########################################

    fpm = np.zeros(nam)
    fppm = np.zeros(nam)
    fm = np.zeros(nam)
    f0m = np.zeros(nam)

    for i in range(nam):
        fpfppdata = np.loadtxt(
            data_dir / Path(constants.Z[int(xyzm[i, 0])].lower() + ".nff"),
            delimiter="\t",
            skiprows=1,
            usecols=(0, 1, 2),
        )
        fpm[i] = np.interp(energy, fpfppdata[:, 0], fpfppdata[:, 1]) - xyzm[i, 0]
        fppm[i] = np.interp(energy, fpfppdata[:, 0], fpfppdata[:, 2])
        fm[i] = (
            np.interp(0.5 * dhm ** (-1), f0[:, 0], f0[:, int(xyzm[i, 0]) - 1])
            + fpm[i]
            + 1j * fppm[i]
        )
        f0m[i] = xyzm[i, 0] + fpm[i] + 1j * fppm[i]

    hrm = xyzm[:, 1:4] @ hm
    Fhm = (np.exp(2 * np.pi * 1j * hrm.T) @ (fm * xyzm[:, 4])) * np.exp(
        -DWBm / dhm**2 / 4
    )
    Fhbm = (np.exp(-2 * np.pi * 1j * hrm.T) @ (fm * xyzm[:, 4])) * np.exp(
        -DWBm / dhm**2 / 4
    )
    F0m = np.sum(f0m * xyzm[:, 4]) * np.exp(-DWBm / dhm**2 / 4)

    gamm = 2.818e-5 * lambda_**2 / np.pi / ucvm
    chihm = -gamm * Fhm
    chihbm = -gamm * Fhbm
    chi0m = -gamm * F0m

    ############################################
    # Gaussian (normalized to integrated area)
    ############################################

    areagaus = 1
    ngaus = 1010

    # if fwhmgaus > 0:
    rangegaus = 20  # (max(datar[:, 1]) - min(datar[:, 1])) * 2
    degaus = rangegaus / (ngaus - 1)
    egaus = np.arange(-np.floor((ngaus - 1) / 2), np.ceil((ngaus - 1) / 2) + 1) * degaus
    # else:
    # degaus = 0.005

    #########################
    # Sample rocking curve
    ############################
    #   df is the angle between photon incidence and the surface towards the
    #   detector
    #   di is the angle between photon incidence and the surface away from the

    di = 45 / 180 * np.pi
    df = np.pi - di
    bs = -np.sin(np.deg2rad(THETAB - alphaB)) / np.sin(np.deg2rad(THETAB + alphaB))
    Ps = 1.0
    ewidths = (
        energy * np.abs(np.real(chihs) * Ps) / np.sin(thbs) ** 2 / np.sqrt(np.abs(bs))
    )
    print(chihs)

    #########################################
    # These parameters can be played with
    ########################################
    des1 = -10 * ewidths
    des2 = 10 * ewidths
    colour = "g"

    ################################
    # End
    ################################

    nsteps = ((des2 - des1) / degaus).round(decimals=0)
    print(nsteps)
    a1 = np.arange(1, nsteps + 1)
    des = des1 + (a1 - 1) * degaus
    etas = (
        (2 * bs * des * np.sin(thbs) ** 2 / energy - chi0s * (1 - bs) / 2)
        / Ps
        / np.sqrt(np.abs(bs) * chihs * chihbs)
    )

    xs = np.zeros(int(nsteps))
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
    # rs[2754] * 10 ** 4

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
    print(chihbm.shape)

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

    xm = np.zeros(int(nstepm))
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
    # rsgm(3500)*10^4

    #########################
    # Fit rocking curve
    #########################
    escale = width
    print("##################################")
    print(energy)
    eoffset = energy  # X0[np.argmax(Y)] - 0.6
    rscale = 1  # (np.max(datar[:,1]) - 0.5 * (datar[-1,1] + datar[0,1])) / 0.9 * 3
    rbgoffset = 0  # 0.5 * (datar[-1,1] + datar[0,1]) / rscale
    rbgslope = 0  # (datar[-1,2] - datar[0,2]) / (datar[-1,1] - datar[0,1]) / rscale
    steps = 100

    # TODO look at this section again
    def f1(p):
        p = np.array(
            [
                p[0] / multiples_p[0],
                p[1] / multiples_p[1],
                p[2] / multiples_p[2],
                p[3] / multiples_p[3],
                p[4] / multiples_p[4],
            ]
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

        return e

    # Define the objective function
    def objective(p):
        return f1(p)

    # Define the scaling factors
    multiples_p = np.array([10, 1e-3, 1e-4, 1e3, 1e2])

    # Define the initial parameter values
    p0 = np.array(
        [
            escale * multiples_p[0],
            eoffset * multiples_p[1],
            rscale * multiples_p[2],
            0 * multiples_p[3],
            rbgoffset * multiples_p[4],
        ]
    )

    # Define the lower and upper bounds
    lb = np.array(
        [
            0 * multiples_p[0],
            (eoffset - 1) * multiples_p[1],
            0 * multiples_p[2],
            -1.0 * multiples_p[3],
            0 * multiples_p[4],
        ]
    )

    ub = np.array(
        [
            0.3 * multiples_p[0],
            (eoffset + 1) * multiples_p[1],
            rscale * 10 * multiples_p[2],
            1.0 * multiples_p[3],
            rbgoffset * 10 * multiples_p[4] + 0.0000000001,
        ]
    )
    print(f"{lb}:{ub}")
    # Perform the least squares optimization
    result = least_squares(objective, p0, bounds=(lb, ub), method="trf")

    # Extract the optimized parameter values
    p = result.x / multiples_p
    v = result.cost
    dedth = energy * (np.pi / 180) * np.cos(thbs) / np.sin(thbs)

    # Screen output of fitting results for rocking curve

    if plotter > 0:
        print("                      dE/dth =", dedth, "eV/deg")
        print("                Least-square =", v)
        print("fitted Gaussian width = p(1) =", p[0])
        print("        Energy offset = p(2) =", p[1])
        print("   Incident intensity = p(3) =", p[2])
        print("   R background slope = p(4) =", p[3])
        print("  R background offset = p(5) =", p[4])
        print("                        Rmax =", max(rsgm))

        titleline = (
            "Energy = "
            + str(energy)
            + " eV; mono = Si("
            + str(hm[0])
            + " "
            + str(hm[1])
            + " "
            + str(hm[2])
            + "); sample = ("
            + str(hs[0])
            + " "
            + str(hs[1])
            + " "
            + str(hs[2])
            + "); dE/dth = "
            + str(dedth)
            + " (ev/deg)"
        )
        rline1 = (
            " bs = "
            + str(bs)
            + "; bm = "
            + str(bm)
            + "; Gaussian width = "
            + str(fwhmgaus)
            + " (eV); Gaussian area = "
            + str(areagaus)
        )
        rline2 = "Rmax = " + str(max(rsgm)) + ";   I0 = " + str(p[2]) + " (cts)"

    ##########################
    # Fit XSW yield curve
    ###########################

    # datayf=datay;
    # datayf[:,1]=datay[:,1]-p[1]
    noffbragg = 5
    # datayf[:,2]=datay[:,2]*2*noffbragg/np.sum(datay[[range(noffbragg),range(-noffbragg,0)]][:,1])
    # X=datayf[:,1]
    # Y=datayf[:,2]

    # lb = [0,0,0]
    # ub = [4,2,2]
    # step = 50
    # res = np.zeros((2*step, step))
    # for nn in range(2*step):
    #     for ll in range(step):
    #         fh = nn/step
    #         ph = ll/step
    #         res[nn,ll] = np.sum(np.abs(f2([1,fh,ph])))

    q0 = np.array([1, fh0, ph0])

    # TODO where if f2 defined

    q = np.array([1, fh0, ph0])

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

    def f2(q):
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
        print("               Least-square =", v2)
        print("                  fH = q[1] =", q[1])
        print("                  PH = q[2] =", q[2])
        print("    Non-dipolar corrections used:")
        print("           C1 = (1+Q)/(1-Q) =", C1)
        print("           C2 =   1/(1-Q)   =", C2)
        print("           C3 = phase shift =", C3)

    # Clear the current figure
    plt.clf()

    # Create a figure
    fig = plt.figure()

    # Set figure properties
    # fig.set_figwidth(12)
    # fig.set_figheight(6)

    # Create subplot 1
    plt.subplot(1, 2, 1)

    # Set subplot properties
    # ax = plt.gca()
    # ax.set_xlim([min(datarf[:, 0]), max(datarf[:, 0])])
    # ax.set_ylim([-0.05, max(datayf[:, 1]) * 1.1])

    # Plot the data
    # plt.plot(datarf[:, 0], datarf[:, 1], '-k')
    # plt.plot(datayf[:, 0], datayf[:, 1], '-k')
    # plt.plot(desgm, rsgm, '-r')
    # plt.plot(desgm, ysgm, '-r')

    # Set font properties
    # ax.set_xlabel('E - E_Bragg (eV)', fontsize=10)
    # ax.set_ylabel('Intensity', fontsize=10)
    # ax.tick_params(axis='both', which='major', labelsize=8)

    # Create subplot 2
    plt.subplot(1, 2, 2)

    # Set subplot properties
    # ax = plt.gca()
    # ax.set_xlim([min(datarf[:, 0]), max(datarf[:, 0])])
    # ax.set_ylim([-0.05, max(datayf[:, 1]) * 1.1])

    # Plot the data
    # plt.plot(datarf[:, 0], datarf[:, 1], '-k')
    # plt.plot(datayf[:, 0], datayf[:, 1], '-k')
    # plt.plot(desgm, rsgm, '-r')
    # plt.plot(desgm, ysgm, '-r')

    # Set font properties
    # ax.set_xlabel('E - E_Bragg (eV)', fontsize=10)
    # ax.set_ylabel('Intensity', fontsize=10)
    # ax.tick_params(axis='both', which='major', labelsize=8)

    # Adjust the spacing between subplots
    # plt.subplots_adjust(wspace=0.3)

    # Show the plot
    plt.show()

    # Create subplot 2
    plt.subplot(1, 2, 2)

    # Set subplot properties
    # ax = plt.gca()
    # ax.margins = [0.01, 0.0, 0.1, 0.15]

    # Plot dummy data
    # plt.plot([0, 1], [0, 1], '-k')
    # plt.axis('off')

    # Set text properties
    # ax.text(0.02, 0.05 + 7 * 0.1, datafile2, fontsize=8)
    # ax.text(0.02, 0.05 + 6 * 0.1, titleline, fontsize=8)
    # ax.text(0.02, 0.05 + 5 * 0.1, rline1, fontsize=8)
    # ax.text(0.06, 0.05 + 4 * 0.1, rline2, fontsize=8)
    # ax.text(0.06, 0.05 + 3 * 0.1, rline3, fontsize=8)
    # ax.text(0.03, 0.05 + 2 * 0.1, yline1, fontsize=8)
    # ax.text(0.03, 0.05 + 1 * 0.1, yline2, fontsize=8)
    # ax.text(0.06, 0.05 + 0 * 0.1, yline3, fontsize=8)

    # Set font size
    # ax.tick_params(axis='both', which='major', labelsize=6)

    # Adjust the spacing between subplots
    # plt.subplots_adjust(wspace=0.3)

    # Show the plot
    plt.show()

    ##################################
    # Save the data, fit and figure
    ##################################
    # Calculate desg and desgm
    desg = des[0] - (rangegaus + egaus[0]) + a1 * degaus
    desgm = desg[0] - (np.max(dem) - cfwhmm) + a1 * degaus

    # Uncomment the following lines if needed
    # ifit = np.where((desgm >= np.min(datarf[:, 1])) & (desgm <= np.max(datarf[:, 1])))[0]
    # ifit = np.concatenate(([np.min(ifit) - 1], ifit, [np.max(ifit) + 1]))
    # etadata = (2 * bs * datarf[:, 1] * np.sin(thbs) ** 2 / energy - chi0s * (1 - bs) / 2) / Ps / np.sqrt(np.abs(bs) * chihs * chihbs)
    # etafit = (2 * bs * desgm[ifit] * np.sin(thbs) ** 2 / energy - chi0s * (1 - bs) / 2) / Ps / np.sqrt(np.abs(bs) * chihs * chihbs)
    # e2th = 1 / (energy * 1e-6 * np.cos(thbs) / np.sin(thbs))
    # dataout = np.column_stack((np.real(etadata), datarf, datayf[:, 2]))
    # fitout = np.column_stack((np.real(etafit), desgm[ifit], rsgm[ifit], ysgm[ifit]))
    # datatitle = 'eta E-Eb(eV) R_data Y_data'
    # fittitle = 'eta E-Eb(eV) R_fit Y_fit'
    # fprintfMat(dataoutfile, dataout, '%.6f', datatitle)
    # fprintfMat(fitoutfile, fitout, '%.6f', fittitle)
    # xs2pdf(0, figurefile)

    #########################################
    ########   Alternative Plotter   ########
    #########################################

    if plotter == 1:
        plt.figure(100)
        plt.clf()
        plt.figure(100)

        # Set background color (optional)
        # plt.gca().set_facecolor((1, 1, 1))

        # Limit the plot range on the x-axis (auto on the y-axis)

        # Plot XSW profiles (relative to Bragg reflection)

        # Plot fits (relative to Bragg reflection)
        plt.plot(desgm, rsga, color="k", linewidth=3)
        plt.plot(desgm, ysgm, color="r", linewidth=3)

        # Label plots
        plt.xlabel("E - E_Bragg (eV)", fontsize=20, fontweight="bold", color="k")
        plt.ylabel("Relative Absorption", fontsize=20, fontweight="bold", color="k")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()
        # plt.text(0.6, 1,
        #  '                 Filename: ' + dataprefix + '\n\n' +
        #  'General Information: ' + '\n' +
        #  '   Energy = ' + str(energy) + ' eV,      mono = Si(' + str(hm[0]) + ' ' + str(hm[1]) + ' ' + str(hm[2]) + ')\n' +
        #  '   sample = ' + samp + '(' + str(hs[0]) + ' ' + str(hs[1]) + ' ' + str(hs[2]) + '),     dE/dth = ' + str(dedth) + ' (ev/deg)\n' +
        #  'Fitting the Bulk Reflection Profile: ' + '\n' +
        #  '   bs = ' + str(bs) + ',       bm = ' + str(bm) + '\n' +
        #  '   Gaussian width = ' + str(fwhmgaus) + ' (eV),   Gaussian area = ' + str(areagaus) + '\n' +
        #  '   Rmax = ' + str(np.max(rsgm)) + ',                 I0 = ' + str(p[2]) + '(cts)\n' +
        #  'Fitting the XSW Absorption Curve: ' + '\n' +
        #  '   fH = ' + str(q[1]) + '\n' +
        #  '   PH = ' + str(q[2]) + '\n' +
        #  'Non-dipolar corrections used: ' + '\n' +
        #  '   C1 = (1+Q)/(1-Q) = ' + str(C1) + '\n' +
        #  '   C2 = 1/(1-Q) = ' + str(C2) + '\n' +
        #  '   C3 = phase shift = ' + str(C3) + '\n' +
        #  '   Q =  ' + str(Q) + '\n',
        #  fontsize=10, verticalalignment='top', horizontalalignment='left', fontfamily='monospace')

        plt.axis([-3, 5, -0.1, 3.5])
        # plt.savefig(dataprefix + '.pdf')
        # plt.savefig(dataprefix + '.svg')
        plt.show()

    the_out = np.column_stack((desgm, ysgm))
    the_nix_out = np.column_stack((desgm, rsga))
    fit_out = [the_out, the_nix_out]
    q = q[1:]

    ######################################
    #####   write out data      ##########
    ######################################

    # filename = input("Save Fit Data As: ")

    # # Coherent fraction
    # coherent_fraction = q[0]
    # np.savetxt(filename, ["CoherentFraction"], delimiter='', newline='\r\n', fmt='%s')
    # np.savetxt(filename, [coherent_fraction], delimiter='', newline='\r\n', fmt='%.10f', append=True)

    # # Coherent position
    # coherent_position = q[1]
    # np.savetxt(filename, ["CoherentPosition"], delimiter='', newline='\r\n', fmt='%s', append=True)
    # np.savetxt(filename, [coherent_position], delimiter='', newline='\r\n', fmt='%.10f', append=True)

    # # RelativePhotonEnergy XSWyield XSWreflection
    # np.savetxt(filename, ["RelativePhotonEnergy XSWyield XSWreflection"], delimiter='', newline='\r\n', fmt='%s', append=True)
    # combine = np.column_stack((datarf[:,0], datayf[:,1], datarf[:,1]))
    # np.savetxt(filename, combine, delimiter='\t', newline='\r\n', fmt='%.10f', append=True)

    # # RelativePhotonEnergy XSWyieldFit
    # np.savetxt(filename, ["RelativePhotonEnergy XSWyieldFIT"], delimiter='', newline='\r\n', fmt='%s', append=True)
    # combine = np.column_stack((desgm, ysgm))
    # np.savetxt(filename, combine, delimiter='\t', newline='\r\n', fmt='%.10f', append=True)

    # # RelativePhotonEnergy XSWreflectionFit
    # np.savetxt(filename, ["RelativePhotonEnergy XSWreflectionFIT"], delimiter='', newline='\r\n', fmt='%s', append=True)
    # combine = np.column_stack((desgm, rsgm))
    # np.savetxt(filename, combine, delimiter='\t', newline='\r\n', fmt='%.10f', append=True)
