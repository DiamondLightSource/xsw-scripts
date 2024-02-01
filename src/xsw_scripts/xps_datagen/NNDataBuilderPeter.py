import copy
import math
import random

import matplotlib.pyplot as plt  # ## COMMENTED OUT FOR CLUSTER
import numpy as np
from scipy import interpolate as interp
from scipy.signal import convolve as conv

# gen_cs
#
# Uses random number generation to create a 2D array of parameters, 'cs' (values in the interval (0, 1)), which will be
# used to build XP spectra.
#
# Each parameter, after being appropriately scaled, will eventually be used to describe specific features of the spectra
# - e.g. number of peaks, peak positions, widths, the noise intensity etc. The array columns correspond to separate
# spectra, while the rows correspond to specific parameter types. Since the number of peaks is also a variable, this
# leads to empty positions in the array when there are fewer peaks than the maximum number allowed (usually 6, but this
# can be varied). These spaces are filled with 0s. Usually there are 46 parameters per spectrum - 6 parameters for each
# peak (6*6 = 36) plus 10 more to describe the background, noise, energy range and number of datapoints.
#
# Arguments:
#
#   num_spectra:
#       The number of spectra to be generated (i.e. the number of columns in the array).
#
#   seed: (= None)
#       The seed value for the random number generator - this allows us to regenerate the same set of spectra at a
#       later date if we want to compare/check something.
#
#   num_peaks: (= None)
#       The number of peaks we want the spectra to contain. If this value is an integer in the interval
#       [1, max_num_peaks] then all the spectra will have this number of peaks. If the value is not an integer, or is
#       outide the interval, then the number of peaks will be uniformally randomised (within the interval) for each
#       spectrum (this is the default behaviour).
#
#   max_num_peaks: (= 6)
#       The maximum number of peaks permitted per spectrum (see above).
#
#   edge_marg: (= 0.05)
#       The closest that peak centres may approach the edges of the spectra, as a fraction of the spectra's lengths.
#
#   w_sep: (= 1)
#       A multiplicative factor which controls how closely any 2 peaks may approach one another, as a function of their
#       width parameters (summed in quadrature). Higher values mean peaks are forced further apart and vis versa.
#
#   min_allowed_int: (= 4)
#       The minimum allowed ratio between the peak intensities and the standard deviation (~intensity) of the noise.
#
#   min_slope: (= -0.5)
#       By default, spectra can be generated with backgrounds that slope positive or negative (from -0.5 to +0.5, in
#       increasing binding energy). In real spectra, positive slopes are more common, so this parameter allows you to
#       exclude what you consider to be excessively negative slopes. (It applies only to the 1st order coefficient of
#       the background polynomial).
#
#   cutoff_higher: (= 0.95)
#       Probability that the number of peaks in a spectrum is one higher than expected, this is an attempt to force the
#       network to prefer to identify intense shoulders over weak isolated peaks
#
# Outputs:
#
#   cs (normalised):
#       An n*m array of parameters describing m spectra with n parameters each.
#
# Anatomy of the outputted cs array:
#   Each column is a spectrum, each row corresponds to a feature of the spectra. Unless otherwise stated, the values are
#   random variables in the interval (0, 1).
#
#   [0] number of peaks (immediately converted to an integer in the interval [1, max_num_peaks])
#   [1] standard deviation of the normally distributed noise
#   [2] seed for generating the noise
#   [3] binding energy range of the spectrum
#   [4] number of datapoints in the spectrum
#
#   [5] 0th order coefficient of the background polynomial
#   [6] 1st order coefficient of the background polynomial
#   [7] 2nd order coefficient of the background polynomial
#   [8] 3rd order coefficient of the background polynomial
#   [9] 4th order coefficient of the background polynomial
#
#   [10, 16, 22, 28, 34, 40] peak energies - constrained by edge_marg (e.g. in interval (0.05, 0.95))
#   [11, 17, 23, 29, 35, 41] peak gaussian widths - constrained to (0.05, 1) to avoid excessively narrow peaks
#   [12, 18, 24, 30, 36, 42] peak lorentzian widths - constrained to (0.05, 1) to avoid excessively narrow peaks
#   [13, 19, 25, 31, 37, 43] peak asymmetries
#   [14, 20, 26, 32, 38, 44] peak step intensity coefficients
#   [15, 21, 27, 33, 39, 45] peak intensities (defined here as maximum height)
#
#   As you can see, the array columns are set up so that all the variables of a given peak are adjacent to one another:
#
#   [ *10 non-peak parameters* | 1 1 1 1 1 1 | 2 2 2 2 2 2 | 3 3 3 3 3 3 | 4 4 4 4 4 4 | 5 5 5 5 5 5 | 6 6 6 6 6 6 ]
#
#   When max_num_peaks is greater (or less) than 6, this pattern can then be easily extended (or shortened).


def gen_cs(
    num_spectra,
    seed=None,
    num_peaks=None,
    max_num_peaks=6,
    edge_marg=0.05,
    w_sep=0.75,
    min_allowed_int=4,
    min_slope=-0.5,
    cutoff_higher=0.95,
):
    if isinstance(num_peaks, int) and (num_peaks > max_num_peaks):
        print(
            "Requested number of peaks exceeds maximum allowed, generating {} instead.".format(
                max_num_peaks
            )
        )
    # Notifies you if you request more peaks than you set as a limit.

    if isinstance(seed, int):
        random.seed(seed)
    # Sets up the random number generator with the seed, if you provided one.

    num_cs = max_num_peaks * 6 + 10
    # Number of parameters to generate per spectrum (number of rows, depends on maximum number of peaks).

    cs = np.array(
        [[random.uniform(0, 1) for m in range(num_spectra)] for n in range(num_cs)],
        dtype="float32",
    )
    # Generates an n*m array of parameters (random variables in the interval (0, 1)) where m is the number of spectra
    # (columns) and n is the number of parameters per spectrum (rows).

    if num_peaks in range(1, max_num_peaks + 1):
        # if (randHigher < cutoff_higher) or (num_peaks==max_num_peaks):
        nums_peaks = cs[0, :] = np.full(num_spectra, num_peaks)
        #    # Override the generated value for number of peaks with the stipulated value num_peaks, if it is an integer in
        #    # [1, max_num_peaks].

    else:
        nums_peaks = cs[0, :] = np.floor(cs[0, :] * max_num_peaks + 1)
    # Converts the parameters corresponding to number of peaks into the actual integer numbers of peaks (0.598 -> 3,
    # 0.002 -> 1 etc.), if num_peaks doesn't satisfy the above conditions.

    e_ind = np.array(range(10, num_cs, 6))  # peak energy (position) indices
    gw_ind = np.array(range(11, num_cs, 6))  # gaussian width indices
    lw_ind = np.array(range(12, num_cs, 6))  # lorentzian width indices
    asym_ind = np.array(range(13, num_cs, 6))  # peak asymmetry parameter indices
    step_ind = np.array(range(14, num_cs, 6))  # peak step coefficient indices
    i_ind = np.array(range(15, num_cs, 6))  # peak intensity indices
    # Assigning parameters to specific peak features.

    indices = (e_ind, i_ind, gw_ind, lw_ind, asym_ind, step_ind)
    # Bundling together for easier iteration later.

    i = 0
    # Initialise iterator for while loop. The purpose of this loop is to go through the spectra and make some
    # preliminary rescales and adjustments, as well as to weed out and correct 'unacceptable' spectra. Specifically,
    # those in which peaks overlap too much so as to be indistinguishable, and those in which the peak intensities are
    # too small compared to the noise.
    done_before = 0
    while i < num_spectra:
        # Loop over all the spectra.
        randHigher = random.uniform(0, 1)
        # chance that a spectrum has more peaks than expected

        cs[e_ind, i] = cs[e_ind, i] * (1 - 2 * edge_marg) + edge_marg
        # Rescale peak energies so they don't approach within edge_marg of the spectra's boundaries. (At this stage,
        # energies are expressed as fractions of the total binding energy range in the spectrum).

        # cs[gw_ind, i] = cs[gw_ind, i]
        # cs[lw_ind, i] = cs[lw_ind, i]
        # Gaussian and lorentzian widths rescaled to interval (0.00, 1)

        if (
            (randHigher > cutoff_higher)
            and (cs[0, i] < max_num_peaks)
            and (done_before == 0)
        ):
            cs[0, i] = num_peaks + 1
            # if randHigher is greater than the cutoff_higher value, then set the number of peaks to be one higher than expected
        for ind in indices:
            cs[ind[int(cs[0, i]) :], i] = 0
            # Set all parameters for absent peaks to 0.

        if cs[0, i] > 1:
            # Checking peak positions relative to one another, so ignore 1-peak spectra.

            e_range = cs[3, i] * 45 + 5
            # Generate the binding energy range, between 5 and 50 eV, from the (0, 1) parameter. We're generating this
            # energy range early so we can see to-scale how closely the peaks approach each other.

            order = np.argsort(cs[e_ind[: int(cs[0, i])], i])
            # Get a a list of the peak binding energy indices from the current array slice ordered according to
            # increasing binding energy.

            for ind in indices:
                cs[ind[: int(cs[0, i])], i] = cs[ind[order], i]
                # Rearrange peak ordering in the array column so that lowest binding energy is first and highest is
                # last.

            for j in range(1, int(cs[0, i])):
                # Looping over pairs of peaks in the spectrum.

                e1 = cs[e_ind[j - 1], i] * e_range
                e2 = cs[e_ind[j], i] * e_range
                # Get energies of the 2 peaks (rescaled from fractional to eVs relative to low BE edge).

                gw1 = cs[gw_ind[j - 1], i] * (e_range / 4 - 0.1) + 0.1
                gw2 = cs[gw_ind[j], i] * (e_range / 4 - 0.1) + 0.1
                lw1 = cs[lw_ind[j - 1], i] * (e_range / 4 - 0.1) + 0.1
                lw2 = cs[lw_ind[j], i] * (e_range / 4 - 0.1) + 0.1
                # Get gaussian and lorentzian widths of the 2 peaks.

                e_thresh = (
                    e1 + np.sqrt(gw1**2 + lw1**2 + gw2**2 + lw2**2) * w_sep
                )
                # Define e_thresh as the energy of the first peak plus the quadrature sum of both peaks' width
                # parameters, multiplied by w_sep (default value of 0.75).

                if e2 < e_thresh:
                    # If the second peak has an energy less than e_thresh, it is considered to be too close to the
                    # first peak - this spectrum is unacceptable.

                    for k in range(1, num_cs):
                        csNumPeakHold = cs[0, i]
                        cs[k, i] = random.uniform(0, 1)
                        # Regenerate all parameters for this spectrum (except number of peaks) i.e. 'try again'.
                    done_before = 1
                    i -= 1
                    break
                    # Decrement iterator and break out of loop over peak pairs - rescalings and overlap checks will be
                    # performed again. This repeats until an acceptable spectrum is produced.

            else:
                # If the previous loop completes without failure, this block executes.
                done_before = 0
                max_int = max(cs[i_ind, i])
                # Get the most intense peak's intensity.

                for l in range(int(nums_peaks[i])):
                    # Iterate over all peaks in the spectrum

                    i_thresh = max(
                        0.1 * min_allowed_int * max_int * cs[1, i], 0.05 * max_int
                    )
                    # Define i_thresh, the minimum allowed intensity, as being the largest of either min_allowed_int
                    # times the standard deviation of the noise, or 5% the intensity of the largest peak.

                    if cs[i_ind[l], i] < i_thresh:
                        cs[i_ind[l], i] = random.uniform(i_thresh, max_int)
                        # If a peak's intensity is smaller than i_thresh, change it to a random value between i_thresh
                        # and the intensity of the largest peak. (Slight bug in that max_int might be less than
                        # i_thresh - this system isn't perfect in a few ways...)
                order = np.argsort(-cs[i_ind[: int(nums_peaks[i])], i])
            # Get a a list of the peak intensities indices from the current array slice ordered according to
            # increasing intensity (was fomerly binding energy).

            # for ind in indices:
            #    cs[ind[:int(nums_peaks[i])], i] = cs[ind[order], i]
            # Rearrange peak ordering in the array column so that highest intensity is first. Demarcated as, with
            #   a committee approach to the fitting, if multiple members disagreed on what is the second / third highest
            #   peak, caused a significant uncertainty in the fitting.

        i += 1
        # Increment iterator at end of loop - if peak overlap check failed, then the value will end up being the same as
        # it was at the beginning of the step - i.e. reset to try again after regenerating this spectrum.

    cs[6, :] = -(min_slope - 0.5) * cs[6, :] + min_slope + 0.5
    # Rescale the 1st order background polynomial coefficient so that when it is transformed in rescale_cs the resulting
    # minimum possible value is min_slope.

    return cs


# rescale_cs
#
# Takes in the array of parameters produced by gen_cs and rescales them so that the they are physically realistic.
#
# Note: while there is indeed some rescaling done in gen_cs, this is limited to cases where the rescaled values are
# still within the interval (0, 1) (with the exception of number of peaks). The significance of this is that neural
# networks are best trained on data that is normalised, so it is useful to keep a separate copy of the parameters in
# their 'primordial' form.
#
# Arguments:
#
#   cs_in:
#       The n*m array of (normalised) parameters produced by gen_cs.
#
# Outputs:
#
#   cs (rescaled):
#       An n*m array of parameters describing m spectra with n parameters each. The parameters are now physically
#       realistic.


def rescale_cs(cs_in):
    cs = copy.copy(cs_in)
    # Creates a copy of the inputted array so that we don't overwrite it with the rescaled values.

    num_cs = cs.shape[0]
    # Get the number of parameters describing each spectrum (number of rows).

    e_ind = np.array(range(10, num_cs, 6))
    gw_ind = np.array(range(11, num_cs, 6))
    lw_ind = np.array(range(12, num_cs, 6))
    asym_ind = np.array(range(13, num_cs, 6))
    step_ind = np.array(range(14, num_cs, 6))
    i_ind = np.array(range(15, num_cs, 6))
    # Assigning parameters to peak features (as before).

    cs[3, :] = e_range = cs[3, :] * 45 + 5
    # Rescale binding energy range to the interval (5 eV, 50 eV).

    cs[e_ind, :] = cs[e_ind, :] * e_range
    # Convert all peak energies in all spectra from fractional to (relative) binding energy in eV.

    cs[gw_ind, :] = cs[gw_ind, :] * (e_range / 4 - 0.1) + 0.1
    cs[lw_ind, :] = cs[lw_ind, :] * (e_range / 4 - 0.1) + 0.1
    # Rescale peak width parameters so that they occupy a reasonable range (~ 0.1 eV - 1/4th of total energy range eV)

    cs[asym_ind, :] *= 0.25
    cs[step_ind, :] *= 0.25
    # Rescale asymmetry parameters and step function intensity coefficients to the interval (0, 0.25).

    cs[i_ind, :] = cs[i_ind, :] * 1e9
    # Rescale intensity values to a realistic order of magnitude.

    cs[5, :] = (cs[5, :] - 0.5) * 1e9
    cs[6, :] = (cs[6, :] - 0.5) * 1e8
    cs[7, :] = (cs[7, :] - 0.5) * 1e6
    cs[8, :] = (
        cs[8, :] - 0.5
    ) * 1e2  # 3rd and 4th order coefficients are relatively very small.
    cs[9, :] = (cs[9, :] - 0.5) * 1e-1
    # Rescale background polynomial coefficients to realistic orders of magnitude, and to be both positive and negative.

    cs[1, :] = (
        cs[1, :] * np.array([max(ints) for ints in cs[i_ind, :].transpose()]) * 0.1
    )
    # Noise standard deviation is at most 1/10 the size of the largest peak intensity.

    cs[4, :] = (cs[4, :] * 1950 + 50).astype("int")
    # Rescale the parameter corresponding to number of datapoints to an integer in the interval [50, 2000].

    return cs


# build_spectra
#
# Takes in the array of appropriately scaled parameters from rescale_cs and uses them to render spectra.
#
# Note that spectra are rendered with binding energy increasing left to right (inverted compared to the standard
# presentation of XPS) as an increasing x-axis is more convenient.
#
# Arguments:
#
#   cs:
#       An n*m array of parameters describing m spectra with n parameters each.
#
#   report_step: (= None)
#       Controls the frequency of progress printouts during spectra rendering. If set to None, False or 0, there will be
#       no printouts (default behaviour).
#
#   plot: (= False)
#       If True, images of the first 10 spectra will be drawn as they're rendered, underlayed with the individual peaks
#       as well as the background. This is useful for quickly visually inspecting the results of any changes made to the
#       generation and scaling steps.
#
#   dark: (= False)
#       Whether to use light or dark theme for spectra plotting (default is light).
#
# Outputs:
#
#   spectra:
#       A list containing the rendered spectra. List elements are pairs of numpy arrays, one containing the binding
#       energy scale and the other containing the corresponding intensities. The spectra will all have different lengths
#       (number of datapoints) at this stage.


def build_spectra(cs, report_step=None, plot=False, dark=False):
    def doniach_sunjic(_e_scale, _energy, _width, _asym, _intens):
        # _e_scale: The array of energy values over which the function is to be drawn (the x-axis).
        # _energy: The energy (centroid) of the peak.
        # _width: Parameter related to the width of the peak.
        # _asym: Parameter describing how asymmetrical the peak is.
        # _intens: The intensity of the peak (defined here as the maximum height).

        arctan = np.arctan((_e_scale - _energy) / _width)
        numer = np.cos(0.5 * np.pi * _asym + (1 - _asym) * arctan)
        denom = (_width**2 + (_e_scale - _energy) ** 2) ** (0.5 * (1 - _asym))
        quotient = numer / denom
        _lineshape = _intens * quotient / max(quotient)

        return _lineshape

    # Implementation of the Doniach-Sunjic lineshape (see REF) (intensity is defined here as the peak maximum).

    def gaussian(_e_scale, _energy, _width, _intens):
        # _e_scale: The array of energy values over which the function is to be drawn (the x-axis).
        # _energy: The energy (centre) of the peak.
        # _width: Parameter related to the width of the peak (the standard deviation).
        # _intens: The intensity of the peak (defined here as the maximum height).

        _lineshape = _intens * np.exp(-((_e_scale - _energy) ** 2) / (2 * _width**2))

        return _lineshape

    # Implementation of the Gaussian lineshape (intensity is defined here as the peak maximum).

    def step_func(_e_scale, _energy, _width, _intens):
        # _e_scale: The array of energy values over which the function is to be drawn (the x-axis).
        # _energy: The energy (centre) of the peak.
        # _width: Parameter related to the width of the peak (the standard deviation).
        # _intens: The intensity of the peak (defined here as the maximum height).

        _lineshape = (
            0.5
            * _intens
            * (
                np.array(
                    [math.erf((x - _energy) / _width) for x in _e_scale],
                    dtype="float32",
                )
                + 1
            )
        )

        return _lineshape

    # Implementation of the step function (based on the Gaussian error function)

    cs = cs.transpose()
    # Transpose parameter array so that rows are now spectra and columns are different parameter types. This is done for
    # efficiency's sake - it is faster to iterate over the 0th dimension of a numpy array. Speed is more of a concern in
    # this function as this is the slowest part of the spectra generation process, whereas the other orientation was
    # used previously out of convenience for elegant indexing etc.

    num_spectra = cs.shape[0]
    # Get the number of spectra (number of rows).

    num_cs = cs.shape[1]
    # Get the number of parameters describing each spectrum (number of columns).

    e_ind = np.array(range(10, num_cs, 6))
    gw_ind = np.array(range(11, num_cs, 6))
    lw_ind = np.array(range(12, num_cs, 6))
    asym_ind = np.array(range(13, num_cs, 6))
    step_ind = np.array(range(14, num_cs, 6))
    i_ind = np.array(range(15, num_cs, 6))
    # Assigning parameters to peak features (as before).

    spectra = [0] * num_spectra
    # Initialise the list that will contain the rendered spectra.

    if dark:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    # Set up plotting style as light or dark.

    colors = (
        "royalblue",
        "dodgerblue",
        "c",
        "mediumseagreen",
        "forestgreen",
        "limegreen",
    )
    # Plotting colours for each of the 6 possible peaks in the spectrum.

    for i in range(num_spectra):
        # Iterate over all spectra.

        e_range = cs[i, 3]
        # Get the binding energy range for this spectrum in eV from the parameter array.
        e_scale, e_step = np.linspace(
            0, e_range, int(cs[i, 4]), retstep=True, dtype="float32"
        )
        # Create the binding energy scale from the binding energy range and the number of datapoints (cs[i, 4]). This
        # also provides us with the energy step between datapoints.

        # We're going to be separately rendering the Doniach-Sunjic and Gaussian components of each peak and then taking
        # the convolution. In order to get full convolutions over our given binding energy scale, we need to carry them
        # out over a larger range than this. We extend the binding energy scale equally to the left and right so that
        # the resulting scale is 4 times longer:

        e_low_ext = np.flip(
            np.arange(e_scale[0], -1.5 * e_range, -e_step, dtype="float32"), axis=0
        )
        e_high_ext = np.arange(e_scale[-1], 2.5 * e_range, e_step, dtype="float32")
        # Create extentions of the binding energy scale in the negative and positive directions.

        e_scale_long = np.concatenate((e_low_ext[:-1], e_scale, e_high_ext[1:]))
        # Stitch together the extended energy scale from the 3 fragments - note that the last value of the left
        # extention and the first value of the right extention are omitted since they overlap with the terminal values
        # of the central energy scale.

        len_e = len(e_scale)
        len_low = len(e_low_ext)
        # Get the lengths of the low energy scale extention and the original unextended energy scale.

        background = np.array(
            cs[i, 5]
            + cs[i, 6] * e_scale
            + cs[i, 7] * e_scale**2
            + cs[i, 8] * e_scale**3
            + cs[i, 9] * e_scale**4,
            dtype="float32",
        )
        # Render the base layer of the background as a 4th order polynomial (higher order coefficients are small).

        # for j in range(int(cs[i, 0])):
        #
        #     energy = cs[i, e_ind[j]]
        #     ds_width = cs[i, lw_ind[j]]
        #     g_width = cs[i, gw_ind[j]]
        #     intens = cs[i, i_ind[j]]
        #     step_coef = cs[i, step_ind[j]]
        #
        #     step = step_coef * step_func(e_scale, energy, max(g_width, ds_width), intens)
        #
        #     background += step

        lineshape = copy.copy(background)
        # Initialise the lineshape to which the rest of the spectrum's components will be added. Make as a copy of the
        # background so that the latter can be kept and separately plotted.

        for j in range(int(cs[i, 0])):
            # Iterate over all peaks in the current spectrum.

            energy = cs[i, e_ind[j]]
            ds_width = cs[i, lw_ind[j]]
            g_width = cs[i, gw_ind[j]]
            asym = cs[i, asym_ind[j]]
            intens = cs[i, i_ind[j]]
            step_coef = cs[i, step_ind[j]]
            # Get the peak's parameters from the parameter array.

            step = step_coef * step_func(
                e_scale, energy, max(g_width, ds_width), intens
            )
            # Render the peak's step. Use the broadest of the 2 width components.

            background += step
            # Add the step to the background.

            ds_peak = doniach_sunjic(-e_scale_long, -energy, ds_width, asym, intens)
            # Render the peak's Doniach-Sunjic component over the extended energy scale, in preparation for convolution.

            g_peak = gaussian(e_scale_long, energy, g_width, intens)
            # Render the peak's Gaussian component over the extended energy scale, in preparation for convolution.

            conv_peak = conv(ds_peak, g_peak)[::2]
            # Take the convolution of the two peak components. The discrete convolution function from scipy ends up
            # doubling the number of datapoints - we correct for this by taking every second value.

            conv_peak = conv_peak[len_low : len_low + len_e] / max(conv_peak) * intens
            # Extract the central part of the convolved peak corresponding to the original, unextended energy scale.
            # Then, rescale the peak intensity (maximum height) to the value from the parameter array.

            if plot and (i < 10):
                # If 'plot' is True and less than 10 spectra have been rendered so far, begin plotting this spectrum.

                plt.figure(i)
                # Initialise the plot with a figure label corresponding to the spectrum number.

                plt.fill_between(
                    e_scale,
                    background,
                    conv_peak + background,
                    color=colors[j],
                    edgecolor="k",
                    label="Peak {}".format(j + 1),
                    zorder=j + 1,
                )
                # Plot the current peak and its step with a block colour above the background.

            lineshape += conv_peak + step
            # Add the peak and its step to the overall accumulating lineshape.

        # random.seed(cs[i, 2])
        # Set up the random number generator for the noise with the noise seed.

        noise = np.array([random.gauss(0, 1) for x in range(len_e)]) * cs[i, 1]
        # Generate normally distributed noise and then scale with noise intensity parameter.

        lineshape += noise
        # Add noise to the overall lineshape. Congratulations. You have made an XP spectrum.

        if plot and (i < 10):
            # If 'plot' is True and less than 10 spectra have been rendered so far, continue plotting this spectrum.

            plt.plot(
                e_scale,
                background,
                color="orange",
                linewidth=4,
                label="Background",
                zorder=14,
            )
            # Plot the background polynomial with a nice big thiccc orange line. It will line up nicely with the bottom
            # of the peak blocks.

            if dark:
                color = "w"
            else:
                color = "k"
            # Set overall lineshape colour as white or black depending on whether the background is dark or light.

            plt.plot(
                e_scale, lineshape, color=color, linewidth=1, label="Result", zorder=15
            )
            # Plot the overall lineshape.

            plt.xlim(np.asarray(plt.xlim())[[1, 0]])
            # Reverse the direction of the x-axis (binding energy) so that it is decreasing from left to right - this
            # presents the spectrum in the canonical orientation.

            plt.xlabel(
                "Relative Binding Energy / eV",
                fontdict={"family": "sans", "size": "16"},
            )
            plt.ylabel(
                "Intensity (Arbitrary)", fontdict={"family": "sans", "size": "16"}
            )
            # Set the axis labels so they are neat and clear.

            plt.yticks([])
            # Remove y-axis ticks since intensity is ultimately arbitrary.

            plt.legend(edgecolor="k", framealpha=0)
            # Add a legend so everything is clear. :)

            # plt.tight_layout()
            # Figure borders tight to plot for neatness.

        spectra[i] = [e_scale, lineshape]
        # Add the binding energy scale and completed lineshape to the output spectra list.

        if report_step:
            if not (i + 1) % int(report_step):
                print("Completed {} of {} spectra.".format(i + 1, num_spectra))
                # If the current spectrum number is exactly divisible by /report_step/, print out the number of spectra
                # rendered so far.

    return spectra


# resize_spectra
#
# Resizes and rescales the spectra produced by 'build_spectra' so that they are ready to be fed to a neural network for
# training or inference.
#
# The spectra must be rescaled in two ways: First (although this is actually done second), the intensity values are
# rescaled so that in every spectrum the minimum is 0 and the maximum is 1. This makes learning much easier by putting
# all the data on the same scale. Second, the spectra must all have the same number of datapoints as neural networks
# have set dimensions. The spectra are thus interpolated to a set number of datapoints (200). This resizing must also
# be carried out for any real data being fed to the networks at test time. One might wonder why we bother generating a
# variable number of datapoints in the first place. The reason is that it is necessary to ensure that the network is
# exposed to the varied effects of interpolation from a variable number of datapoints, as these will be present in real
# data.
#
# Arguments:
#
#   spectra:
#       The list of raw spectra produced by 'build_spectra'.
#
#   num_points (= 200):
#       The number of datapoints to interpolate the spectra to. (Dictated by the input size of the networks).
#
#   spline_order: (= 1)
#       The order of the polynomials used in interpolating the spectra - 1 = linear, 3 = cubic etc.
#
#   minmax: (= True)
#       Whether or not to scale the intensities to the interval (0, 1) (default behaviour is to rescale).
#
# Outputs:
#
#   rescaled_spectra:
#       An n*m array containing n rescaled spectra each comprised of m datapoints (only the intensity values).
#
#   e_minmax:
#       An n*2 array containing the original minimum and maximum binding energy values from the spectra.


def resize_spectra(spectra, num_points=200, spline_order=1, minmax=True):
    num_spectra = len(spectra)
    # Get the number of spectra.

    resized_spectra = [0] * num_spectra
    # Initialise the list that will contain the resized spectra.

    e_minmax = [0] * num_spectra
    # Initialise the list that will contain the minimum and maximum binding energies of each spectrum.

    for i in range(num_spectra):
        # Iterate over all spectra

        energy = spectra[i][0]
        # Get the spectrum's binding energy scale.

        intensity = spectra[i][1]
        # get the spectrum's intensity scale.

        e_min, e_max = min(energy), max(energy)
        # Get the spectrum's minimum and maximum binding energy values.

        ent = np.linspace(e_min, e_max, num_points)
        # Define the new num_points-length energy scale to which the spectrum will be interpolated.

        tck = interp.splrep(energy, intensity, k=spline_order, s=0)
        # Get the spline representation of the spectrum, (t, c, k). t is the vector of knots, c contains the B-spline
        # coefficients and k is the order of the spline.

        resized_spectra[i] = interp.splev(ent, tck, der=0)
        # Generate the interpolated intensity scale and add to the list of rescaled spectra. Note that we are no longer
        # including the binding energy scale.

        e_minmax[i] = (e_min, e_max)
        # Save the minimum and maximum binding energies. The whose binding energy scale can be regenerated from these
        # later if it's needed.

    resized_spectra = np.array(resized_spectra, dtype="float32")
    # Convert to array after all spectra have been added.

    if minmax:
        resized_spectra = minmax_scale(resized_spectra, two_d=True)
        # If minmax is True, rescale all the spectra so that their intensities range from 0 to 1.

    return resized_spectra, np.array(e_minmax, dtype="float32")


# minmax_scale
#
# Rescales a list or array of values so that they range from 0 to 1. Very simplistic implementation - can either take a
# 1D array or list and rescale its values, or can take a 2D array or list of lists (not necessarily all the same length)
# and rescale each one individually (i.e. along 1st axis (not 0th)).
#
# Arguments:
#
#   x:
#       The list or array containing the series of values to be rescaled.
#
#   two_d: (= False)
#       Whether the list/array itself is to be rescaled (False, default), or contains a series of arrays/lists each of
#       which is to be rescaled (True).
#
# Outputs:
#
#   y:
#       An array containing the rescaled values.


def minmax_scale(x, two_d=False):
    if not two_d:
        y = np.array(x)
        y = y - np.min(y)
        y = y / np.max(y)
        # Rescale a 1D list of numbers to range from 0 to 1.

    else:
        y = []
        # Initialise the list that will contain the rescaled arrays.

        for series in x:
            # Get each list or array to be rescaled.

            if not isinstance(series, np.ndarray):
                series = np.array(series)
                # If it is a list (or tuple), convert it to a numpy array.

            series = series - np.min(series)
            series = series / np.max(series)
            y.append(series)
            # Append the rescaled array to the output.

    return np.array(y, dtype="float32")
