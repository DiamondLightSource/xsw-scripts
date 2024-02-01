import copy

import matplotlib.pyplot as plt  ### COMMENTED OUT FOR CLUSTER
from . import NNDataBuilderPeter as nnD

import h5py  ### COMMENTED OUT FOR CLUSTER
import tkinter as tk  ### COMMENTED OUT FOR CLUSTER
from tkinter.filedialog import askopenfilenames as aof  ### COMMENTED OUT FOR CLUSTER
import numpy as np

# SpectrumDataset
#
# This object serves as a container for spectra (both generated and real) together with all their labels and other
# information, for when any training or testing is done in-memory (and for data inspection).
#
# Initialisation Arguments:
#
#   data_from: (= 'gen')
#       Determines whether data is to be generated or real data is to be imported from nexus files, or neither. In the
#       last case, all the attributes of the dataset are initialised as NoneTypes.
#
#   args & kwargs:
#       These are the arguments passed to either the data generation function or the nexus data importing function.
#
# Attributes of the initialised SpectrumDataset:
#
#   cs_norm:
#       The 2D array of normalised parameters describing all the spectra in the dataset initially generated in 'gen_cs'
#       in NNDataBuilderPeter. For imported data it is set to NoneType. 1st axis, rather than 0th, spans spectra.
#
#   cs:
#       The 2D array of rescaled parameters describing all the spectra in the dataset. These parameters are physcially
#       realistic, and are used directly to render spectra. For imported data it is set to NoneType. 1st axis, rather
#       than 0th, spans spectra.
#
#   seed:
#       The seed value that was used to generate the data in this dataset. For imported data it is set to NoneType.
#
#   inputs:
#       A 2D array containing rescaled intensity scales for all the spectra (set number of datapoints, intensities range
#       from 0 to 1 in each spectrum). Called 'inputs' because these are also the input vectors for the neural networks.
#
#   e_minmax:
#       Stored maximum and minimum binding energy values from the spectra before they were resized and rescaled.
#
#   num_spectra:
#       The number of spectra contained in the dataset.
#
#   num_points:
#       The number of datapoints the spectra have been resized to.
#
#   peak_count_OH:
#       A 2D array of one-hot vectors for the numbers of peaks in each spectrum (see 'get_categorical_count'). NoneType
#       for imported data, optional for generated.
#
#   peak_pos_OH_concat:
#       A 2D array of one-hot vectors for the peak positions in each spectrum (vectors for each peak in a spectrum are
#       concatenated to one longer vector, see 'get_categorical_pos'). NoneType for imported data, optional for
#       generated.
#
#   peak_pos_MH:
#       A 2D array of multi-hot vectors for the peak positions in each spectrum (see 'get_categorical_pos'). Nonetype
#       for imported data, optional for generated.
#
# Note: At the moment a  standard dictionary object could be used to hold data rather than this thing, but early on when
# things were messier I used to assign certain methods to these datasets as well, which required a class definition.


class SpectrumDataset:
    def __init__(self, data_from="gen", *args, **kwargs):
        if data_from in ("generate", "gen"):
            gen_dataset(self, *args, **kwargs)
            # Generate data.

        elif data_from in ("nexus", "real", "nxs"):
            get_real_spectra(self, *args, **kwargs)
            # Retrieve real data from nexus files.

        else:
            self.cs_norm = None
            self.cs = None
            self.seed = None
            self.inputs = None
            self.e_minmax = None
            self.num_spectra = None
            self.num_points = None
            self.peak_count_OH = None
            self.peak_pos_OH_concat = None
            self.peak_pos_MH = None
            # Neither generate nor import data, instead initialise all attributes as NoneTypes (this is variously useful
            # for manipulating datasets - stitching them together, extracting subsets etc. - as will be seen).


# gen_dataset
#
# Generates data and appends to an empty SpectrumDataset container object. Essentially wraps everything in
# NNDataBuilderPeter into one function call.
#
# Arguments:
#
#   dataset:
#       The SpectrumDataset container object into which the generated data will be placed.
#
# ############# The following argument descriptions are copy/pasted from functions in 'NNDataBuilderPeter' #############
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
#   report_step: (= None)
#       Controls the frequency of progress printouts during spectra rendering. If set to None, False or 0, there will be
#       no printouts (default behaviour).
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
# ######################################################################################################################
#
#   pos_oh (= True):
#       Whether to append 'one-hot' categorical labels for the peak positions in the spectra (explained in
#       'get_categorical_pos').
#
#   pos_mh (= True):
#       Whether to append 'multi-hot' categorical labels for the peak positions in the spectra (explained in
#       'get_categorical_pos').
#  cutoff_higher (= 0.95):
#       percentage chance that the number of peaks in the calculation will actually be greater than num_peaks


def gen_dataset(
    dataset,
    num_spectra,
    seed=None,
    num_peaks=None,
    max_num_peaks=6,
    peak_marg=0.05,
    w_sep=1,
    min_allowed_int=4,
    min_slope=-0.5,
    report_step=None,
    num_points=200,
    spline_order=1,
    minmax=True,
    pos_oh=True,
    pos_mh=True,
    cutoff_higher=0.95,
):
    if type(report_step) == int and report_step <= 0:
        report_step = num_spectra // 10
        # If a negative integer or zero is given for report_step, set it to be 1/10 of the total number of spectrum to
        # be generated (i.e. get a printout for every 10% progress).

    dataset.num_spectra = num_spectra
    dataset.num_points = num_points
    dataset.seed = seed
    # Store values for number of spectra, number of datapoints per spectrum and the generation seed for this dataset.

    dataset.cs_norm = nnD.gen_cs(
        num_spectra,
        seed,
        num_peaks,
        max_num_peaks,
        peak_marg,
        w_sep,
        min_allowed_int,
        min_slope,
        cutoff_higher,
    )
    # Generate and store the array of (normalised) parameters for this dataset.

    dataset.cs = nnD.rescale_cs(dataset.cs_norm)
    # Rescale the parameters to be physically realistic and store.

    dataset.inputs, dataset.e_minmax = nnD.resize_spectra(
        nnD.build_spectra(dataset.cs, report_step, plot=True),
        num_points,
        spline_order,
        minmax,
    )
    # Render and rescale the spectra, storing the intensity scales as well as the minimum and maximum binding energies.
    # Intensity scales are called 'inputs' because they will be the inputs to the neural networks.

    dataset.peak_count_OH = get_categorical_count(dataset.cs[0], max_num_peaks)
    # Convert the number of peaks for each spectrum into 'one-hot' vectors and store (see 'get_categorical_count' for an
    # explanation).

    if pos_oh:
        dataset.peak_pos_OH_concat = get_categorical_pos(
            dataset, num_bins=200, max_num_peaks=max_num_peaks
        )
    else:
        dataset.peak_pos_OH_concat = None
    if pos_mh:
        dataset.peak_pos_MH = get_categorical_pos(dataset, num_bins=200)
    else:
        dataset.peak_pos_MH = None
    # Generate and store 'one-hot' and/or 'multi-hot' vector labels for peak positions if pos_oh and/or pos_mh are True
    # (see 'get_binned_peak_pos' for an explanation). Otherwise, set to placeholder NoneTypes.


# get_real_spectra
#
# Imports and resizes/rescales real data from nexus files and appends to an empty SpectrumDataset container object.
#
# Arguments:
#
#   dataset:
#       The SpectrumDataset container object into which the imported data will be placed.
#
#   file_dir:
#       Directory containing the nexus files to be imported.
#
#   file_numbers: (= None)
#       A list of filenumbers (or a single integer filenumber) for the files you want to load. If NoneType, a window is
#       opened for you to select files by hand (default behaviour).
#
#   num_points (= 200):
#       The number of datapoints to interpolate the spectra to. (Dictated by the input size of the networks).
#
#   spline_order: (= 1)
#       The order of the polynomials used in interpolating the spectra - 1 = linear, 3 = cubic etc.
#
#   minmax: (= True)
#       Whether or not to rescale the intensities to the interval (0, 1) (default behaviour is to rescale).


def get_real_spectra(
    dataset, file_dir="", file_numbers=None, num_points=200, spline_order=1, minmax=True
):
    dataset.num_points = num_points
    # Store value for number of datapoints per spectrum.

    dataset.seed = None
    dataset.cs_norm = None
    dataset.cs = None
    dataset.peak_count_OH = None
    dataset.peak_pos_OH_concat = None
    dataset.peak_pos_MH = None
    # Store placeholder NoneTypes for all atttributes which only pertain to generated (labelled) data.

    spectra = []
    # Initialise the list which will hold the retrieved raw spectra.

    nexus_data = get_nexus_data(file_dir, file_numbers)
    # Load in the raw data from the nexus files (this is a dictionary of dictionaries containing all the directly
    # relevant spectral data from the files, see 'get_nexus_data' for details of how it's structured).

    for file_number in nexus_data:
        # Iterate over all the imported nexus files.

        if any(nexus_data[file_number]):
            # Check that there is actually data associated with this filenumber (some nexus files may have contained
            # nothing).

            for name in nexus_data[file_number]:
                # Iterate over the spectra within the file (there can be more than one - 'name' is the user-defined
                # heading created when the spectrum was captured on the beamline).

                spectrum = nexus_data[file_number][name]
                # Get the spectrum.

                th_sum_image = spectrum["th_sum_image_data"]
                # Get the angle-summed intensity values array for the spectrum.

                for i in range(len(th_sum_image)):
                    # Iterate over the length of the angle-summed intensity data array - 0th axis, corresponding to
                    # excitation energy. We're considering each excitation energy to be a different spectrum if there's
                    # more than one (e.g. in a standing waves file).

                    spectra.append(
                        [np.flip(spectrum["energies"][0]), np.flip(th_sum_image[i])]
                    )
                    # Get the binding energy and intensity scales for the spectrum in the same format as is produced by
                    # 'build_spectra' when we generate data - the arrays are flipped so that the data has binding energy
                    # increasing left to right (***i.e. it is assumed to be decreasing in the nexus files***).

    dataset.inputs, dataset.e_minmax = nnD.resize_spectra(
        spectra, num_points, spline_order, minmax
    )
    # Rescale the spectra and store the intensity scales as well as the minimum and maximum binding energies. This is
    # the same function used when resizing and rescaling generated data (in 'NNDataBuilderPeter'). Intensity scales are
    # called 'inputs' because they will be the inputs to neural networks.

    dataset.num_spectra = len(dataset.inputs)
    # Store the value for the number of spectra in the dataset.


# get_nexus_data
#
# Imports data from one or more nexus files and organises it into separate spectra (in dictionaries). If standing wave
# files are imported, each excitation energy is considered to be a separate spectrum - i.e. they're not automatically
# summed. This function is called inside 'get_real_spectra'.
#
# Arguments:
#
#   file_dir: (= '')
#       Directory containing the nexus files to be imported.
#
#   file_numbers: (= None)
#       A list of filenumbers (or a single filenumber) for the files you want to load. If NoneType, a window is opened
#       for you to select files by hand (default behaviour).
#
# Outputs:
#
#   files_data:
#       A nested dictionary containing the spectral data from the nexus files.
#
#       'files_data'
#          |
#       [file number(s)...]
#          |
#       [spectra name(s)...]
#          |
#       |'angles'            - array containing the the discrete angles of each of the detector elements.
#       |'energies'          - array containing the energy scale of the spectrum (I assume binding, sometimes untrue).
#       |'excitation_energy' - array containing the photon energies
#       |'image_data'        - 3D array containing the detector counts (intensities) over the above 3 axes.
#       |'th_sum_image_data' - 2D array containing the detector counts (intensities) summed along the angles (0th) axis.


def get_nexus_data(file_dir="", file_numbers=None):
    nexus_files = {}
    files_data = {}
    # Initialise the dictionaries that will hold the nexus files and the extracted data.

    exclude = (
        "end_time",
        "entry_identifier",
        "experiment_identifier",
        "hm3amp20",
        "instrument",
        "program_name",
        "scan_command",
        "scan_dimensions",
        "scan_identifier",
        "smpmamp39",
        "start_time",
        "user01",
    )
    # List of groups inside the nexus files that we are not interested in.

    tag = "lens_mode"
    # This is an attribute which can always be found in groups that actually contain spectral data. The group names
    # themselves are user-defined and so don't follow a pattern that can be used to determine what's in them.

    if isinstance(file_numbers, int):
        nexus_files[file_numbers] = h5py.File(
            "{}\\i09-{}.nxs".format(file_dir, file_numbers), "r"
        )
        # If a single file number was given, load in that nexus file using the h5py module's 'File' function and append
        # to the 'nexus_files' dictionary with the file number as key.

    elif isinstance(file_numbers, list) or isinstance(file_numbers, tuple):
        for file_number in file_numbers:
            nexus_files[file_number] = h5py.File(
                "{}\\i09-{}.nxs".format(file_dir, file_number), "r"
            )
            # If a list of file numbers was given, load in each nexus file using the h5py module's 'File' function and
            # append to the 'nexus_files' dictionary with the filenumbers as keys.

    else:
        root = tk.Tk()
        root.withdraw()
        file_paths = aof(
            initialdir=file_dir,
            title="Select Files:",
            filetypes=(("nexus files", "*.nxs"), ("all files", "*.*")),
        )
        root.destroy()
        # If no file numbers were given, open a window (tkinter) which allows you to select nexus files manually and
        # return the file paths automatically.

        for file_path in file_paths:
            nexus_files[int(file_path[-10:-4])] = h5py.File(file_path, "r")
            # Load in each nexus file using the h5py module's 'File' function and append to the 'nexus_files' dictionary
            # with the file numbers as keys (file numbers are extracted from the file paths assuming 6 digits and a
            # '.nxs' file extension).

    for file_number in nexus_files:
        # Iterate over all the imported files.

        spectra_dict = {}
        # Initialise the dictionary that will contain the spectral data.

        entry1 = nexus_files[file_number]["entry1"]
        # Get the top (entry) layer of the file.

        for name in entry1:
            # Iterate over the groups contained in the entry layer.

            if name not in exclude:
                group = entry1[name]
                # If the group name is not blacklisted, retrieve it.

                if tag in group:
                    # If the tag can be found inside the group, then this group does indeed contain the spectral
                    # information we're looking for.

                    spectra_dict[name] = {
                        "angles": np.array(group["angles"]),
                        "energies": np.array(group["energies"]),
                        "excitation_energy": np.array(group["excitation_energy"]),
                        "image_data": np.array(group["image_data"]),
                    }
                    # Get the spectral data and append to 'spectra_dict'.

                    spectra_dict[name]["th_sum_image_data"] = np.sum(
                        spectra_dict[name]["image_data"], axis=1
                    )
                    # Take the sum of the intensity data ('image_data') over the angles axis (0th axis) and append to
                    # 'spectra_dict'. 'image_data' is a 3D tensor with axes for angles, excitation energy and binding
                    # energy. Summing over the angles reduces it to a 2D array.

        files_data[file_number] = spectra_dict
        # Append the 'spectra_dict' to the 'files_data' dictionary. This may now be passed to 'get_real_spectra' for
        # formatting and rescaling.

    return files_data


# ======================================================================================================================

# get_categorical_count
#
# Generates one-hot vector labels for number of peaks.
#
# A one-hot vector is a vector whose entries are all 0s except for a single 1. They are a natural way to present
# categorical data to a computer - the position of the 1 in the vector corresponds to the class to which the associated
# datapoint belongs. In our case, the (mutually exclusive) classes are 1, 2, 3, 4, 5 and 6 peaks. '1 peak' is converted
# to (1, 0, 0, 0, 0, 0), while '4 peaks' is converted to (0, 0, 0, 1, 0 ,0) etc.
#
# Arguments:
#
#   nums_peaks:
#       A list or array of peak numbers, to be converted to a list of one-hot vectors.
#
#   max_num_peaks: (= None)
#       The maximum number of peaks expected - this is needed to decide the length of the vectors (number of classes).
#       If set to a non-integer value, or a value less than the maximum in nums_peaks, then the latter will be used
#       (default behaviour).
#
# Outputs:
#
#   peak_count_oh:
#       A numpy array of all the one-hot vectors specifying number of peaks.


def get_categorical_count(nums_peaks, max_num_peaks=None):
    if (not isinstance(max_num_peaks, int)) or (max_num_peaks < max(nums_peaks)):
        max_num_peaks = int(max(nums_peaks))
        # If 'max_num_peaks' is not an integer, or has a value less than the maximum value in 'nums_peaks', it is
        # reassigned to the latter.

    peak_count_oh = []
    # Initialise the list that will contain the one-hot vectors.

    for i in nums_peaks:
        # Iterate over all numbers of peaks (i.e. all spectra).

        count = np.zeros(max_num_peaks)
        # Initialise the one-hot vector as an array of all 0s.

        count[int(i - 1)] = 1
        # Change the value in the position corresponding to the number of peaks to a 1 (i - 1 because indexing from 0).

        peak_count_oh.append(count)
        # Append the one-hot vector to the list.

    return np.array(peak_count_oh, dtype="int")


# get_binned_peak_pos
#
# Generates one-hot or mutli-hot vector labels for the peak positions in a dataset of pre-generated spectra and appends
# them to the dataset object. This function is optionally called inside 'gen_dataset'.
#
# When it comes to the neural network training process, this is recasting the task of finding peak positions as
# continuous variables - a regression problem - to a classification task. The binding energy scale is divided into a
# number of bins (default 200, to match interpolated spectra default size) and each bin is labelled as being the closest
# one to a peak centre or not. We can generate one-hot vectors for each peak and concatenate them, or generate a single
# 'multi-hot' vector for all the peaks in the spectra. Consider a spectrum containing 3 peaks at (fractional) positions
# 0.27, 0.45 and 0.89, and where we divide the energy scale into 10 bins:
#
# One-hot vectors:
#
#    Peak 1              Peak 2              Peak 3              Peak 4   Peak 5   Peak 6
#   [0 1 0 0 0 0 0 0 0 0|0 0 0 0 1 0 0 0 0 0|0 0 0 0 0 0 0 0 1 0| all 0s | all 0s | all 0s ]
#   Length = /max_num_peaks/*/num_bins/ (e.g. 1200 for max 6 peaks and 200 bins)
#
# Multi-hot vector:
#
#   [0 1 0 0 1 0 0 0 1 0]
#
# There are advantages and disadvantages to both alternatives (discussed elsewhere), but the essential point is that
# networks trained using these 'discrete' position labels can output probability distributions for the peak positions
# rather than just point estimates. Theoretically this allows one to derive confidence intervals which could inform the
# bounding values fed to a fitting algorithm.
#
# Arguments:
#
#   dataset:
#       The SpectrumDataset container object from which the peak position information is retrieved.
#
#   num_bins: (= 200)
#       The number of bins to divide the spectrum into when creating the one-hot/multi-hot vector labels.
#
#   max_num_peaks: (= None)
#       The maximum number of peaks expected - this is needed to decide how many vectors to make and concatenate, in the
#       case of the one-hot vectors. This argument also detemines whether one-hot or multi-hot vectors are generated.
#       If a positive integer is provided, one-hot vectors are generated. If any other value or data type is provided,
#       multi-hot vectors are generated (default behaviour).
#
# Outputs:
#
#   peak_pos_oh (optional):
#       A vector comprised of /max_num_peaks/ concatenated one-hot vectors, each of which is /num_bins/ long, specifying
#       all the peak positions in the spectra.
#
#   peak_pos_mh (optional):
#       A multi-hot vector of length /num_bins/, specifying all the peak positions in the spectra.


def get_categorical_pos(
    dataset, num_bins=200, max_num_peaks=None, negative_labels=False
):
    e_ind = np.array(range(10, len(dataset.cs_norm), 6))
    # Assign peak energy (position) indices within the parameter arrays.

    if isinstance(max_num_peaks, int) and (max_num_peaks > 0):
        # If you provided a positive integer for 'max_num_peaks', create and append concatenated one-hot vectors.

        peak_pos_oh = []
        # Initialise the list that will contain the concatenated one-hot vectors.

        for i in range(dataset.num_spectra):
            # Iterate over all the spectra in the dataset.

            binned_pos = np.zeros(num_bins * max_num_peaks)
            # Initialise the concatenated one-hot vector as an array of 0s. Its length is /max_num_peaks/ times the
            # length of a single one-hot vector (so usually 1200).

            raw_pos = dataset.cs_norm[e_ind, i]
            # Get a list of the fractional positions of all the peaks in this spectrum from the normalised parameter
            # array.

            for j in range(int(dataset.cs[0, i])):
                # Iterate over the number of peaks in this spectrum.

                for k in range(len(raw_pos)):
                    # Iterate over the fractional positions in 'raw_pos'.

                    if k == j:
                        binned_pos[int(np.floor((raw_pos[k] + j) * num_bins))] = 1
                    elif negative_labels:
                        binned_pos[int(np.floor((raw_pos[k] + j) * num_bins))] = -1
                # Determine which bin is closest to the kth peak position and set it to 1. Notice that for each
                # successive peak (indexed by j = 0, 1, 2, ...), j is being added to the fractional peak position. This
                # ensures that each peak's one-hot vector is correctly singly populated with a 1, within the overall
                # concatenated vector. [ADDED NEGATIVE LABELS OPTION AS AN EXPERIMENT]

            peak_pos_oh.append(binned_pos)
            # Append the concatenated one-hot vector to the list.

        return np.array(peak_pos_oh, dtype="int")

    else:
        # If max_num_peaks is not a positive integer, create and append multi-hot vectors.

        peak_pos_mh = []
        # Initialise the list that will contain the mutli-hot vectors.

        for i in range(dataset.num_spectra):
            # Iterate over all the spectra in the dataset.

            binned_pos = np.zeros(num_bins)
            # Initialise the multi-hot vector as an array of 0s.

            for j in range(int(dataset.cs[0, i])):
                # iterate over the peaks in this spectrum.

                raw_pos = dataset.cs_norm[e_ind[j], i]
                # Get the fractional energy (position) of this peak from the dataset's (normalised) parameter array.

                binned_pos[int(np.floor(raw_pos * num_bins))] = 1
                # Determine which bin is closest to the peak position and set it to 1. Notice that j is not added to the
                # fractional position this time as we are multipully populating a single multi-hot vector.

            peak_pos_mh.append(binned_pos)
            # Append the multi-hot vector to the list.

        return np.array(peak_pos_mh, dtype="int")


# ======================================================================================================================

# count_peaks
#
# Takes in a SpectrumDataset object containing generated data and tallies up the number of spectra containing each
# number of peaks, returning the results in an array.
#
# Arguments:
#
#   dataset:
#       The SpectrumDataset container object from which the number of peaks information is retrieved. It should
#       contain generated data.
#
# Outputs:
#
#   counts:
#       An array containing the total number of spectra having each number of peaks.


def count_peaks(dataset):
    if not dataset.cs:
        print("This dataset does not have the necessary labels.")
        return
    # If the dataset parameter arrays are NoneType this means the data wasn't generated and hence doesn't have peak
    # count labels. Cancel and return nothing.

    counts = np.zeros((dataset.cs.shape[0] - 10) // 6, dtype=int)
    # Initialise the array that will contain the number of peaks tallies with all 0s.

    for i in range(len(dataset.inputs)):
        # Iterate over all the spectra in the dataset.

        counts[int(dataset.cs[0, i]) - 1] += 1
        # Increment the value of the entry corresponding to the number of peaks in this spectrum (0th entry = 1 peak,
        # 1st entry = 2 peaks etc.).

    return counts


# extract_from_dataset
#
# Takes in a SpectrumDataset and a list/array of indices and extracts all the spectra corresponding to these given
# indices, depositing them into a new SpectrumDataset object. Each spectrum, i.e. datapoint in the dataset, is composed
# of several separate features stored in different attributes of the dataset (e.g. slices from the parameter arrays and
# 'inputs' attribute etc.). When we go through extracting spectra, we need to make sure we retrieve all its different
# features which are stored in parallel.
#
# Arguments:
#
#   dataset:
#       The 'parent' SpectrumDataset object from which spectra are to be extracted.
#
#   to_extract:
#       A list/array of the indices corresponding to the spectra to be extracted.
#
# Outputs:
#
#   dataset_out:
#       The 'child' SpectrumDataset object containing the extracted spectra.


def extract_from_dataset(dataset, to_extract):
    dataset_out = SpectrumDataset("None")
    # Initialise the empty dataset object that will contain the extracted spectra.

    dataset_out.seed = dataset.seed
    # Store the seed value from the parent dataset in the output dataset.

    for name in dataset.__dir__():
        attr = dataset.__getattribute__(name)
        # Iterate over the and get each of the attributes of the parent dataset object.

        if isinstance(attr, np.ndarray):
            # Check if the attribute is a numpy array - if it is, it contains part of the spectral information we want
            # to extract.

            if name not in ("cs", "cs_norm"):
                # Check that the attribute isn't either of the parameter arrays, as their 1st rather than 0th axis span
                # the spectra. Every other array containing parts of the datapoint (e.g. the 'inputs' (intensity scale),
                # the arrays of one-hot vectors for peak counts and peak positions etc.) have the 0th axis spanning the
                # spectra.

                dataset_out.__setattr__(name, attr[to_extract])
                # Extract array slices from the attribute corresponding to the indices provided.

            else:
                # For the parameter arrays (normalised and rescaled), the 1st rather than 0th axis spans the spectra.
                # Need use [:, to_extract] instead of [to_extract].

                dataset_out.__setattr__(name, attr[:, to_extract])
                # Extract array slices from the attribute corresponding to the indices provided.

        elif isinstance(attr, list):
            # While none of the functions I've written should generate a SpectrumDataset containing list attributes,
            # occasionally there have been cases where I have manually stitched together a dataset to include the raw
            # spectra (i.e. unrescaled binding energy and intensity scales) for one reason or another. Since these raw
            # spectra all have different lengths, they can't be stored in a numpy array.

            attr_out = []
            # Initiate the list that will contain the extracted items from this list attribute.

            for i in to_extract:
                # Lists don't have the fancy functionality of numpy arrays - we can't grab multiple items at once,
                # instead we must iterate through each index in the 'to_extract' list of indices.

                attr_out.append(attr[i])
                # Get each desired item and append it to the list.

            dataset_out.__setattr__(name, attr_out)
            # Append the new extracted attribute to the child dataset.

    if isinstance(dataset_out.inputs, np.ndarray):
        # Check that the child dataset does actually have 'inputs' data in a numpy array (rescaled spectra intensity
        # scales). In general it of course should, but this provision allows you to use this function on datasets that
        # you've messed around with in some way, without it crashing due to the following lines.

        dataset_out.num_spectra = len(dataset_out.inputs)
        # Get the new number of (extracted) spectra and store it in the new dataset.

    return dataset_out


# delete_from_dataset
#
# Takes in a SpectrumDataset and a list/array of indices and deletes all the spectra corresponding to these given
# indices. The dataset can either be modified in place, or a new dataset containing the surviving spectra can be
# outputted (default behaviour). Each spectrum, i.e. datapoint in the dataset, is composed of several separate features
# stored in different attributes of the dataset (e.g. slices from the parameter arrays and inputs attribute etc.). When
# we go through deleting spectra, we need to make sure we delete all its different features which are stored in
# parallel.
#
# Arguments:
#
#   dataset:
#       The 'parent' SpectrumDataset object from which spectra are to be deleted.
#
#   to_delete:
#       A list/array of the indices corresponding to the spectra to be deleted.
#
#   returner: (= True)
#       Controls whether spectra are deleted from the input dataset in place, or a new dataset containing the surviving
#       spectra is returned (default behaviour).
#
# Outputs:
#
#   dataset_out (optional):
#       The 'child' SpectrumDataset object containing the surviving spectra.


def delete_from_dataset(dataset, to_delete, returner=True):
    if returner:
        dataset_out = copy.deepcopy(dataset)
        # If you want a child dataset to be returned, initialise it as a copy of the parent dataset so that spectra
        # don't get deleted from the latter.

    else:
        dataset_out = dataset
        # If you want to modify the dataset in place, set the child to be the parent.

    for name in dataset_out.__dir__():
        attr = dataset_out.__getattribute__(name)
        # Iterate over the and get each of the attributes of the parent dataset object.

        if isinstance(attr, np.ndarray):
            # Check if the attribute is a numpy array - if it is, it contains part of the spectral information we want
            # to delete.

            axis = 0
            # Most of the attributes have their 0th axis spanning the spectra.

            if name in ("cs", "cs_norm"):
                axis = 1
                # For the parameter arrays (normalised and rescaled), the 1st rather than 0th axis spans the spectra.

            dataset_out.__setattr__(name, np.delete(attr, to_delete, axis=axis))
            # Get the array resulting from the deletion of the relevant slices from the current attribute according to
            # the indices in to_delete (along the axis which spans the spectra). Set this array as the replacement for
            # the attribute in the child dataset (or overwrite the attribute in the parent - depends which you selected
            # above).

        elif isinstance(attr, list) and (name != "__slotnames__"):
            # While none of the functions I've written should generate a SpectrumDataset containing list attributes,
            # occasionally there have been cases where I have manually stitched together a dataset to include the raw
            # spectra (i.e. unrescaled binding energy and intensity scales) for one reason or another. Since these raw
            # spectra all have different lengths, they can't be stored in a numpy array.

            attr_out = copy.copy(attr)
            # Initialise the output (child) attribute as a copy of the input attribute.

            for i in sorted(to_delete, reverse=True):
                # Iterate over the indices in 'to_delete' in decreasing order - this way we delete array entries from
                # highest index to lowest, so that the indices of subsequent entries don't change before we've deleted
                # them.

                del attr_out[i]
                # Delete each relevant entry in the list according to the indices in to_delete.

            dataset_out.__setattr__(name, attr_out)
            # Set the resulting list as the replacement for the attribute in the child dataset (or overwrite the
            # attribute in the parent - depends which you selected above.

    dataset_out.num_spectra = len(dataset_out.inputs)
    # Get the new number of (surviving) spectra and store it in the new dataset.

    if returner:
        return dataset_out
    # Return the child dataset containing the surviving spectra if you chose to do so.


# get_real_n_peaks_indices
#
# Searches through a SpectrumDataset object and returns a list of the indices of the spectra containing a certain number
# (or numbers) of peaks that you choose.
#
# Arguments:
#
#   dataset:
#       The SpectrumDataset to be searched for a given number of peaks. Should contain generated data.
#
#   nums_peaks:
#       An integer or list/tuple/array of integers whose value(s) are the number(s) of peaks to be searched for.
#
# Outputs:
#
#   indices:
#       A list containing the indices of the spectra that contain the desired number(s) of peaks.


def get_real_n_peaks_indices(dataset, nums_peaks):
    indices = []
    # Initialise the list to which the indices will be appended.

    if type(nums_peaks) == int:
        nums_peaks = (nums_peaks,)
        # If nums_peaks is an integer, convert it into an iterable format (a tuple).

    for i in range(len(dataset.inputs)):
        # iterate over all the spectra in the dataset.

        if dataset.cs_norm[0, i] in nums_peaks:
            indices.append(i)
            # Check the number of peaks in each spectrum (in the normalised parameter array) and if it matches one of
            # the values in nums_peaks, append the spectrum's index to the list.

    return indices


# get_real_n_peaks
#
# Extracts all the spectra containing a given number or numbers of peaks from an input dataset and places them into a
# new dataset object.
#
# Arguments:
#
#   dataset:
#       The 'parent' dataset object from which the spectra are to be extracted. Should contain generated data.
#
#   nums_peaks:
#       An integer or list/tuple/array of integers whose value(s) are the number(s) of peaks to be retrieved. If not in
#       the integer range of possible values in the dataset, the function instead retrieves each single number of peaks
#       individually and outputs a separate dataset containing each, in a list.
#
# Outputs:
#
#   dataset_out (optional):
#       The 'child' dataset object into which the extracted spectra are deposited.
#
#   datasets_out (optional):
#       A list of 'child' dataset objects, one for each number of peaks in the 'parent' dataset.


def get_real_n_peaks(dataset, nums_peaks=None):
    all_nums_peaks = range(1, (dataset.cs_norm.shape[0] - 10) // 6 + 1)
    # Get an integer range of all the possible numbers of peaks in the parent dataset, based on the length of the its
    # normalised parameter array. (6 parameters per peak, plus an additional 10 non-peak parameters).

    if nums_peaks not in all_nums_peaks:
        datasets_out = []
        # If the given number(s) of peaks to retrieve is not inside the possible range (or is not an integer
        # altogether), initialise the list that will contain the separate child datasets for each number of peaks.

        for i in all_nums_peaks:
            # Iterate over all the possible numbers of peaks in the parent dataset.

            to_extract = get_real_n_peaks_indices(dataset, (i,))
            # Get the indices of all the spectra with /i/ peaks.

            dataset_out = extract_from_dataset(dataset, to_extract)
            # Create a child dataset containing the spectra with /i/ peaks.

            datasets_out.append(dataset_out)
            # Append this dataset to the list of child datasets.

        return datasets_out

    else:
        # If the requested number(s) of peaks are possible within

        to_extract = get_real_n_peaks_indices(dataset, nums_peaks)
        # Get the indices of all the spectra with the requested number(s) of peaks.

        dataset_out = extract_from_dataset(dataset, to_extract)
        # Create a child dataset containing the spectra with the requested number(s) of peaks.

        return dataset_out


# ======================================================================================================================

# plot_spectra
#
# Plots the spectra in a SpectrumDataset object for quick inspection.
#
# Arguments:
#
#   dataset:
#       The SpectrumDataset object containing the spectra to be plotted. The spectra can be generated or real data from
#       nexus files.
#
#   num_to_plot: (= 10)
#       The number of spectrum to plot (creates /num_to_plot/ separate windows). The plotted spectra will be adjacent to
#       each-other in the dataset (e.g. plot spectum 0 through spectrum to 9 or spectrum 14 through to spectrum 18).
#
#   start: (= 0)
#       The spectrum to begin plotting from.
#
#   real_len: (= True)
#       Whether to plot the spectra with their original, unrescaled number of datapoints. By default, unrescaled copies
#       of the spectra are not stored in the datasets (to save space) meaning that these aren't available to plot.
#       However, in the case of generated spectra, we have the original parameter arrays used to render them and so this
#       can be done again in-situ. Datasets containing real spectra therefore can't be plotted with their original
#       number of datapoints.


def plot_spectra(dataset, num_to_plot=10, start=0, real_len=True, figure_ind=None):
    num_to_plot = min(len(dataset.inputs[start:]), num_to_plot)
    # If num_to_plot is greater than the number of spectra in the dataset, reduce it to this value.

    for i in range(start, start + num_to_plot):
        # Iterate over the indicies corresponding the the spectra to plot.

        if isinstance(figure_ind, int):
            plt.figure(figure_ind)
        else:
            plt.figure(i)
        # Initialise a plot window with the spectrum's index as figure label.

        if real_len and isinstance(dataset.cs, np.ndarray):
            # If plotting with real lengths, check that the dataset has a parameter array (i.e. has generated data)
            # otherwise this won't be possible.

            lineshape = nnD.build_spectra(np.expand_dims(dataset.cs[:, i], axis=1))[0][
                1
            ]
            # Re-render the original generated spectrum from the parameter array to get the original number of
            # datapoints, but retain only the intensity scale.

            x = np.linspace(0, 1, len(lineshape))
            # Create the x-axis (spanning from 0 to 1) with the same number of datapoints as the intensity scale
            # ('lineshape').

        else:
            lineshape = dataset.inputs[i]
            # Simply get the rescaled intensity scales ('inputs' attribute) from the dataset.

            x = np.linspace(0, 1, dataset.num_points)
            # Create the x-axis (spanning from 0 to 1) with the same number of datapoints as the intensity scale
            # ('lineshape').

        plt.plot(x, lineshape, color="k", linewidth=1)
        # Plot the lineshape.

        if isinstance(dataset.cs_norm, np.ndarray):
            # If the dataset has a parameter array (i.e. has generated data), we can also overlay indicators for the
            # peak positons.

            nums_peaks = np.array((dataset.cs_norm[0]), dtype="int")
            # Get the slice from the normalised parameter array that contains all the numbers of peaks for each
            # spectrum.

            e_ind = np.array(range(10, len(dataset.cs_norm), 6))
            # Assign peak energy (position) indices.

            for j in range(nums_peaks[i]):
                # Iterate over the spectra.

                frac_pos = dataset.cs_norm[e_ind[j], i]
                # Get the fractional peak energies (positions) from the normalised parameter array.

                plt.axvline(frac_pos, color="k", linestyle=":", linewidth=2, zorder=0)
                # Plot dotted vertical lines at each peak position.

            plt.title(
                "Real: {}".format(
                    np.round(dataset.cs_norm[e_ind[: nums_peaks[i]], i], 3)
                )
            )
            # Add a title listing the fractional peak positions (under 'Real').

        plt.yticks([])
        # Remove ticks from the y-axis (intensity is arbitrary).

        plt.tight_layout()
        # Use tight layout - looks neater.
