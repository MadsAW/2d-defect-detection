from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from skimage.draw import disk
import numpy as np
import torch

def add_background_class(label):
    """
    Add a background class to a label tensor.

    This function ensures that all values in the label tensor are between 0 and 1,
    and then adds a background class. The background class is calculated as the
    complement of the maximum value across all other classes.

    Args:
        label (numpy.ndarray): The input label tensor of shape (C, H, W),
                               where C is the number of classes, and H, W are
                               the height and width of the label image.

    Returns:
        numpy.ndarray: A new label tensor of shape (C+1, H, W), where the last
                       channel represents the background class.

    Note:
        - If any value in the input label is greater than 1, it will be set to 1.
        - The background class is computed as 1 minus the maximum value across
          all other classes, ensuring that the sum of all classes (including
          background) is always 1.
        - This function is particularly useful for preparing labels for
          multi-class segmentation tasks where a background class is needed.
    """
    # make sure no value is larger than 1 if so set it to 1
    label = np.min([label, np.ones(label.shape)], axis=0)
    # add background class
    shape = [1, *label.shape[1:]]
    # Note there is an iffy use of np.max in regards to softmax (but this is to ensure that 1>=label>=0)
    # this is only for defects when doing multiclass labeling at once!
    label = np.concatenate(
        [label, (np.ones(shape) - np.max(label, axis=0).reshape(shape))], axis=0
    )
    return label


## Label functionals
def gaussian_label(positions, grid, sigma, amplitude=1):
    """
    Generate Gaussian labels for given positions on a 2D grid.

    Args:
        positions (numpy.ndarray): Array of shape (n, 2) containing n 2D positions.
        grid (numpy.ndarray): Array of shape (m, 2) representing the 2D grid points.
        sigma (float): Standard deviation of the Gaussian distribution.
        amplitude (float, optional): Amplitude of the Gaussian peaks. Defaults to 1.

    Returns:
        numpy.ndarray: Array of shape (n, m) containing Gaussian labels for each position.
                       Can be reshaped to match the original grid shape if needed.

    Note:
        The Gaussian label is calculated as: amplitude * exp(-distance^2 / (2 * sigma^2))
        where distance is the Euclidean distance between each position and grid point.
    """
    inner = cdist(positions, grid.T) ** 2 / (2 * sigma**2)
    return amplitude * np.exp(-inner)


def circ_label(positions, grid, radius):
    """
    Returns circular label given position in 2d coordinates.

    Args:
        positions (numpy.ndarray): Array of shape (n, 2) containing n 2D positions.
        grid (numpy.ndarray): Array of shape (m, 2) representing the 2D grid points.
        radius (float): Radius of the circular label.

    Returns:
        numpy.ndarray: Boolean array of shape (n, m) where True values indicate
                       points within the specified radius of each position.

    Note:
        The output can be reshaped to match the original grid shape if needed.
    """
    return cdist(positions, grid.T) < radius


## Label generators


def position_label(
    atoms,
    parameters,
    sigma=0.5,
    cluster_distance=0.5,
    label_function="gauss",
    tol=1e-3,
    background=True,
    species=None,
):
    """
    Generate position labels for atoms in a 2D grid.

    This function creates labels for atom positions, optionally clustering nearby atoms
    and applying either Gaussian or circular labels.

    Args:
        atoms (ase.Atoms): Atomic structure containing positions.
        parameters (dict): Dictionary containing 'scanning_sampling' and 'size'.
        sigma (float, optional): Width of Gaussian or radius of circular label. Defaults to 0.5.
        cluster_distance (float, optional): Distance for clustering nearby atoms. Defaults to 0.5.
        label_function (str, optional): Type of label function, either 'gauss' or 'circ'. Defaults to 'gauss'.
        tol (float, optional): Tolerance for Gaussian label cutoff. Defaults to 1e-3.
        background (bool, optional): Whether to add a background class. Defaults to True.
        species (str, optional): Chemical symbol to filter atoms. Defaults to None (all atoms).

    Returns:
        tuple: (labels, label_names)
            - labels (numpy.ndarray): 3D array of shape (channels, height, width) containing labels.
            - label_names (numpy.ndarray): 1D array of label names corresponding to channels.

    Raises:
        ValueError: If an unknown label function is specified.

    Note:
        The function projects 3D atomic positions onto a 2D grid and applies the specified
        labeling method. It can handle both Gaussian and circular labels, and optionally
        clusters nearby atoms.
    """
    ## Returns labels, label_names
    # Project atoms
    pos = atoms.get_positions()[:, :2]
    if species:
        pos = pos[np.array(atoms.get_chemical_symbols()) == species, :]
    sampling = parameters["scanning_sampling"]
    lengths = parameters["size"]
    p, q = [int(lengths[0] / sampling), int(lengths[1] / sampling)]
    yV, xV = np.mgrid[0:p, 0:q] * sampling

    c = fcluster(linkage(pos), t=cluster_distance, criterion="distance")
    # positions altering
    # Note seems to fuck about when zone tilt is too large (projected positions become too close)
    clusters = np.array([np.mean(pos[c == i + 1], axis=0) for i in range(np.max(c))])
    grid = np.array([xV, yV])

    label = np.zeros([p, q], dtype=np.float64)
    if label_function == "circ":
        mask_size = sigma / sampling
    elif label_function == "gauss":
        mask_size = np.ceil(np.sqrt(-2 * np.log(tol)) * sigma / sampling)
    else:
        raise ValueError(
            f"Unknown label function: {label_function} \nUse 'circ' or 'gauss'"
        )
    for cluster_center in clusters:
        # the coordinates must have x and y switched to match the current transposed version of the label
        cluster_center = cluster_center[::-1]
        rr, cc = disk(cluster_center / sampling, mask_size, shape=[p, q])

        if label_function == "circ":
            label[rr, cc] += 1
        elif label_function == "gauss":
            inner = cdist([cluster_center[::-1]], grid[:, rr, cc].T) ** 2 / (
                2 * sigma**2
            )
            amplitude = 1
            label[rr, cc] += amplitude * np.exp(-inner).T.squeeze()
    # add channel information
    label = np.expand_dims(label, 0)
    if background:
        label = add_background_class(label)
    label_names = np.concatenate([["position"], ["background"] * background])
    return np.transpose(label, axes=(0, 2, 1)), label_names

def defect_column_label(
    defects,
    parameters,
    sigma=0.5,
    cluster_distance=1,
    label_function=gaussian_label,
    background=True,
):
    """
    Create labels for defect columns in a STEM image.

    This function generates labels for defect columns in a Scanning Transmission Electron Microscopy (STEM) image
    based on the provided defect positions and parameters.

    Args:
        defects (dict): A dictionary where keys are defect types and values are lists of defect positions.
        parameters (dict): A dictionary containing various parameters for label generation, including:
            - 'scanning_sampling': The sampling rate of the scan in Angstroms per pixel.
            - 'size': The dimensions of the image in Angstroms.
        sigma (float, optional): The standard deviation of the Gaussian distribution used for labeling. Defaults to 0.5.
        cluster_distance (float, optional): The distance threshold for clustering defects. Defaults to 1.
        label_function (callable, optional): The function used to generate labels. Defaults to gaussian_label.
        background (bool, optional): Whether to add a background class to the labels. Defaults to True.

    Returns:
        tuple: A tuple containing two elements:
            1. numpy.ndarray: An array of labels with shape (num_channels, height, width).
            2. list: A list of label names corresponding to each channel in the label array.

    Note:
        The function sorts defect types alphabetically to ensure consistent ordering across different runs.
        The label array is transposed to match the expected format (channels, height, width).
    """
    # Sigma is in [Å]
    # Create a channel for each of the different defects
    ## Important sort the list to ensure all data has same sorting!
    defect_types = sorted(defects.keys(), key=lambda x: x.lower())
    # Generate grid
    sampling = parameters["scanning_sampling"]
    lengths = parameters["size"]
    p, q = [int(lengths[0] / sampling), int(lengths[1] / sampling)]
    yV, xV = np.mgrid[0:p, 0:q] * sampling
    grid = np.array([xV.ravel(), yV.ravel()])

    # multiply by two so multiple defects goes to label N+i
    N = len(defect_types)
    label = np.zeros([N, p, q], dtype=np.float64)
    # initialize label
    for j, defect_type in enumerate(defect_types):
        if len(defects[defect_type]):
            # Projected positions
            positions = defects[defect_type]
            for position in positions:
                label[j, :, :] += label_function([position], grid, sigma).reshape(
                    [p, q]
                )
    if background:
        label = add_background_class(label)
    label_names = np.concatenate([defect_types, ["background"] * background])
    return np.transpose(label, axes=(0, 2, 1)), label_names


def create_labels(samples, parameters):
    """
    Create labels for a set of samples based on the specified parameters.

    This function generates labels for each sample in the input set, using either
    a defect column labeling approach or an atom spotting approach, depending on
    the specified label mode.

    Args:
        samples (list): A list of sample data, where each sample is expected to be
                        in a format compatible with the chosen labeling mode.
        parameters (dict): A dictionary containing various parameters that control
                           the label creation process. Expected keys include:
                           - 'mode': The labeling mode ('defects' or 'atomspotter')
                           - 'label_sigma': The sigma value for label generation
                           - 'label_type': The type of label function to use ('circ' or 'gauss')
                           - 'label_species': (optional) Species information for atom spotting

    Returns:
        tuple: A tuple containing two elements:
               1. torch.Tensor: A tensor of labels for all samples
               2. list: A list of label names corresponding to the label channels

    Note:
        The function supports two main labeling modes:
        1. 'defects': Uses the defect_column_label function to generate labels
        2. 'atomspotter': Uses the position_label function to generate labels

        The choice of label function (circular or Gaussian) is determined by the
        'label_type' parameter.
    """
    labels = []
    label_mode = parameters["mode"]
    sigma = parameters["label_sigma"]
    # get label function
    if parameters["label_type"]=="circ":
        label_function = circ_label
    elif parameters["label_type"]=="gauss":
        label_function = gaussian_label
    # create labels
    for i in range(len(samples)):
        if label_mode == "defects":
            label, label_names = defect_column_label(samples[i], parameters, sigma=sigma, label_function=label_function)
        elif label_mode == "atomspotter":
            if "label_species" in parameters:
                label, label_names = position_label(samples[i], parameters, sigma=sigma, label_function=parameters["label_type"], species=parameters["label_species"])
            else:
                label, label_names = position_label(samples[i], parameters, sigma=sigma, label_function=parameters["label_type"])
        labels.append(label)
    return torch.Tensor(np.array(labels)), label_names