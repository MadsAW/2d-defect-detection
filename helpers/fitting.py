from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def show_wave(
    image,
    sampling=None,
    quantile=None,
    vmax=None,
    ax=None,
    title=None,
    cmap="inferno",
    scalebar=False,
    colorbar=False,
    extent=None,
    unit="#",
    **kwargs,
):
    """
    Display an image with optional features similar to the Atomic Simulation Environment (ASE).

    This function shows a 2D image, taking into account sampling and transposing the image.
    It offers various customization options such as color scaling, title, scalebar, and colorbar.

    Parameters
    ----------
    image : numpy.ndarray
        2D array representing the image to be displayed.
    sampling : float, optional
        Sampling rate of each pixel in [Å/#]. If provided, affects the extent of the image.
    quantile : float, optional
        Scales the maximum color to a specific quantile. Useful for STEM images with extreme values.
        Default is None. For noisy images, consider using `quantile=0.999`.
    vmax : float, optional
        Maximum value for the color scale. If None and quantile is specified, vmax is set to the quantile value.
    ax : matplotlib.axes.Axes, optional
        Axes object to display the image. If None, a new figure and axes are created.
    title : str, optional
        Title for the plot.
    cmap : str, optional
        Colormap for the image. Default is 'inferno'.
    scalebar : bool, optional
        If True, displays a scalebar on the image.
    colorbar : bool, optional
        If True, displays a colorbar next to the image.
    extent : list or tuple, optional
        The extent of the image [left, right, bottom, top] in data coordinates.
    unit : str, optional
        Unit for the axes labels and scalebar. Default is '#'.
    **kwargs
        Additional keyword arguments passed to `plt.imshow`.

    Returns
    -------
    tuple
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes objects of the resulting plot.

    Notes
    -----
    - The image is transposed and displayed with origin='lower'.
    - If sampling is provided, the extent and unit are adjusted accordingly.
    - The function supports adding a scalebar and colorbar to the plot.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = None
    if sampling:
        extent_max = np.array(image.shape) * sampling
        extent = [0, extent_max[0], 0, extent_max[1]]
        unit = "Å"
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    if vmax is None:
        if quantile is not None:
            vmax = np.quantile(image, quantile)
    im = ax.imshow(
        image.T, extent=extent, cmap=cmap, origin="lower", vmax=vmax, **kwargs
    )
    if title is not None:
        ax.set_title(title)
    if scalebar:
        fontprops = fm.FontProperties(size=18)
        scalebar = AnchoredSizeBar(
            ax.transData,
            5,
            f"5 {unit}",
            "lower right",
            pad=0.2,
            color="white",
            frameon=False,
            size_vertical=0.1,
            fontproperties=fontprops,
        )

        ax.add_artist(scalebar)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)  # cax option later
    return fig, ax


def find_peaks(
    image, min_dist=1, threshold=None, kernel_size=None, show=False, ax=None
):
    """
    Find peaks in a given image by searching for local maxima.

    This function identifies peaks in an image using local maxima detection. It allows for
    customization of the peak detection process through minimum distance and threshold parameters.
    Optional Gaussian smoothing can be applied to reduce noise before peak detection.

    Parameters
    ----------
    image : numpy.ndarray
        2D array representing the image in which to find peaks.
    min_dist : int, optional
        Minimum number of pixels separating peaks. Smaller values detect more peaks. Default is 1.
    threshold : float, optional
        Minimum intensity of peaks. Only peaks with intensity greater than threshold will be detected.
        If None, no threshold is applied. Default is None.
    kernel_size : float, optional
        Standard deviation for Gaussian kernel used for image smoothing. If None, no smoothing is applied.
        Should be roughly the size of the features for optimal performance. Default is None.
    show : bool, optional
        If True, visualizes the image and detected peaks using the `show_wave` function. Default is False.
    ax : matplotlib.axes.Axes, optional
        If provided, the visualization will be displayed on this axes. Default is None.

    Returns
    -------
    numpy.ndarray
        An Nx2 array where each row represents the (x, y) coordinates of a detected peak.
        N is the number of peaks found.

    Notes
    -----
    - The function uses scipy.ndimage.gaussian_filter for smoothing and skimage.feature.peak_local_max for peak detection.
    - Visualization includes the number of peaks found and indicates if smoothing was applied.
    """
    # Perform gaussian smoothing
    if kernel_size:
        image = gaussian_filter(image, sigma=kernel_size)
    peaks = peak_local_max(image, min_distance=min_dist, threshold_rel=threshold)
    if show or ax:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        show_wave(
            image,
            title=f"{len(peaks)} peaks found{f' using smoothed image k={kernel_size}' if kernel_size else ''}",
            ax=ax,
        )
        ax.scatter(peaks[:, 0], peaks[:, 1])
    return peaks


## Find nearest neighbors
def find_nn(
    points, N_nn=20, nn_tol=0.6, distance=5, clustering="density", show=False, ax=None
):
    """
    Find vectors to the nearest neighbors for a group of points.

    This function identifies the nearest neighbor relationships within a set of points,
    clusters these relationships, and returns the representative vectors sorted by magnitude.

    Parameters
    ----------
    points : ndarray (Nx2)
        The input points for which to find nearest neighbor relations.
    N_nn : int, optional
        Number of nearest neighbors to consider for each point. Default is 20.
    nn_tol : float, optional
        Threshold for the fraction of points that must exhibit a specific nearest neighbor relation. Default is 0.6.
    distance : float, optional
        Distance threshold for clustering. Default is 5 pixels. Can be calibrated with sampling if needed.
    clustering : {'density', 'distance'}, optional
        Method for clustering nearest neighbor vectors. 
        'density': Uses DBSCAN algorithm, faster and better suited for most uses.
        'distance': Uses hierarchical clustering.
        Default is 'density'.
    show : bool, optional
        If True, visualizes the nearest neighbor vectors and the clustered results. Default is False.
    ax : matplotlib.axes.Axes, optional
        If provided, displays the visualization on this axes object. Default is None.

    Returns
    -------
    nn_vectors : ndarray (Mx2)
        Array of clustered nearest neighbor vectors, sorted by magnitude (shortest to longest).
        M is the number of unique nearest neighbor relationships identified.

    Notes
    -----
    - The function uses KDTree for efficient nearest neighbor searches.
    - Clustering helps identify consistent nearest neighbor relationships across the point set.
    - 'density' clustering is generally faster and more robust to strain-induced variations.
    """
    point_tree = KDTree(points, leafsize=100)

    nearest_neighbors = []
    for point in points:
        nearest_neighbors.append(point_tree.query(point, N_nn))
    nearest_neighbors = np.array(nearest_neighbors)

    # Find vectors
    vs = []
    for nn in nearest_neighbors:
        length, nn_i = nn
        nn_i = np.array(nn_i, dtype=int)
        nn_vectors = points[nn_i[1:]] - points[nn_i[0]]
        vs.append(nn_vectors)
    vs = np.concatenate(vs, axis=0)
    # Cluster vectors
    nn_vectors = []
    if clustering == "distance":
        c = fcluster(linkage(vs), t=distance, criterion="distance")
        for i in range(np.max(c)):
            # check that neighbour is present in 60%+ (tolerance) unitcells
            if np.sum(c == i + 1) >= len(points) * nn_tol:
                nn_vectors.append(np.mean(vs[c == i + 1], axis=0))
    elif clustering == "density":
        db = DBSCAN(eps=distance, min_samples=int(len(points) * nn_tol)).fit(vs)
        labels = db.labels_
        for i in range(np.max(labels) + 1):
            nn_vectors.append(np.mean(vs[labels == i], axis=0))
    else:
        raise ValueError(
            f"Unrecognized clustering option {clustering}. Should be 'distance' or 'density'."
        )
    nn_vectors = np.array(nn_vectors)

    if show or ax:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(vs[:, 0], vs[:, 1])
        ax.scatter(nn_vectors[:, 0], nn_vectors[:, 1])
        ax.legend(["All nearest neighbours", "Clustered nearest neighbours"])
        ax.set_aspect("equal", "box")
    # return sorted list
    return nn_vectors[np.argsort(np.linalg.norm(nn_vectors, axis=1))]


## Lattice tools
def in_bounding_box(points, dimensions):
    """
    Checks if a set of points is within a bounding box centered at the origin.

    This function determines whether each point in a given set of points falls within
    a bounding box. The bounding box is defined as extending from -dimension/2 to
    +dimension/2 in each dimension.

    Parameters
    ----------
    points : numpy.ndarray
        An Nx2 array where each row represents a point's (x, y) coordinates.
    dimensions : numpy.ndarray
        A 1D array of length 2 specifying the dimensions of the bounding box.
        For example, this could be the shape of an image.

    Returns
    -------
    numpy.ndarray
        A boolean array of length N, where True indicates that the corresponding
        point is within the bounding box, and False indicates it is outside.

    Notes
    -----
    This function currently works only for 2D points and rectangular bounding boxes.
    Future improvements could extend this to work with arbitrary polygons or
    higher dimensions.

    Example
    -------
    >>> points = np.array([[0, 0], [5, 5], [-3, 2]])
    >>> dimensions = np.array([10, 8])
    >>> in_bounding_box(points, dimensions)
    array([ True,  True,  True])
    """
    mask = (
        (points[:, 0] > -dimensions[0] / 2)
        & (points[:, 0] < dimensions[0] / 2)
        & (points[:, 1] > -dimensions[1] / 2)
        & (points[:, 1] < dimensions[1] / 2)
    )
    return mask


def closest_node(node, nodes):
    """
    Find the closest node in a set of nodes to a given node.

    This function calculates the Euclidean distance between the given node and all nodes
    in the provided set, then returns the node with the smallest distance.

    Parameters
    ----------
    node : array-like
        The reference node, typically a 1D array or list of coordinates.
    nodes : array-like
        A set of nodes to search through, typically a 2D array where each row
        represents a node's coordinates.

    Returns
    -------
    array-like
        The node from the 'nodes' set that is closest to the reference 'node'.

    Notes
    -----
    This function uses scipy's cdist function for efficient distance calculation.
    It assumes that 'node' and each element in 'nodes' have the same dimensionality.

    Example
    -------
    >>> reference = [0, 0]
    >>> node_set = [[1, 1], [2, 2], [-1, 0]]
    >>> closest_node(reference, node_set)
    array([-1,  0])
    """
    from scipy.spatial.distance import cdist
    closest_index = cdist([node], nodes).argmin()
    return nodes[closest_index]


def ideal_lattice(dimensions, lattice_vectors, snapto=None):
    """
    Create an ideal lattice for a rectangular area defined by given dimensions.

    This function calculates the most effective lattice replications and returns
    a full lattice within the specified bounding box. It can optionally snap the
    lattice to a given point.

    Parameters
    ----------
    dimensions : array-like, shape (2,)
        The dimensions of the rectangular area to create an ideal lattice for.
        For example, [width, height].
    lattice_vectors : array-like, shape (2, 2)
        The lattice vectors to use for creating the ideal lattice.
        Each row represents a lattice vector.
    snapto : array-like, shape (2,), optional
        Coordinates to snap the ideal lattice to. If provided, the lattice
        will be shifted so that the nearest lattice point aligns with these
        coordinates.

    Returns
    -------
    lattice_points : ndarray, shape (N, 2)
        An array of all lattice points inside the bounding rectangle.
        Each row represents the (x, y) coordinates of a lattice point.
    """
    lattice_vectors = np.array(lattice_vectors)
    ## Check that all have dim 2x2 or 2 (im)
    dimensions = np.array(dimensions)
    ## finding the ideal lattice for a square (could be any shape we would just need more projections I think)
    # define corner points
    points = np.array(
        [
            [-dimensions[0] / 2, -dimensions[1] / 2],
            [dimensions[0] / 2, -dimensions[1] / 2],
            [dimensions[0] / 2, dimensions[1] / 2],
            [-dimensions[0] / 2, dimensions[1] / 2],
        ]
    )
    # find linear combination of vectors to each of the points

    projections = np.linalg.solve(lattice_vectors.T, np.array(points).T)

    # find the minimum and maximum repetitions for each of the unit cell vectors
    repetitions = np.array(
        [np.floor(np.min(projections, axis=1)), np.ceil(np.max(projections, axis=1))],
        dtype=int,
    ).T

    lattice_points = []
    # notice -1 since negative values ceil are actually floored and +1 due to the loop/0 indexing
    for i in range(repetitions[0, 0], repetitions[0, 1] + 1):
        for j in range(repetitions[1, 0], repetitions[1, 1] + 1):
            lattice_points.append(lattice_vectors[0] * i + lattice_vectors[1] * j)
    lattice_points = np.array(lattice_points)
    # calculate offset
    if snapto is not None:
        snapto = np.array(
            snapto.copy(), dtype=float
        )  # we do not want to alter entry if passed from list
        snapto -= dimensions / 2  # center
        # find nearest point
        # find lower left position
        nn = closest_node(snapto, lattice_points)
        lattice_points += snapto - nn
        # translate back to middle

    # create mask for lattice vectors inside area
    mask_inside = in_bounding_box(lattice_points, dimensions)
    # add offset and center at middle of input dimensions
    lattice_points += dimensions / 2
    # lastly remove lattice_points which were outside original are
    return lattice_points[mask_inside]


# fit lattice to points
def fit_lattice(
    lattice,
    peaks,
    snapto=None,
    tolerance=10,
    N_nn=4,
    show=False,
    ax=None,
    verbose=False,
):
    """
    Iteratively fits a lattice to a set of peaks using a tolerance-based approach and a specified starting point.

    This function uses an iterative process to fit lattice sites to nearby peaks, starting from a specified point
    and expanding to neighboring sites. It uses a KDTree to efficiently find nearest neighbors in the lattice.
    The lattice is adjusted during the fitting process, with newly locked points causing slight shifts in
    neighboring lattice sites.

    Parameters
    ----------
    lattice : ndarray (Nx2)
        The initial lattice structure, typically created using the `ideal_lattice` function.
    peaks : ndarray (Nx2)
        The set of peak coordinates extracted from the image to fit the lattice to.
    snapto : ndarray (2) or int, optional
        Starting point for the fitting process, either as coordinates or an index in the peaks array.
        If None, the point with the smallest offset between lattice and peaks is used.
    tolerance : float, optional
        Maximum distance between a lattice point and a peak for the lattice point to be considered "locked".
    N_nn : int, optional
        Number of nearest neighbors to consider in the lattice for each point.
    show : bool, optional
        If True, visualizes the fitted lattice, missing atoms, and used peaks. Default is False.
    ax : matplotlib.axes.Axes, optional
        Specific axes for displaying the visualization. Used only if show is True.
    verbose : bool, optional
        If True, prints the number of peaks fitted during the process.

    Returns
    -------
    lattice : ndarray (Nx2)
        The fitted lattice points, including both locked and unlocked points.
    locked : ndarray (bool, shape=(N,))
        Boolean array indicating which lattice points were successfully locked to peaks.
    peaks_found : ndarray (bool, shape=(M,))
        Boolean array indicating which input peaks were matched to lattice points.

    Notes
    -----
    - The function uses a KDTree for efficient nearest neighbor searches.
    - The fitting process is iterative, expanding from the initial point to neighboring sites.
    - Lattice points are "locked" when they are within the specified tolerance of a peak.
    - The process continues until no more peaks can be fitted within the tolerance.
    """
    # NN tree
    # want neighbours in the lattice but find distances from lattice to the real points
    nn_tree = KDTree(lattice, leafsize=100)
    _, neighbours = nn_tree.query(lattice, k=N_nn + 1)
    NN = neighbours[:, 1:]
    # only search points near locked points
    peak_tree = KDTree(peaks, leafsize=100)

    if snapto is None:
        dists, indices = peak_tree.query(lattice)
        snapto = indices[np.argmin(dists)]
    else:
        if not type(snapto) == int:
            # find index closest to the snapto point
            _, snapto = peak_tree.query(snapto)
    lattice = lattice.copy()

    # which points are locked in the grid
    locked = np.array([False] * len(lattice))
    searched = np.array([False] * len(lattice))
    locked[snapto] = True
    searched[snapto] = True

    peak_found = np.array(
        [False] * len(peaks), dtype=bool
    )  # found peaks should end up populated by all peaks

    to_search = []
    chosen_neigbours = NN[snapto]
    chosen_neigbours[locked[chosen_neigbours] == False]
    to_search.extend(chosen_neigbours[searched[chosen_neigbours] == False])
    searched[chosen_neigbours] = True

    # begin to fit lattice to the peaks
    peaks_to_check = True
    while peaks_to_check:
        dists, index = peak_tree.query(lattice[to_search])
        best_fit = np.argmin(
            dists
        )  ## HMM BEST FIT would be the first one where the peak is not already found!
        peaks_best_fit = index[best_fit]
        lat_best_fit = to_search[best_fit]
        # update to search
        if dists[best_fit] < tolerance:  # within tolerance
            locked[lat_best_fit] = True
            peak_found[peaks_best_fit] = (
                True  # what if we find the same peak several times???? - suppose tolerance solves this!
            )
            # add new unsearched neighbour nodes to search
            chosen_neigbours = NN[lat_best_fit]
            to_search.extend(chosen_neigbours[searched[chosen_neigbours] == False])
            searched[chosen_neigbours] = True

            ## Mesh deformation
            # update lattice location and move all unlocked neighbours
            delta = peaks[peaks_best_fit] - lattice[lat_best_fit]
            lattice[lat_best_fit] = peaks[peaks_best_fit]
            unlocked_NN = chosen_neigbours[locked[chosen_neigbours] == False]
            for nn in unlocked_NN:
                lattice[nn] += delta / 2
                # and we could choose to do neighbour's neigbour if we wanted

            # remove current locked point from the to search list
            del to_search[best_fit]

        # if no peaks fall within tolerances then we break out loop and have found the amount of peaks we could!
        else:
            if verbose:
                print(
                    f"No more peaks fall in tolerance, found {np.sum(peak_found)}/{len(peak_found)}"
                )
            break
        # break loop when all peaks are categorized
        if peak_found.all():
            peaks_to_check = False

    # visualize
    if show or ax:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(lattice[locked, 0], lattice[locked, 1], color="blue")
        ax.scatter(lattice[~locked, 0], lattice[~locked, 1], color="black")
        ax.scatter(peaks[:, 0], peaks[:, 1], color="red", marker="2")
        ax.legend(["Locked lattice", "Missing atoms", "Peaks"])

    return lattice, locked, peak_found
