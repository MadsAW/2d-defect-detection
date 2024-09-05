from skimage.draw import disk
from scipy.ndimage import rotate
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from .fitting import find_nn, ideal_lattice, fit_lattice, show_wave

def find_sampling(signal):
    """
    Calculate the sampling rate (Å/pixel) for the microscope used in experiments.

    This function computes the sampling rate based on the microscope's calibration
    and the current magnification settings. It adjusts for the capture resolution
    of the image.

    Args:
        signal (hyperspy.signals.Signal2D): The signal object containing the image
                                            and metadata.

    Returns:
        float: The calculated sampling rate in Å/pixel.

    Note:
        This function assumes specific metadata structure and calibration values.
        Adjust these if using a different microscope or setup.
    """
    calibration = 0.0631*8e6 #Å*Mx/pixel
    calibration *= (2096/signal.data.shape[1]) # calibration depending on capture resolution
    sampling = calibration/(signal.metadata["Acquisition_instrument"]["TEM"]["magnification"]+1.5e6) # Å/pixel
    return sampling

def rotatedRectWithMaxArea(w, h, angle):
    """
    Compute the dimensions of the largest axis-aligned rectangle within a rotated rectangle.

    Given a rectangle of size wxh that has been rotated by 'angle' (in radians),
    this function calculates the width and height of the largest possible
    axis-aligned rectangle that can fit inside the rotated rectangle.

    Args:
        w (float): Width of the original rectangle.
        h (float): Height of the original rectangle.
        angle (float): Rotation angle in radians.

    Returns:
        tuple: A tuple containing the width and height (in that order) of the
               largest axis-aligned rectangle within the rotated rectangle.

    Note:
        This implementation is based on the solution provided in the following
        Stack Overflow discussion:
        https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def rotate_and_cut(image, angle):
    """
    Rotate an image by a specified angle and crop it to remove black borders.

    This function rotates the input image by the given angle plus 90 degrees,
    then crops the rotated image to remove any black borders that result from
    the rotation. It uses the rotatedRectWithMaxArea function to determine
    the optimal crop size.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image to be rotated and cropped.
    angle : float
        The rotation angle in degrees. Note that 90 degrees will be added to this angle.

    Returns:
    --------
    numpy.ndarray
        The rotated and cropped image.
    """
    rotated = rotate(image, angle + 90)
    shape = rotated.shape

    # find limits to cut the rotated image
    lims = int(
        (shape[0] - rotatedRectWithMaxArea(image.shape[0], image.shape[1], angle)[0])
        / 2
    )

    return rotated[lims : (shape[0] - lims), lims : (shape[1] - lims)]


def model_predict(
    im, network, scale=None, cutout=False, power=None, mask=None, astype=torch.int32
):
    """
    Given an image(s) and a neural network the function will preprocess the image(s) and
    perform a prediction using the network using GPU.
    Parameters
    ----------
    im : np.array or torch.tensor
        Input is either a numpy array or torch tensor of shapes [N_x,N_y]. Alternatively,
        an array of images can also predicted all at once with the shape [N_ims, N_x, N_y].

    network : torch.model
        The model used to make the prediction. The model should already be loaded onto the GPU.

    scale : int, optional
        To get better predictions it is often helpful to rescale the input images by a scaling
        factor. The images will be rescaled to scale*im.shape[-2:]. Alternatively, part of the
        images can be cutout to not increase the memory requirement. Default is None.

    cutout : bool, optional
        When true the upper left corner of the images will be cut out to match the previous
        dimensions of the image (to lessen memory requirements). Default is False

    power : float, optional
        It can sometimes be helpful to apply a power to the image before normalization to
        slightly change the contrast. Default is False.

    astype : type
        The type to interpret the data as. In most cases this will be either
        torch.int32 or torch.float32. Default is torch.int32
    Returns
    -------
    prediction : torch.tensor
        Returns the prediction of the network on the input images with dimensions
        [N_ims, N_channels, N_x, N_y].

    Example
    >>> positions = model_predict(image, atomspotter)[0,0,:,:] # get the position map for the first image
    """
    # make sure not to alter original object
    im = im.copy()
    im = torch.tensor(im, dtype=astype)

    if len(im.shape) == 2:
        im = im.unsqueeze(0)
    # normalize with mask
    im = im - torch.min(im.view(im.shape[0], -1), axis=1).values[:, None, None]
    im = im / torch.max(im.view(im.shape[0], -1), axis=1).values[:, None, None]
    if scale:
        shape = im.shape
        im = torchvision.transforms.Resize(int(shape[1] * scale))(im)
        if cutout:
            im = im[:, : shape[1], : shape[2]]  # change to act from middle
    if power:
        im = torch.pow(im, power)
    prediction = network(im.unsqueeze(1).cuda()).cpu().detach().numpy()
    return prediction


def col_intensities(im, peaks, mask_size):
    """
    Extract the intensity around a peak using a circular mask of radius mask_size
    """
    intensities = []
    im = np.array(im)
    for peak in peaks:
        # find mask center
        mask = np.zeros(im.shape, dtype=bool)
        rr, cc = disk(peak, mask_size, shape=im.shape)
        mask[rr, cc] = 1
        intensities.append(np.sum(im[mask]))
    return np.array(intensities)


## Fit to lattice the function of interest
def lock_peaks_to_lattice(
    peaks, im, threshold=0.7, nn_distance=5, tolerance=10, show=False
):
    """
    Locks a list of peaks to a CrSBr lattice by using the structural knowledge of
    CrSBr (hardcoded). This is done by finding nearest neighbours, performing clustering
    to get lattice vectors, creating and ideal lattice and lastly "annealing" the lattice
    points to the real peaks. The function also uses the image to classify which column
    is which and for plotting reasons.
    Parameters
    ----------
    peaks : Nx2 np.array
        The peak positions to refine (should be in #pixels)
    image : 2D np.array
        The image to use for refining the peak positions
    threshold : float, optional
        Changes the snap to point until the amount of peaks found exceed the threshold.
        Default is 0.7 corresponding to 70% of all peaks must be found.
    nn_distance : float, optional
        The distance to use while clustering points for lattice analysis.
        Default is 5.
    tolerance : float, optional
        The tolerance of how close a lattice must be to a peak in order to be locked during iterations.
        Default is 10 pixels.
    show : bool, optional
        Visualizes the found lattice on top of the images and all intermediate
        steps. Default is False.
    Returns
    -------
    Cr : ndarray (Nx3)
        Chromium positions in a list of [x_pos, y_pos, 0] (0 being a class for atomai integration)
    SBr : ndarray (Nx3)
        Sulfur Bromine positions in a list of [x_pos, y_pos, 0] (0 being a class for atomai integration)
    lattice_vectors : ndarray (2x2)
        The lattice vectors extracted during the function
    Example
    >>> Cr, SBr, lat_vectors = lock_peaks_to_lattice(peaks, im, show=True)
    """
    nn_vectors = find_nn(peaks, distance=nn_distance, show=show)
    # Extract lattice vectors, knowing CrSBr structure
    ## Note material dependant!
    motifs = nn_vectors[[0, 2]]
    lat_vectors = nn_vectors[[4, 6]]
    # create an ideal lattice
    lattice = ideal_lattice(im.shape, lat_vectors, snapto=peaks[2])

    # adding motifs
    ## the indexes for the different sublattices
    sublattice1 = np.arange(len(lattice))
    sublattice2 = np.arange(len(lattice), len(lattice) * 2)
    ## entire lattice (points)
    crystal = np.concatenate([lattice, lattice + motifs[0]])

    # fit the peaks to the lattice
    ## Choose random snap to until threshold is satisfied
    snapto = 0
    while True:
        relaxed_crystal, locked, peak_found = fit_lattice(
            crystal, peaks, snapto=snapto, tolerance=tolerance, show=show
        )
        if np.sum(peak_found) > threshold * len(peak_found):
            break
        # else try another snap to point
        snapto += 1

    # take crystal position and give integer value to extract pixel in real image
    crystal2int = np.array(relaxed_crystal, dtype=int)
    # for different classes / columns sort these
    ## Could also use Voronoi instead which would be slightly slower
    if np.median(im[crystal2int[sublattice1[locked[sublattice1]]]]) < np.median(
        im[crystal2int[sublattice2[locked[sublattice2]]]]
    ):
        Cr = sublattice1[locked[sublattice1]]
        SBr = sublattice2[locked[sublattice2]]
    else:
        Cr = sublattice2[locked[sublattice2]]
        SBr = sublattice1[locked[sublattice1]]
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        show_wave(im, ax=ax)
        ax.scatter(relaxed_crystal[Cr, 0], relaxed_crystal[Cr, 1])
        ax.scatter(relaxed_crystal[SBr, 0], relaxed_crystal[SBr, 1])
        ax.legend(["Cr", "SBr"])

    # Convert to list of [x_pos, y_pos, 0]
    Cr = relaxed_crystal[Cr, :]
    SBr = relaxed_crystal[SBr, :]
    return Cr, SBr, lat_vectors
