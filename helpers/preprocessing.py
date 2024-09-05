from typing import List, Optional, Tuple, Union
from collections.abc import Sequence
from torchvision.utils import _log_api_usage_once
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import torchvision.transforms.functional as F
from torchvision.transforms.functional import (
    _interpolation_modes_from_int,
    InterpolationMode,
)
from torch import Tensor
from noise import snoise2
import hyperspy.api as hs
import numpy as np
import warnings
import random
import torch
import math
import os

from .helpers import draw_uniform
from .labels import create_labels
from .processing import find_sampling


def normalize(image):
    """
    Normalizes a torch image to the range [0, 1].

    This function scales the input image linearly such that the minimum pixel value
    becomes 0 and the maximum pixel value becomes 1.

    Args:
        image (torch.Tensor): The input image tensor to be normalized.

    Returns:
        torch.Tensor: The normalized image tensor with values in the range [0, 1].
    """
    image = image - torch.min(image)
    image = image / torch.max(image)
    return image


def create_contamination(shape, octaves=8, seed=None, **kwargs):
    """
    Generate a background image using simplex noise with a given shape.

    This function creates a 2D array of simplex noise, which can be used to simulate
    contamination or background patterns in images.

    Args:
        shape (tuple): The shape of the output array (height, width).
        octaves (int, optional): The number of octaves for the simplex noise. Default is 8.
        seed (int, optional): Seed for random number generation. If None, a random seed is used.
        **kwargs: Additional keyword arguments to pass to the snoise2 function.

    Returns:
        numpy.ndarray: A 2D array of simplex noise with values approximately in the range [0, 1].

    Example:
        To generate a stack of contamination images:

        ```python
        from helpers.preprocessing import create_contamination
        import numpy as np

        contamination_stack = []
        N = 100
        for i in range(N):
            contamination_stack.append(create_contamination((256, 256), octaves=6))

        contamination_stack = np.array(contamination_stack)
        np.savez('contamination_stack.npz', backgrounds=contamination_stack)
        ```
    """
    # random base for creating random patterns
    if not seed:
        seed = np.random.randint(0, 2**10)
    background = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            background[i, j] = snoise2(
                i / shape[0], j / shape[1], octaves=octaves, base=seed, **kwargs
            )
            # for parameters view https://github.com/caseman/noise/blob/master/_simplex.c
    # Normalize approximately bt. 0-1
    background = (background - background.min()) / np.sqrt(2)
    return background


# Custom transformations
class Normalize(object):
    """Normalizes the image to range [0, 1].

    This class provides a callable object that normalizes input images to the range [0, 1].
    It can be used as a transformation in data preprocessing pipelines.

    Args:
        localized_size (float, optional): Size of the local region (in pixels) used for
            normalization. If None, global normalization is applied. Default is None.

    Methods:
        __call__(image): Normalizes the input image.

    Returns:
        numpy.ndarray: Normalized image with values in the range [0, 1].

    Note:
        The actual normalization is performed by the `normalize` function, which should
        be defined elsewhere in the code.
    """

    def __init__(self, localized_size=None):
        pass

    def __call__(self, image):
        return normalize(image)


class RandomRot(object):
    """Rotate the input image by a randomly chosen angle from a predefined set.

    This class provides a callable object that rotates an input image by a randomly
    selected angle from a list of predefined angles. It can be used as a transformation
    in data augmentation pipelines.

    Args:
        angles (list): A list of angles (in degrees) to choose from for rotation.

    Methods:
        __call__(x): Applies the random rotation to the input image.

    Returns:
        torch.Tensor: The rotated image.

    Note:
        This class uses torchvision.transforms.functional.rotate for the actual rotation.
    """

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class EllipticalGaussianBlur(object):
    """Applies an elliptical Gaussian blur to the input image.

    This class creates a callable object that blurs an input image using an elliptical
    Gaussian kernel. The blur is applied with different standard deviations (sigmas)
    in the x and y directions, allowing for anisotropic blurring.

    Args:
        kernel_size (list): Size of the Gaussian kernel. Should be a list of two integers.
        sigmas_x (list): Range of sigma values for x-direction. Should be a list of two floats.
        sigmas_y (list): Range of sigma values for y-direction. Should be a list of two floats.

    Methods:
        __call__(img): Applies the elliptical Gaussian blur to the input image.

    Returns:
        torch.Tensor: The blurred image.

    Example:
        blur = EllipticalGaussianBlur([5, 5], [0, 0.1], [2, 3])
        blurred_img = blur(input_img)

    Note:
        This class uses torchvision.transforms.functional.gaussian_blur for the actual blurring.
        The sigma values for x and y directions are randomly chosen from their respective ranges
        for each call.
    """

    def __init__(self, kernel_size, sigmas_x, sigmas_y):
        self.kernel_size = kernel_size
        self.sigmas_x = sigmas_x
        self.sigmas_y = sigmas_y

    def __call__(self, img):
        sigma_x = np.random.uniform(self.sigmas_x[0], self.sigmas_x[1])
        sigma_y = np.random.uniform(self.sigmas_y[0], self.sigmas_y[1])
        return TF.gaussian_blur(img, self.kernel_size, [sigma_x, sigma_y])


# Loading experimental data helper function
def load_experimental_data(directory, file, in_channels=1, N_sum=-1, N_start=0):
    """
    Load experimental data from a file and return the image and sampling rate.

    This function supports loading data from .npy files and other formats supported by HyperSpy.
    For .npy files, it assumes the sampling rate is 1 pixel. For other formats, it uses the
    find_sampling function to determine the sampling rate.

    Args:
        directory (str): The directory containing the file.
        file (str): The name of the file to load.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        N_sum (int, optional): Number of frames to sum. If -1, sum all frames. Defaults to -1.
        N_start (int, optional): Starting frame for summing. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - im (numpy.ndarray): The loaded image data.
            - sampling (float): The sampling rate of the image.

    Raises:
        ValueError: If the file format is not supported or if there's an error in loading the file.
    """
    name, ext = os.path.splitext(file)
    if ext == ".npy":
        im = np.load(directory + file)
        sampling = 1  # just consider image as pixels
    else:
        s = hs.load(directory + file)
        if type(s) == list:
            for signal in s:
                if signal.metadata["General"]["title"] == "DCFI":
                    im = np.sum(signal.data[N_start:N_sum], axis=0)
                    sampling = find_sampling(signal)
        elif len(s.data.shape) == 3:
            im = np.sum(s.data[N_start:N_sum], axis=0)
            sampling = find_sampling(s)
        else:
            im = s.data
            sampling = find_sampling(s)
    return im, sampling


def load_data(parameters, dataname="dataset"):
    """
    Load .npz files in a given directory as a single dataset for training.

    Args:
        parameters (dict): A dictionary containing configuration parameters.
        dataname (str, optional): Base name of the dataset files. Defaults to "dataset".

    Returns:
        tuple: A tuple containing:
            - features (numpy.ndarray): Concatenated measurements from all files.
            - labels (torch.Tensor): Concatenated labels from all files.
            - targets (numpy.ndarray): Concatenated targets (samples, defect locations, or placeholders).

    Raises:
        ValueError: If no matching files are found or if label structures differ between files.

    Note:
        This function expects .npz files named as '{dataname}_{i}.npz' where i is an incrementing integer.
        It loads and concatenates data from all matching files in the directory specified by parameters["data_path"].
    """
    features = []
    labels = []
    targets = []
    i = 0
    while True:
        i += 1
        if os.path.isfile(f'{parameters["data_path"]}/{dataname}_{i}.npz'):
            # load data from file
            data = np.load(
                f'{parameters["data_path"]}/{dataname}_{i}.npz', allow_pickle=True
            )
            features.append(data["measurements"])
            labels.append(data["labels"])
            if parameters["mode"] == "atomspotter":
                targets.append(data["samples"])
            elif parameters["mode"] == "defects":
                targets.append(data["defect_locations"])
            else:
                targets.append([0])
                pass  # ignore if the target is unknown
            if i == 1:
                label_names = data["parameters"].item()["label_names"]
            if np.array_equal(label_names, data["parameters"].item()["label_names"]):
                label_names = data["parameters"].item()["label_names"]
            else:
                raise ValueError(
                    f"Label structure differs at indexes {i} and {i-1}!\n{i}: {data['parameters'].item()['label_names']} \n{i-1}: {label_names}"
                )
        else:
            if i == 1:
                ValueError(
                    f"Seems that no matching files are found: {parameters['data_path']}{dataname}_[i]"
                )
            break

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    targets = np.concatenate(targets, axis=0)
    labels = torch.Tensor(labels)
    assert len(labels) == len(features)
    if parameters["mode"] in ["atomspotter", "defects"]:
        assert len(targets) == len(features)

    # Extract parameters from data and alter original parameters
    pars = data["parameters"].item()
    parameters["scanning_sampling"] = pars["scanning_sampling"]
    parameters["size"] = pars["size"]
    parameters["label_names"] = label_names

    return features, labels, targets


def generate_dataset(
    parameters,
    contamination_stack=None,
    N_augmentations=10,
    save=False,
    dataname="validation",
    save_name="validation_set",
):
    """
    Generate a dataset with augmentations for training or validation.

    This function loads data, applies various data augmentation techniques, and creates
    a dataset suitable for training or validation of deep learning models for STEM image analysis.

    Args:
        parameters (dict): A dictionary containing various parameters for data generation and augmentation.
        contamination_stack (numpy.ndarray or str, optional): Stack of contamination images or path to the file.
        N_augmentations (int, optional): Number of augmentations to apply to each image. Defaults to 10.
        save (bool, optional): Whether to save the generated dataset. Defaults to False.
        dataname (str, optional): Name of the dataset to load. Defaults to "validation".
        save_name (str, optional): Name to use when saving the dataset. Defaults to "validation_set".

    Returns:
        tuple: A tuple containing:
            - dataset (STEMDataset): The generated dataset with augmentations.
            - parameters (dict): Updated parameters dictionary.

    Raises:
        ValueError: If the contamination stack is not properly specified.

    Note:
        This function applies various transformations including rotation, flipping,
        resizing, blurring, and normalization. It also handles label changes if specified
        in the parameters.
    """
    # Load datasets
    features, labels, targets = load_data(parameters, dataname=dataname)
    if type(contamination_stack) == str:
        contamination_stack = np.load(
            parameters["contamination_stack"], allow_pickle=True
        )["backgrounds"]
    # Define tranformations for data augmentation
    generate_image = GenerateImage(
        dose=parameters["dose"],
        contamination_str=parameters["contamination_str"],
        background_str=parameters["background_str"],
        contamination_stack=contamination_stack,
        sampling=parameters["scanning_sampling"],
    )
    rotate = RandomRot([0, 90, 180, 270])
    vflip = T.RandomVerticalFlip()
    hflip = T.RandomHorizontalFlip()
    size = np.array(parameters["resize_size"])
    scale = size / features.shape[1:]
    scale *= parameters["resize_scale"]
    crop = T.RandomResizedCrop(
        size=list(size), scale=list(scale), ratio=parameters["resize_ratio"]
    )
    kernel_size = [5, 5]
    blur = EllipticalGaussianBlur(
        kernel_size, parameters["blur_x"], parameters["blur_y"]
    )
    normalization = Normalize()

    composed = T.Compose([rotate, crop, blur, vflip, hflip, normalization])
    composed_label = T.Compose([rotate, crop, vflip, hflip])

    ## Change label function
    change_label_function = parameters["label_change"]

    if change_label_function:
        labels, label_names = create_labels(targets, parameters)
        del targets

    dataset = STEMDataset(
        features,
        labels,
        transform=composed,
        label_transform=composed_label,
        make_image=generate_image,
    )

    # generate the validation dataset
    N = len(features)
    features = np.zeros(
        (N * N_augmentations, 1, *parameters["resize_size"]), dtype=np.float32
    )
    labels = np.zeros(
        (N * N_augmentations, labels.shape[1], *parameters["resize_size"]),
        dtype=np.float32,
    )

    for j in range(N_augmentations):
        dataloader = iter(dataset)
        for i in range(N):
            img, label = next(dataloader)
            features[j * N + i, :, :, :] = img
            labels[j * N + i, :, :, :] = label

    ## Save data
    if save:
        np.savez_compressed(
            os.path.join(f'{parameters["data_path"]}', f"{save_name}.npz"),
            labels=labels,
            measurements=features,
        )
    return torch.Tensor(features), torch.Tensor(labels)


def generate_contamination_stack(parameters, shape, N=100, save=False, **kwargs):
    """
    Generate a stack of contamination images.

    This function creates a stack of contamination images using the provided parameters.
    Each image in the stack represents a different contamination pattern.

    Args:
        parameters (dict): A dictionary containing configuration parameters.
        shape (tuple): The shape of each contamination image (height, width).
        N (int, optional): The number of contamination images to generate. Defaults to 100.
        save (bool, optional): Whether to save the generated stack. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        numpy.ndarray: A stack of contamination images.

    If save is True, the stack is saved as a .npz file in the directory specified
    by parameters["data_path"].
    """
    contamination_stack = []
    for i in range(N):
        contamination_stack.append(
            create_contamination(
                shape,
                octaves=parameters["octaves"],
                persistence=parameters["persistence"],
            )
        )
    contamination_stack = np.array(contamination_stack)
    if save:
        np.savez(
            os.path.join(f'{parameters["data_path"]}', "contamination_stack.npz"),
            backgrounds=contamination_stack,
        )
    return np.array(contamination_stack)

class RandomResizedCrop(torch.nn.Module):
    """Crop a random portion of image and resize it to a given size.

    If the image is a torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train Inception networks.

    Args:
        size (int or sequence): Expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            Note: In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.

        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): Lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing. Default is None.
            It only affects **tensors** with bilinear or bicubic modes and is ignored otherwise:
            - On PIL images, antialiasing is always applied for bilinear or bicubic modes.
            - On other modes (for PIL images and tensors), antialiasing is ignored.

            Possible values are:
            - ``True``: Apply antialiasing for bilinear or bicubic modes on tensors.
            - ``False``: Do not apply antialiasing for tensors on any mode.
            - ``None``: Equivalent to ``False`` for tensors and ``True`` for PIL images.

            The default will change to ``True`` in v0.17 for consistency across PIL and Tensor backends.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.size = size

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
        img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = img.shape
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            (height, width): height width of cut before resize. Needed to apply dose
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return (h, w), F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias})"
        return format_string


def shot_noise(img, dose, sampling):
    """Adds shot noise to an image based on the electron dose per area.

    This function simulates the effect of shot noise in electron microscopy images
    by applying Poisson noise to the input image. The noise level is determined
    by the electron dose and the image sampling rate.

    Args:
        img (numpy.ndarray): Input image to add shot noise to.
        dose (float): Electron dose used in the measurement, given in e/Å².
        sampling (float): Sampling rate of the image in Å/pixel.

    Returns:
        numpy.ndarray: Image with added shot noise.

    Note:
        The function assumes that the input image intensity is proportional
        to the electron count. The output image will have Poisson-distributed
        pixel values based on the input image scaled by the dose and sampling area.
    """
    return np.random.poisson(img * dose * sampling**2)


# new dataset function
## Here we do augmentations first and afterwards we will do our Generate image


class GenerateImage(object):
    """Transforms a measurement into an image with different noise sources
    emulating expected outputs from STEM experiments.

    This class models various noise sources typically encountered in STEM experiments,
    including electron counting noise, background noise from stray electrons or detector,
    and contamination. The contrast of each resulting image is influenced by the
    relationship between dose, background strength, and column weight.

    Args:
        dose (tuple of float, optional): Range of electron doses (e/Å²) to simulate
            electron counting noise (Poisson noise). Doses are uniformly distributed
            within this range.
        background_str (tuple of float, optional): Range for the background fraction
            of the dose to apply, simulating imperfect microscope conditions. 
            Requires 'dose' to be specified.
        contamination_str (tuple of float, optional): Range for the contamination
            fraction of image intensity to add to the sample. Requires
            'contamination_stack' to be specified.
        contamination_stack (numpy.ndarray, optional): Stack of contamination images,
            often simulated using `create_contamination`. Required if
            'contamination_str' is specified.

    Note:
        The 'sampling' parameter (float, in Å/pixel) is not an initialization
        parameter but is required in the __call__ method to calibrate the dose
        to the image.
    """

    def __init__(
        self,
        dose=None,
        background_str=None,
        contamination_str=None,
        contamination_stack=None,
    ):
        self.dose = dose
        self.background_str = background_str
        self.contamination_str = contamination_str
        self.contamination_stack = contamination_stack
        if (contamination_stack is None) != (contamination_str is None):
            raise ValueError(
                "Contamination strength and/or stack specified without the other"
            )

    def __call__(self, measurement, sampling):
        """
        Returns an array representing the measurement with noise added
        """
        if self.contamination_str is not None:
            contamination_str = draw_uniform(self.contamination_str)
            chosen_contamination = self.contamination_stack[
                np.random.randint(self.contamination_stack.shape[0])
            ]
            background = chosen_contamination * contamination_str * np.max(measurement)
            measurement += background

        if self.dose is not None:
            dose = draw_uniform(self.dose)
            measurement = shot_noise(measurement, dose, sampling)

        if self.background_str is not None:
            if self.dose is None:
                raise ValueError("No dose supplied!")
            background_str = draw_uniform(self.background_str)
            electrons_per_pixel = background_str * dose * (sampling**2)
            noise = np.random.poisson(electrons_per_pixel, measurement.shape)
            measurement += noise

        img = torch.Tensor(measurement).unsqueeze(0)
        return img


class STEMDataset(Dataset):
    """
    STEM image dataset for segmentation tasks.

    This class represents a dataset of Scanning Transmission Electron Microscopy (STEM) 
    images and their corresponding segmentation labels. It provides functionality for 
    loading, transforming, and accessing the image-label pairs for training segmentation 
    models.

    The dataset supports various optional transformations, including image augmentation, 
    label transformation, cropping and resizing, image generation, and normalization.

    Attributes:
        x (torch.Tensor): The STEM images.
        y (torch.Tensor): The corresponding segmentation labels.
        transform (callable, optional): Transform to be applied to the images.
        make_image (callable, optional): Function to generate images from raw data.
        label_transform (callable, optional): Transform to be applied to the labels.
        crop_resize (callable, optional): Function to crop and resize images.
        normalize (callable, optional): Function to normalize the images.
        parameters (dict, optional): Additional parameters for the dataset.

    Note:
        The dataset assumes that the images and labels are properly aligned and have 
        compatible dimensions for segmentation tasks.
    """

    def __init__(
        self,
        X,
        y,
        transform=None,
        label_transform=None,
        crop_resize=None,
        make_image=GenerateImage(),
        normalize=Normalize(),
        parameters=None,
    ):
        """
        Args:
            images (np.array): array of images
            labels (np.array): array of labels
            transform (callable, optional): Optional transform to be applied to samples
            label_transform (callable, optional): Optional transform to be applied to labels of samples
            make_image (callable, optional):
            crop_resize (callable):
            normalize (allable, optional):
        """

        self.x = torch.tensor(X).unsqueeze(1)
        self.y = torch.tensor(y)

        # make background class
        self.transform = transform
        self.make_image = make_image
        self.label_transform = label_transform
        self.crop_resize = crop_resize
        self.normalize = normalize
        self.parameters = parameters

    def __getitem__(self, index):
        label = self.y[index]
        image = self.x[index]

        if self.transform:
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed)  # apply this seed to img transforms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            image = self.transform(image)
            # reseed before crop resize
            random.seed(seed)  # apply this seed to img transforms
            torch.manual_seed(seed)  # needed for torchvision 0.7

            shape, image = self.crop_resize(image)
            random.seed(seed)  # apply this seed to target transforms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            label = self.label_transform(label)

            # reseed before crop resize
            random.seed(seed)  # apply this seed to img transforms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            shape, label = self.crop_resize(label)

        # extract sizes, then have generate_Image keep track of sample sizes??
        new_sampling = (
            np.mean(np.array(shape) / np.array(self.parameters["resize_size"]))
            * self.parameters["scanning_sampling"]
        )
        image = self.make_image(np.array(image[0]), sampling=new_sampling)
        image = self.normalize(image)
        return image, label

    def __len__(self):
        return len(self.x)


def create_static_dataset(
    parameters,
    features,
    labels,
    transform,
    label_transform,
    generate_image,
    normalize=Normalize(),
    N_augmentations=10,
    partial_threshold=0.1,
    crop_resize=None,
):
    """
    Create a static dataset with augmented samples for training or evaluation.

    This function generates a dataset of augmented images and labels based on the
    provided parameters and transformation functions. It applies data augmentation
    techniques and filters out samples that have significant edge artifacts.

    Args:
        parameters (dict): A dictionary containing configuration parameters.
        features (numpy.ndarray): The input features/images.
        labels (numpy.ndarray): The corresponding labels for the features.
        transform (callable): A function to apply transformations to the features.
        label_transform (callable): A function to apply transformations to the labels.
        generate_image (callable): A function to generate the final image from transformed features.
        normalize (callable, optional): A function to normalize the generated images. Defaults to Normalize().
        N_augmentations (int, optional): The number of augmentations to create for each sample. Defaults to 10.
        partial_threshold (float, optional): The threshold for filtering out samples with edge artifacts. Defaults to 0.1.
        crop_resize (callable, optional): A function to crop and resize the images. Defaults to None.

    Returns:
        tuple: A tuple containing two torch.Tensor objects:
            - features (torch.Tensor): The augmented and processed features.
            - labels (torch.Tensor): The corresponding augmented and processed labels.

    Note:
        This function creates a STEMDataset instance and uses it to generate
        augmented samples. It filters out samples where the maximum edge label
        value is below the partial_threshold to ensure quality of the dataset.
    """
    # Define tranformations for data augmentation
    dataset = STEMDataset(
        features,
        labels,
        transform=transform,
        label_transform=label_transform,
        make_image=generate_image,
        crop_resize=crop_resize,
        normalize=normalize,
        parameters=parameters,
    )

    # generate the validation dataset
    N = len(features)
    features = np.zeros(
        (N * N_augmentations, 1, *parameters["resize_size"]), dtype=np.float32
    )
    labels = np.zeros(
        (N * N_augmentations, labels.shape[1], *parameters["resize_size"]),
        dtype=np.float32,
    )

    total_size = N_augmentations * N
    i = 0
    dataloader = iter(dataset)
    # fill dataset to given size
    while i != total_size:
        try:
            img, label = next(dataloader)
            max_edge_label_value = np.concatenate(
                [
                    label[:-1, :, -1],
                    label[:-1, :, 0],
                    label[:-1, 0, :],
                    label[:-1, -1, :],
                ]
            ).max()  # :-1 for background
            # check if edge value is below partial threshold
            if max_edge_label_value < partial_threshold:
                labels[i, :, :, :] = label
                features[i, :, :, :] = img
                i += 1
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            dataloader = iter(dataset)

    return torch.Tensor(features), torch.Tensor(labels)
