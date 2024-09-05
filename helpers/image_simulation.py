import numpy as np
import abtem

from .helpers import draw_uniform


# Note the arguments are generally inspired by Madsen ...
def generate_STEM_image(
    atoms, parameters, pars=False, gpu=False, pbar=False, N_frozen=1, sigmas=0.1
):
    """
    Generate a simulated STEM (Scanning Transmission Electron Microscopy) image.

    This function simulates a STEM image based on the provided atomic structure and 
    microscope parameters. It supports various features including frozen phonons, 
    aberration coefficients, and detector configurations.

    Args:
        atoms (ase.Atoms): Atomic structure to be imaged.
        parameters (dict): Dictionary containing simulation parameters including:
            - 'energy': Electron beam energy in keV.
            - 'semiangle_cutoff': Range for probe-forming aperture semi-angle in mrad.
            - 'defocus': Range for defocus in Angstroms.
            - 'Cs': Range for spherical aberration in mm.
            - 'C12': Range for 2-fold astigmatism in Angstroms.
            - 'B2': Range for coma in Angstroms.
            - 'AD': Tuple of (inner, outer) angles for annular detector in mrad.
            - 'gpts': Number of grid points for potential calculation.
            - 'interpolation': Interpolation method for S-matrix calculation.
            - 'scanning_sampling': Sampling rate for the scan in Angstroms.
        pars (bool, optional): If True, return simulation parameters. Defaults to False.
        gpu (bool, optional): If True, use GPU for calculations. Defaults to False.
        pbar (bool, optional): If True, show progress bar. Defaults to False.
        N_frozen (int, optional): Number of frozen phonon configurations. Defaults to 1.
        sigmas (float or array-like, optional): Thermal vibration amplitudes for frozen phonons. Defaults to 0.1.

    Returns:
        numpy.ndarray: Simulated STEM image.
        dict (optional): Dictionary of simulation parameters if pars is True.

    Note:
        This function uses the abTEM package for STEM image simulation. It randomly 
        selects values for various aberration coefficients within the specified ranges.
    """

    if gpu:
        device = "gpu"
    else:
        device = "cpu"
    # Get size of the unit cell for scan
    size = atoms.get_cell().lengths()[:2]
    # create frozen phonons if possible
    if N_frozen != 1:
        sample = abtem.FrozenPhonons(atoms, num_configs=N_frozen, sigmas=sigmas)
    else:
        sample = atoms
    energy = parameters["energy"]
    ## Paramters to randomize
    semiangle_cutoff = draw_uniform(
        parameters["semiangle_cutoff"]
    )  # defines probe size
    # describes the maximum scattering angle used in the S-matrix (should be slightly larger than semiangle cutoff)
    defocus = draw_uniform(parameters["defocus"])
    Cs = draw_uniform(parameters["Cs"])
    # Astigmatism
    C12 = draw_uniform(parameters["C12"])
    C12_phi = 2 * np.pi * np.random.random()
    # Coma
    B2 = draw_uniform(parameters["B2"])
    C21 = 3 * B2
    C21_phi = 2 * np.pi * np.random.random()
    # HAADF detector
    AD_inner = parameters["AD"][0]
    AD_outer = parameters["AD"][1]

    ctf_parameters = {
        "C12": C12,
        "phi12": C12_phi,
        "C21": C21,
        "phi21": C21_phi,
        "Cs": Cs,
        "defocus": defocus,
    }
    ctf = abtem.CTF(energy=energy, aberration_coefficients=ctf_parameters)

    # construct potential
    potential = abtem.Potential(sample, gpts=parameters["gpts"])  # slice_thickness

    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        store_on_host=True,
        device=device,
        interpolation=parameters["interpolation"],
    )

    detector = abtem.AnnularDetector(inner=AD_inner, outer=AD_outer)
    gridscan = abtem.GridScan(
        start=[0, 0], end=[size[0], size[1]], sampling=parameters["scanning_sampling"]
    )

    measurement = s_matrix.scan(scan=gridscan, detectors=detector, ctf=ctf).compute(
        progress_bar=pbar
    )
    if pars:
        return measurement, ctf_parameters
    return measurement
