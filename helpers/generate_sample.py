from scipy.cluster.hierarchy import fcluster, linkage
import ase
import numpy as np
import warnings


# cut random specimen
def cut_surface(atoms, fraction=0.3):
    """
    Cut a random specimen from the given atomic structure.

    This function removes atoms from the structure based on random planar cuts,
    simulating a rough surface or an irregularly shaped specimen.

    Args:
        atoms (ase.Atoms): The atomic structure to cut.
        fraction (float, optional): The fraction of the structure to remove. 
                                    Should be between 0 and 1. Defaults to 0.3.

    Returns:
        ase.Atoms: A new Atoms object with a subset of the original atoms, 
                   representing the cut specimen.

    Note:
        The function applies cuts from multiple directions, creating an irregular
        surface. The exact shape of the cut is randomized, but on average removes
        approximately the specified fraction of atoms.
    """
    directions = [
        np.array([i, j, 0], dtype=float)
        for i in range(-1, 2)
        for j in range(-1, 2)
        if not (i == 0 and j == 0)
    ]
    size = atoms.get_cell().lengths()[:2].mean() / 2
    pos = atoms.get_positions() - atoms.get_cell().lengths() / 2
    keep = np.ones(len(pos), bool)
    for direction in directions:
        cdist = size * ((1 - fraction) + fraction * np.random.random())
        direction += np.random.standard_normal(3) * 0.05  # make vectors slightly off
        uv = direction / np.linalg.norm(direction)
        cut = np.dot(pos, uv) < cdist

        keep = np.logical_and(cut, keep)

    return atoms[keep]


# generate bulk
def create_bulk(atoms, cell, origin=[0, 0, 0]):
    """
    Create an ideal bulk lattice for a specified volume.

    This function generates a bulk crystal structure by replicating a given unit cell
    to fill a specified volume. It calculates the optimal number of lattice replications
    and returns a full lattice within the specified bounding box.

    Args:
        atoms (ase.Atoms): The unit cell to create the sample from. The unit cell should be standardized.
        cell (list of float): The dimensions of the cell to produce in Angstroms [x, y, z].
                              For example, [20, 20, 20].
        origin (list of float, optional): The origin of the new cell in Angstroms [x, y, z].
                                          Defaults to [0, 0, 0].

    Returns:
        ase.Atoms: The bulk sample with the specified dimensions.

    Note:
        The function ensures that the returned structure completely fills the specified
        volume while maintaining the crystal structure of the input unit cell.
    """
    # translate origin
    atoms = atoms.copy()
    atoms.translate(-np.array(origin))
    atoms.wrap()

    lattice_vectors = np.array(atoms.get_cell())
    points = np.array(
        [
            [0, 0, 0],
            [0, cell[1], 0],
            [0, cell[1], cell[2]],
            [0, 0, cell[2]],
            [cell[0], 0, 0],
            [cell[0], cell[1], 0],
            [cell[0], 0, cell[2]],
            [cell[0], cell[1], cell[2]],
        ]
    )
    projections = np.linalg.solve(lattice_vectors.T, np.array(points).T)
    # find the minimum and maximum repetitions for each of the unit cell vectors
    repetitions = np.array(np.ceil(np.max(projections, axis=1)), dtype=int)
    # negative repetitions which must be actualized by replicating positvely and then shifting all positions
    neg_repetitions = np.array(np.floor(np.min(projections, axis=1)), dtype=int)

    sample = atoms * (repetitions - neg_repetitions)

    # translate according to negative repetitions
    sample.translate(lattice_vectors.T @ neg_repetitions)

    positions = sample.get_positions()
    mask = (
        (positions[:, 0] < cell[0])
        & (positions[:, 1] < cell[1])
        & (positions[:, 2] < cell[2])
    )
    mask2 = (positions[:, 0] > 0) & (positions[:, 1] > 0) & (positions[:, 2] > 0)
    del sample[~(mask & mask2)]

    sample.set_cell(np.diag(cell))
    return sample


# generate slab of 2D layers
def create_slab(atoms, cell, layers, ucell_layers=1, origin=[0, 0, 0]):
    """
    Create a slab of 2D layers from a given atomic structure.

    This function generates a slab of 2D layers by replicating the input atomic structure
    in the x-y plane and stacking it vertically to create the specified number of layers.

    Parameters
    ----------
    atoms : ase.Atoms
        The unit cell to create the sample from. The unit cell should be standardized.
    cell : list of 2 floats
        The dimensions of the cell to produce in the x-y plane [Å]. For example [20, 20].
    layers : int
        The total number of layers to create in the slab.
    ucell_layers : int, optional
        The number of layers in the input unit cell. Used to calibrate the total height
        of the generated slab. Defaults to 1.
    origin : list of 3 floats, optional
        The origin of the new cell in [Å]. Defaults to [0, 0, 0].

    Returns
    -------
    ase.Atoms
        The generated slab sample with the specified dimensions and number of layers.

    Note
    ----
    The function calculates the appropriate height for the slab based on the input
    unit cell and the desired number of layers. It then uses the create_bulk function
    to generate the final structure.
    """
    # sum abs valus instead
    height = np.sum(np.abs(atoms.get_cell()[:, 2]))
    cell_height = height * layers / ucell_layers
    return create_bulk(atoms, [*cell, cell_height], origin=origin)


def displace_columns(sample, sigma=0.15, cluster_distance=1):
    """
    Displace atomic columns in the sample based on a Gaussian distribution.

    This function applies random displacements to atomic columns in the sample.
    It clusters atoms into columns based on their x-y positions and then applies
    a consistent displacement to all atoms within each column.

    Args:
        sample (ase.Atoms): The atomic structure to modify.
        sigma (float, optional): Standard deviation of the Gaussian distribution
                                 used for displacements. Defaults to 0.15.
        cluster_distance (float, optional): Distance threshold for clustering
                                            atoms into columns. Defaults to 1.

    Returns:
        ase.Atoms: A new Atoms object with displaced atomic columns.

    Note:
        - The function uses hierarchical clustering to group atoms into columns.
        - Displacements are applied in all three dimensions (x, y, z).
        - The original sample is not modified; a copy is returned instead.
    """
    sample = sample.copy()
    pos = sample.get_positions()[:, :2]
    c = fcluster(linkage(pos), t=cluster_distance, criterion="distance")
    displacements = np.random.normal(size=[np.max(c) + 1, 3], scale=sigma)
    displacements = displacements[c]  # gives list of the displacement for each position
    sample.positions += displacements
    return sample


def decode_defect_name(defect_name):
    """
    Decode a defect name string into a list of individual defect components.

    This function takes a defect name string and decodes it into a list of individual
    defect components. It handles both single defects and multiple defects specified
    with a numeric prefix.

    Args:
        defect_name (str): A string representing the defect name. It can include
                           multiple defects separated by '&' and numeric prefixes
                           for repeated defects.

    Returns:
        numpy.ndarray: An array of individual defect components.

    Examples:
        >>> decode_defect_name("2Cr")
        array(['Cr', 'Cr'], dtype='<U2')
        >>> decode_defect_name("V-Br&V-S")
        array(['V-Br', 'V-S'], dtype='<U4')
        >>> decode_defect_name("3V-Cr&S")
        array(['V-Cr', 'V-Cr', 'V-Cr', 'S'], dtype='<U4')
    """
    decoded = []
    for d in defect_name.split("&"):
        if d[0].isdigit():
            decoded.append([d[1:]] * int(d[0]))
        else:
            decoded.append([d])
    return np.concatenate(decoded)


def generate_defects_by_column(
    sample, defects, cluster_distance=1, z_distance=1.5, delete_unknown_cols=True
):
    """
    Creates defects in the given material sample based on specified defect configurations.

    This function introduces defects into atomic columns of the sample according to
    the probabilities and types specified in the 'defects' dictionary.

    Args:
        sample (ase.Atoms): The atomic structure to modify.
        defects (dict): A nested dictionary specifying defect configurations.
            The outer keys represent column compositions (e.g., "2Cr", "2Br&2S").
            The inner keys are defect types with their corresponding probabilities.
        cluster_distance (float, optional): Distance for clustering atoms into columns. Defaults to 1.
        z_distance (float, optional): Vertical distance threshold for defect application. Defaults to 1.5.
        delete_unknown_cols (bool, optional): Whether to delete columns not specified in defects. Defaults to True.

    Returns:
        ase.Atoms: Modified atomic structure with introduced defects.
        dict: Locations of introduced defects.

    Example:
        defects = {
            "2Cr": {
                "V-Cr": 0.2,   # 20% chance of single Cr vacancy
                "2V-Cr": 0.1,  # 10% chance of double Cr vacancy
            },
            "2Br&2S": {
                "2V-Br": 0.1,     # 10% chance of double Br vacancy
                "V-Br": 0.15,     # 15% chance of single Br vacancy
                "V-S": 0.05,      # 5% chance of S vacancy
                "V-Br&V-S": 0.015 # 1.5% chance of both Br and S vacancies
            }
        }
        sample = ase.Atoms('Cr2Br2S2', positions=[[0,0,0], [0,0,2], [1,1,1], [1,1,3], [2,2,0], [2,2,2]])
        modified_sample, defect_locs = generate_defects_by_column(sample, defects)

    Note:
        The function processes each atomic column in the sample and applies defects
        based on the probabilities specified in the 'defects' dictionary. It handles
        complex defect configurations and supports multiple defect types per column.
    """
    atoms = sample.copy()
    to_delete = []
    defect_locations = {}
    ## deal with defects dictionary
    defect_probabilities = {}
    defect_types = {}
    defect_names = {}
    for column in defects:
        column_type = decode_defect_name(column)
        unique, counts = np.unique(column_type, return_counts=True)
        # new defect key
        column_type = str(dict(zip(unique, counts)))
        defect_probabilities[column_type] = []
        defect_types[column_type] = []
        defect_names[column_type] = []
        # construct the defect probabilites and changes for the column
        for defect_type in defects[column]:
            defect_probabilities[column_type].append(defects[column][defect_type])
            defect_types[column_type].append(list(decode_defect_name(defect_type)))
            defect_names[column_type].append(defect_type)
            defect_locations[defect_type] = []
        # do cumsum and confirm lower than 1!
        defect_probabilities[column_type] = np.cumsum(defect_probabilities[column_type])
        if (defect_probabilities[column_type] > 1).any():
            raise ValueError(f"Probabilities exceed 1 for {column}")

    ## find columns
    positions = atoms.get_positions()[:, :2]
    zmax = np.max(atoms.get_positions()[:, 2])
    symbols = np.array(atoms.get_chemical_symbols())
    clusters = fcluster(linkage(positions), t=cluster_distance, criterion="distance")
    # loop over columns and introduce defects
    for i in range(np.max(clusters)):
        atoms_in_column = clusters == i + 1
        col_positions = positions[atoms_in_column]
        col_atoms = symbols[atoms_in_column]
        unique, counts = np.unique(col_atoms, return_counts=True)
        # sorted counts as key
        column_type = str(dict(zip(unique, counts)))
        # get entry from defect dictionary
        if column_type in defect_probabilities:
            column_defects = defect_probabilities[column_type]
            chosen_i = np.sum(column_defects < np.random.random(), axis=0)
            # no change
            if chosen_i == len(column_defects):
                continue
            chosen_defects = defect_types[column_type][chosen_i]
            defect_name = defect_names[column_type][chosen_i]
            # create dict of species and indices at this site
            column_atoms = {
                site: atoms_in_column.nonzero()[0][(col_atoms == site)]
                for site in unique
            }
            for current_defect in chosen_defects:
                occupant, site = current_defect.split("-")
                # deal with adatoms
                if "add" in site:
                    atoms = atoms + ase.Atom(
                        occupant,
                        np.concatenate(
                            [np.mean(col_positions, axis=0), [zmax + z_distance]]
                        ),
                    )
                else:
                    active_site = np.random.choice(column_atoms[site], replace=False)
                    # remove the chosen index from the original array too. Todo: improve this
                    column_atoms[site] = column_atoms[site][
                        ~(column_atoms[site] == active_site)
                    ]
                    if occupant == "V":
                        to_delete.append(active_site)
                    else:
                        atoms.symbols[active_site] = occupant
            # add defect to defect_locations
            defect_locations[defect_name].append(
                np.mean(positions[atoms_in_column], axis=0)
            )
        else:
            if delete_unknown_cols:
                for atom in atoms_in_column.nonzero()[0]:
                    to_delete.append(atom)
            else:
                warnings.warn(
                    f"Column, {column_type}, found which was not in the defect dictionary"
                )
    # delete atoms marked for deletion
    del atoms[to_delete]
    atoms.center(vacuum=2, axis=2)
    return atoms, defect_locations


def generate_sample(atoms, size=(14, 14), pars=False, layers=[1, 4], ucell_layers=1):
    """
    Generate a sample with randomized parameters.

    Args:
        atoms (ase.Atoms): The input atomic structure.
        size (tuple): The size of the sample in Angstroms. Default is (14, 14).
        pars (bool): If True, return the randomized parameters. Default is False.
        layers (list): Range of layers to choose from [min, max]. Default is [1, 4].
        ucell_layers (int): Number of unit cell layers. Default is 1.

    Returns:
        ase.Atoms: The generated sample.
        list: [layers, theta, phi] if pars is True.

    Note:
        - The number of layers is randomly chosen between the given range.
        - Theta (rotation around z-axis) is randomly chosen between 0 and 360 degrees.
        - Phi (tilt angle) is randomly chosen using a normal distribution, capped at 1.6 degrees
          to prevent column overlap.
    """
    # Parameters to randomize
    layers = int(np.random.random() * (layers[1] - layers[0] + 1)) + layers[0]
    theta = np.random.random() * 360
    phi = np.min([np.abs(np.random.normal()), 1.6])
    # Note: maximum 1.6 after that columns overlap too much

    atoms = atoms.copy()
    # rotate around z-axis
    atoms.rotate(theta, "z", rotate_cell=True)
    sample = create_slab(atoms, size, layers, ucell_layers)
    # cut surface
    sample = cut_surface(sample)

    # rotate away from zone-axis
    omega = np.random.random() * 360
    sample.rotate(omega, "z", center="COU")
    sample.rotate(phi, "x", center="COU")
    sample.rotate(-omega, "z", center="COU")
    # center along z-axis and add 2 Å of vacuum to top and bottom
    sample.center(vacuum=2, axis=2)
    if pars:
        return sample, [layers, theta, phi]
    return sample
