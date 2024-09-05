# Helpers
---

This folder contains function definitions for various parts of the project. Below is a list of the different files along with their primary purpose. Most functions within these files have documentation strings that are useful for understanding their usage.

|File|Purpose|
|:--|:--|
|`fitting.py`| Defines functions for fitting images, altering images, and performing operations on peaks and lattices extracted experimentally. |
|`generate_sample.py`| Contains functions for generating samples using `ASE` (Atomic Simulation Environment). |
|`helpers.py`| Provides simple helper functions that are imported and used by other files in the project. |
|`image_simulation.py`| Includes function wrappers for image simulation using `abTEM` (ab initio Transmission Electron Microscopy). |
|`models.py`| Defines functions for creating and managing machine learning models used in the project. |
|`labels.py`| Contains functions for generating labeled samples from atomic models. |
|`preprocessing.py`| Implements data augmentation functions compatible with `PyTorch`. |
|`processing.py`| Provides functions for processing experimental images. |
|`visualization.py`| (If applicable) Contains functions for visualizing data, results, and model outputs. |
|`evaluation.py`| (If applicable) Includes functions for evaluating model performance and analyzing results. |

Each file is designed to handle specific aspects of the deep learning workflow for 2D defect detection in multi-layer material STEM-HAADF images. Users are encouraged to refer to the individual files for more detailed information on the functions they contain.