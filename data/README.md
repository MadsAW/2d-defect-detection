# Data

This folder contains the data necessary to reproduce the results of the manuscript. Below is an overview of the contents in the subfolders and their purposes.

| Subfolder | Description |
|-----------|-------------|
| `crystals/` | Crystal structure files for the materials used in the study |
| `datasets/` | Synthetic and processed experimental datasets used for training and testing, notice the files here are large and as such you should generate your own |
| `experimental/` | Raw experimental STEM-HAADF images of 2L CrSBr |
| `model_weights/` | Saved weights of trained models at different stages |

## Usage

To use this data:

1. Ensure you have cloned the entire repository.
2. The notebooks in the root directory (`01_dataset_generation.ipynb`, `02_training_model.ipynb`, and `03_application.ipynb`) are set up to access these data files directly.
3. If you're using the data for your own scripts, make sure to set the correct path to this data folder.

Note: The actual contents of each subfolder may vary. This README provides an overview of the expected structure and content. Please refer to the specific subfolders for the most up-to-date information on available data.



