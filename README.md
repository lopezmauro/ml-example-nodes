## Overview

This repository contains example implementations of machine learning nodes for Autodesk Maya. These nodes are designed for instructional purposes and are not intended for production use. The nodes included are:

- PCA Node: Performs Principal Component Analysis (PCA) to reduce the dimensionality of input data.
- Regression Node: Performs linear regression inference based on input features.

## Files

- `pca_node.py`: Implementation of the PCA node.
- `regression_node.py`: Implementation of the Regression node.
- `pca_win.py`: UI for creating and managing PCA nodes in Maya.
- `regression_win.py`: UI for creating and managing Regression nodes in Maya.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ml-example-nodes.git
    ```
2. Copy the Python files to your Maya scripts directory.

## Usage

### PCA Node

1. Open Maya and load the script editor.
2. Execute the `pca_win.py` script to open the PCA UI.
3. Select the input attributes, set the number of components, and create the PCA node.

### Regression Node

1. Open Maya and load the script editor.
2. Execute the `regression_win.py` script to open the Regression UI.
3. Select the input attributes, set the output attribute, and create the Regression node.

## Dependencies

- Autodesk Maya
- NumPy
- scikit-learn
- PySide2

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

These nodes are provided as-is for educational purposes only. They are not intended for production use and come with no warranty or support.