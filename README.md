## Overview

This repository contains example implementations of machine learning nodes for Autodesk Maya. These nodes are designed for instructional purposes and are not intended for production use. The nodes included are:

- Regression Node: Performs linear regression inference based on input features.

## Files

- `regression_node.py`: Implementation of the Regression node.
- `regression_win.py`: UI for creating and managing Regression nodes in Maya.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ml-example-nodes.git
    ```
2. Copy the Python files to your Maya scripts directory, or **Add Repository to System Path** using `sys.path.append`.
    ```python
    import sys
    sys.path.append('path/to/your/repo')
    ```

## Usage

### Regression Node

1. Open Maya and load the script editor.
2. Execute the `regression_win.py` script to open the Regression UI.
3. Select the input attributes, set the output attribute, and create the Regression node.

### Steps to Use the UI

1. **Launch the UI**: Run the show() function from regression_ui.py to display the UI.
2. **Set Input Attributes**: Add input attributes to the input list widget.
3. **Set Output Attributes**: Add output attributes to the  target list widget.
4. **Get Animation Data**: If the scene has anmation, use it for train else, check the box to generate random animation curves and set the parameters (start frame, end frame, min value, max value).
5. **Train the Model**: Click the button to train the Linear Regression model using the provided data and settings.

## Dependencies

- Autodesk Maya
- NumPy
- scikit-learn
- PySide2

## Future Work

- **Add PCA Node Example**: Performs Principal Component Analysis (PCA) to reduce the dimensionality of input data.

- **Files**:
    - `pca_node.py`: Implementation of the PCA node.
    - `pca_win.py`: UI for creating and managing PCA nodes in Maya.

- **PCA Node Usage**
    - Open Maya and load the script editor.
    - Execute the `pca_win.py` script to open the PCA UI.
    - Select the input attributes, set the number of components, and create the PCA node.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

These nodes are provided as-is for educational purposes only. They are not intended for production use and come with no warranty or support.