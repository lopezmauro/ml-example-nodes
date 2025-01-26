## Overview

This repository contains example implementations of machine learning nodes for Autodesk Maya. These nodes are designed for instructional purposes and are not intended for production use. The nodes included are:

- Regression Node: Performs linear regression inference based on input features.
- PCA Blendshape Node: Performs Principal Component Analysis (PCA) to create blendshapes for mesh deformation.


## Files

- `regression_node.py`: Implementation of the Regression node.
- `regression_win.py`: UI for creating and managing Regression nodes in Maya.
- `pca_blendshape.py`: Implementation of the PCA blendshape node.
- `pca_win.py`: UI for creating and managing PCA blendshape nodes in Maya.


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
2. To open the Regression UI execute:
```python
import nodes_ui
nodes_ui.regrssion_ui()
```
3. Select the input attributes, set the output attribute, and create the Regression node.

#### Reegression UI

1. **Launch the UI**: Run the show() function from regression_ui.py to display the UI.
2. **Set Input Attributes**: Add input attributes to the input list widget.
3. **Set Output Attributes**: Add output attributes to the  target list widget.
4. **Get Animation Data**: If the scene has anmation, use it for train else, check the box to generate random animation curves and set the parameters (start frame, end frame, min value, max value).
5. **Train the Model**: Click the button to train the Linear Regression model using the provided data and settings.

### PCA Blendshape Node

1. Open Maya and load the script editor.
2. To open the PCA Blendshape UI execute:
```python
import nodes_ui
nodes_ui.pca_ui()
```
3. Select the source blendshape node and target mesh.
4. Click the "Create PCA Blendshape" button to create the PCA blendshape node.

#### PCA Blendshape Features
The PCA blendshape node converts the selected blendshape to the PCA space. By default, it does not compress the data. Instead, it allows the user to choose the desired level of compression. This means the user can decide how much variation to retain, balancing between data size reduction and the amount of variation preserved.
- **Compression Ratio**: The desired variance to keep for compression.
- **Compression Result**: The actual compression percentage achieved.


## Dependencies

- Autodesk Maya
- NumPy
- scikit-learn
- PySide2

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

These nodes are provided as-is for educational purposes only. They are not intended for production use and come with no warranty or support.