\# 🚗 ComfyUI AD Analytics



A specialized tool for **Autonomous Driving (AD)** Data curation. This node allows engineers to visualize the latent distribution of their driving datasets (NuScenes, Waymo, Synthetic) directly within ComfyUI.



\## Features

**AD Data Collector:** Saves Image + Embedding pairs (.pkl) systematically.

**AD Latent Explorer:** visualizes dataset distribution using UMAP to find edge cases and bias.



\## Installation

1\. Clone this repo into `ComfyUI/custom\_nodes`:

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/michaelansahgit/ComfyUI-AD-Analytics.git](https://github.com/michaelansahgit/ComfyUI-AD-Analytics.git)

&nbsp;   ```

2\. Install dependencies:

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



\## Usage

1\. **Collect Data:** Connect your VAE/CLIP output to the `AD Data Collector` node.

2\. **Visualize:** Run the `AD Latent Explorer` pointing to the same folder.

