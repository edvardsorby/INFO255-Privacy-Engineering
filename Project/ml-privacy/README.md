# Explore and Visualize Opacus Experiments

This repository contains two main notebooks that demonstrate the use of [Opacus](https://opacus.ai/) in training different deep learning models with differential privacy and visualizing the results.

## Notebooks

### explore_opacus.ipynb
- **Purpose:**  
  This notebook focuses on the exploration of training different models with privacy using Opacus. It includes:
  - Loading and pre-processing the MNIST dataset.
  - Setting up a simple convolutional neural network.
  - Configuring the PrivacyEngine from Opacus.
  - Running experiments with various noise multipliers and clipping values.
  - Measuring and logging training losses, privacy budgets (ε), and evaluation metrics (accuracy, precision, recall, F1-score).

### visualization_opacus.ipynb
- **Purpose:**  
  This notebook is dedicated to visualizing experiment results from the Opacus runs. It provides:
  - Plots comparing F1-scores, epsilon values, and other metrics for different models.
  - Aggregated barplots and line charts showing how experiment parameters affect model performance and privacy.

## Setup and Requirements

A `requirements.txt` file is included and contains the following main dependencies:

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
opacus>=1.4.0
seaborn>=0.11.2
tqdm>=4.62.0
scikit-learn>=1.0.2
```

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Running the Notebooks

1. **Explore Opacus Notebook:**  
   Open `explore_opacus.ipynb` in a notebook environment. Run the cells sequentially to observe the training and privacy measurement process.

2. **Visualization Opacus Notebook:**  
   Open `visualization_opacus.ipynb` to visualize the collected experiment results. This notebook includes several plotting functions that output graphs comparing the model performance metrics and privacy budget across different runs.

## Additional Information

- **Differential Privacy with Opacus:**  
  The experiments in this repository aim to demonstrate how varying the noise multiplier and gradient clipping norm impacts both model performance (via standard metrics) and the privacy budget (epsilon, δ).

---

This repository serves as a starting point for evaluating and visualizing differential privacy in deep learning using Opacus. Experiment with the provided notebooks to better understand the trade-offs between model accuracy and rigorous privacy guarantees.