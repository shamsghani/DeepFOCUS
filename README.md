# Super-Resolution Microscopy Model Training

This repository contains a series of Jupyter notebooks designed to guide you through the entire process of training a super-resolution microscopy model. The workflow is organized into a clear, sequential pipeline, from data simulation to final model evaluation.

---

## Prerequisites

To get started with this project, you will need a Python environment with the following libraries:

- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `keras`

These libraries are essential for data handling, visualization, and building the neural network.

A crucial external dependency is the **ImageJ software with the ThunderSTORM plugin**. This is required to convert simulated microscopy images into single-molecule localizations, which serve as the ground truth data for model training. Make sure both ImageJ and the ThunderSTORM plugin are properly installed before proceeding to the data generation step.

---

## Notebooks Overview

### `simulator.ipynb`
This is the foundational first step in the pipeline. It simulates raw, low-resolution microscopy data that closely mimics real-world images. By generating data with a known underlying structure, it creates a perfect ground truth for training.

Key features:

- Define image size, number of frames, and structure type (lines, circles, or complex shapes)
- Control over simulation parameters ensures consistent and representative training data

---

### `generate_training_samples.ipynb`
This notebook bridges the simulated data and the machine learning model. It uses the **ImageJ ThunderSTORM plugin** to perform single-molecule localization on the simulated images.

Outputs:

- Precise molecule coordinates from low-resolution frames
- Large training dataset of input "patches" and high-resolution "heatmaps"  
- These patches and heatmaps are the direct inputs and outputs for the neural network, teaching the model to predict high-resolution representations from low-resolution inputs

---

### `initial_model_evaluation.ipynb`
This notebook begins the model training process. It defines a convolutional neural network architecture, typically a **U-Net**, suitable for image-to-image translation tasks like super-resolution.

Features:

- Compiles and trains the model on the patches and heatmaps
- Plots training and validation loss over epochs
- Helps identify issues like overfitting and tune hyperparameters

---

### `complete_model_evaluation.ipynb`
This final notebook performs a comprehensive evaluation of the fully trained model.

Features:

- Quantitative evaluation of super-resolution performance
- Calculates metrics like **Intersection over Union (IoU)** for pixel-level overlap
- Visualizations to compare input images, model predictions, and ground truth
- Enables qualitative assessment of the model’s ability to reconstruct fine details

---

## Usage

1. Ensure all prerequisites are installed.
2. Run `simulator.ipynb` to generate synthetic microscopy data.
3. Process the data with `generate_training_samples.ipynb`.
4. Train the model using `initial_model_evaluation.ipynb`.
5. Evaluate the trained model with `complete_model_evaluation.ipynb`.

