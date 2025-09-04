Super-Resolution Microscopy Model Training
This repository contains a series of Jupyter notebooks designed to take you through the entire process of training a super-resolution microscopy model. The workflow is organized into a clear, sequential pipeline, from data simulation to final model evaluation.

Prerequisites
To get started with this project, you will need a Python environment with the following libraries: numpy, matplotlib, pandas, scikit-learn, tensorflow, and keras. These libraries are essential for data handling, visualization, and building the neural network.

A crucial external dependency is the ImageJ software with the Thunderstorm plugin. This is a critical step for converting the simulated microscopy images into single-molecule localizations, which serve as the ground truth data for the model's training. Ensure you have both ImageJ and the Thunderstorm plugin properly installed and configured before proceeding to the data generation step.

Notebooks Overview
simulator.ipynb
This notebook is the foundational first step in the pipeline. It simulates raw, low-resolution microscopy data that closely mimics the characteristics of real-world images. By generating this data with a known underlying structure, we create a perfect ground truth for our training process. The notebook allows you to define key parameters for the simulation, such as the image size, the number of frames, and the specific type of structure to be simulated (e.g., lines, circles, or more complex shapes). This control over the simulation environment ensures that the training data is both consistent and representative.

generate_training_samples.ipynb
This notebook is the critical bridge between the simulated data and the machine learning model. It uses the ImageJ Thunderstorm plugin to perform single-molecule localization on the simulated images. The Thunderstorm plugin identifies and precisely localizes individual molecules within the low-resolution frames. The output of this process is a list of precise molecule coordinates. This data is then used to generate a large number of training samples, each consisting of a low-resolution input "patch" and a corresponding high-resolution ground truth "heatmap." These patches and heatmaps are the direct inputs and outputs for the neural network, teaching the model to predict a high-resolution representation from a low-resolution input.

initial model evaluation.ipynb
This notebook is where the model training begins. It defines a convolutional neural network architecture, typically a U-Net, which is well-suited for image-to-image translation tasks like super-resolution. The notebook then compiles and trains the model on the patches and heatmaps created in the previous step. It provides a first look at the model's performance by plotting the training and validation loss over epochs. This step is crucial for ensuring the model is learning effectively, identifying potential issues like overfitting, and tuning hyperparameters before moving on to a more in-depth analysis.

complete model evaluation.ipynb
This final notebook performs a comprehensive evaluation of the fully trained model. It moves beyond simple loss metrics to evaluate the model's super-resolution performance quantitatively. The notebook calculates key metrics like Intersection over Union (IoU), which measures the pixel-level overlap between the model's predicted structure and the true ground truth. In addition, it includes visualization code to display side-by-side comparisons of the input image, the model's prediction, and the ground truth. These visual outputs are invaluable for a qualitative assessment of the model's performance and its ability to reconstruct fine details from low-resolution data.