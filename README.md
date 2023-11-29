# Image Comparison Tool

## Overview

This project is designed to compare a set of generated images with a set of original images. 

It calculates the Euclidean distance between feature vectors of each pair of images (one from the generated set and one from the original set) and logs these distances to a file. 

This can be particularly useful for evaluating the performance of image-generating models, such as GANs (Generative Adversarial Networks), by measuring how similar the generated images are to a set of original images.

## Features

- **Feature Extraction:** Uses Histogram of Oriented Gradients (HOG) to extract feature vectors from images.

- **Distance Calculation:** Computes the Euclidean distance between each pair of feature vectors from the generated and original images.

- **Logging:** Outputs the distances to a log file, listing the distances of each generated image to every image in the original dataset.

## Requirements

- Python 3
- Libraries: `cv2` (OpenCV), `numpy`, `skimage`, `sklearn`

Make sure to install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Set up the directories and log file path in the config module under `src/settings/config.py`.

2. Run the script:

```bash
python run.py
```

3. Check the output in the specified log file.

## Configuration

Before running the script, ensure that the paths to the dataset directory, generated images directory, and the log file are correctly set in the `src/settings/config.py` module.
