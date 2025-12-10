# Social media post success predictor

## Overview

This project implements a Deep Learning model using **TensorFlow/Keras** to predict the success of social media posts. Success is defined as posts achieving **above-median engagement (likes) AND high conversion rates** (a key business metric).

The model uses a **Dual-Input Architecture** to process both **numerical metadata** (follower count, time of day, likes, shares, conversion rate) and **post text** via a dedicated **Embedding Layer**.
I
## Optimization and Performance

The final model architecture was selected through **Hyperparameter Tuning** using the **Keras Tuner** library, which systematically searched for the best combination of layer sizes and dropout rates.

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Final Test Accuracy** | **98.00%** | Achieved on synthetic validation data. |
| **Tuning Method** | `RandomSearch` | Optimized for `val_accuracy`. |

## Prerequisites

Before running the model, you must have **Python 3.8+** and the dependencies listed in `requirements.txt` installed.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/CodeWithIva/social-media-post-success-predictor]

cd social-media-post-success-predictor

### 2. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment (venv).

### 3. Install Dependencies

Install all required libraries, including Keras Tuner, using the provided file:

```bash
pip install -r requirements.txt

### 4. Running the Project

Run the main Python script from the root of your project directory:

```bash
python social_media.py

Expected Output:

Terminal Output: Prints the Hyperparameter Search process, the final training progress, and the Final Test Accuracy (98.00%).
Artifacts: Creates a saved model file: best_social_media_model.h5.
Graphs: Two windows will display: Training/Validation Curves and the Confusion Matrix.

### 5. Key Model Architecture

The model uses the Keras functional API to handle two distinct inputs:

Text Path: Input Sequences → Embedding Layer → Flatten → Dense.
Numerical Path: Scaled Features → Dense.
Combination: Outputs are joined using the Concatenate layer before feeding into the classification layers (Dense → Dropout → Sigmoid Output).