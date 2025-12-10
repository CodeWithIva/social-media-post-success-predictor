# Social Media Post Success Predictor

## Overview

This project implements a Deep Learning model using **TensorFlow/Keras** to predict the success of social media posts.

**Success** is defined as posts achieving:

* **Above-median engagement (likes)**
* **High conversion rates** (key business metric)

The model uses a **Dual-Input Architecture** to process:

1. **Numerical metadata**
   (follower count, time of day, likes, shares, conversion rate)
2. **Post text**
   (via Keras **Embedding Layer**)

---

## Optimization and Performance

Hyperparameters were optimized using **Keras Tuner**, which searched for the best layer sizes and dropout rates.

| Metric                  | Result         | Note                                  |
| ----------------------- | -------------- | ------------------------------------- |
| **Final Test Accuracy** | **98.00%**     | Achieved on synthetic validation data |
| **Tuning Method**       | `RandomSearch` | Optimized for `val_accuracy`          |

---

## Prerequisites

You will need:

* **Python 3.8+**
* Packages listed in `requirements.txt`

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <[https://github.com/CodeWithIva/social-media-post-success-predictor]>

cd social-media-post-success-predictor
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment (`venv`).

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the main script:

```bash
python social_media.py
```

### Expected Output

* **Terminal:**
  Hyperparameter search progress, training logs, and final test accuracy (**98%**)

* **Artifacts:**
  `best_social_media_model.h5` (saved model)

* **Graphs:**

  * Training & validation accuracy/loss curves
  * Confusion matrix

---

## Key Model Architecture

The model uses the **Keras Functional API** with two input paths:

### **Text Input Path**

```
Input → Embedding → Flatten → Dense
```

### **Numerical Input Path**

```
Input → Scaling → Dense
```

### **Combining Paths**

Outputs from both branches are concatenated and passed through:

```
Dense → Dropout → Sigmoid Output
```
