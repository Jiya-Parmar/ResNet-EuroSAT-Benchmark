# ResNet-EuroSAT-Benchmark: Optimized Satellite Image Classification

A comprehensive deep learning project implementing a **ResNet50** Convolutional Neural Network (CNN) to classify high-resolution images from the **EuroSAT dataset** into 10 distinct land-use/land-cover categories.

This project focuses on **optimization and regularization** to achieve a robust, high-performing model that effectively minimizes overfitting.

## 🌟 Project Overview

| Feature | Details |
| :--- | :--- |
| **Goal** | Establish a highly optimized and reproducible classification benchmark for the EuroSAT dataset. |
| **Model** | Custom-built **ResNet50** architecture (implemented from scratch using Keras Functional API). |
| **Dataset** | EuroSAT (27,000 satellite images across 10 classes). |
| **Key Techniques** | Data Augmentation, L2 Regularization, Dropout, Dynamic Learning Rate Scheduling. |

## 🚀 Setup and Execution

### Prerequisites

* Google Colab (Recommended) or a local Python environment.
* GPU access (Recommended for fast training).
* Required libraries (installed via the notebook): `tensorflow`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

### 1. Repository Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jiya-Parmar/ResNet-EuroSAT-Benchmark.git](https://github.com/Jiya-Parmar/ResNet-EuroSAT-Benchmark.git)
    cd ResNet-EuroSAT-Benchmark
    ```
2.  **Download the Dataset:**
    The project relies on the **EuroSAT dataset**. **The ZIP file (`EuroSAT.zip`) is assumed to be available** for the Colab environment. In the Colab notebook, the code expects the zip file at the path:
    ```python
    zip_path = '/content/EuroSAT.zip'
    ```
    *If running locally, update the `zip_path` variable in the notebook to your local file location.*

### 2. Running the Code (Colab)

1.  Open the file **`Untitled5.ipynb`** in Google Colab.
2.  Go to **Runtime** -> **Run all** (or execute cells sequentially).
3.  The notebook will automatically:
    * Mount Google Drive (to access the zip file).
    * Unzip the dataset into `/content/EuroSAT_unzipped/2750`.
    * Load and prepare the training and validation datasets with augmentation.
    * Define and compile the optimized ResNet50 model.
    * Train the model for up to 50 epochs (with Early Stopping).
    * Generate and display performance plots and the Confusion Matrix.

## 🛠 Model Architecture and Optimization

The following techniques were critical in moving past the initial overfitting stage to achieve robust performance:

### 1. ResNet Blocks with L2 Regularization

The `identity_block` and `convolutional_block` functions incorporate **L2 weight regularization** (`l2(1e-4)`) on all `Conv2D` layers to penalize large weights and prevent model complexity from scaling unchecked.

### 2. Data Augmentation

A robust sequential data augmentation pipeline is applied **only to the training dataset** using the `tf.keras.Sequential` layer:
* `RandomFlip("horizontal")`
* `RandomRotation(0.05)`
* `RandomZoom(0.1)`
* `RandomContrast(0.1)`

### 3. Classifier Head Regularization

A custom classifier head was added after the final `AveragePooling2D` layer to enhance regularization:
* `Dropout(0.5)`
* A hidden `Dense(256)` layer with ReLU activation and `l2(1e-4)` regularization.
* A second `Dropout(0.5)`.

### 4. Dynamic Training Callbacks

The training process uses a dynamic approach rather than a fixed decay schedule:

| Callback | Setting | Purpose |
| :--- | :--- | :--- |
| **`ReduceLROnPlateau`** | `patience=3`, `factor=0.5` | Reduces the learning rate by 50% if the validation loss doesn't improve for 3 epochs. |
| **`EarlyStopping`** | `patience=10`, `monitor='val_loss'` | Stops training after 10 epochs without improvement in validation loss, restoring the best weights. |

## 📊 Results and Analysis

After training, the notebook generates the following visualizations:

1.  **Accuracy/Loss Plots:** Shows the training and validation curves over epochs, demonstrating the effectiveness of the regularization in closing the generalization gap.
2.  **Confusion Matrix:** Provides a detailed, class-by-class visualization of the model's predictive performance.

### Example Performance (Visualized)

The final plots show the training history and the detailed performance breakdown.

*(The actual plots will appear after you run the notebook and generate the figures.)*
