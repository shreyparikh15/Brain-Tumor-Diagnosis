# Brain Tumor MRI Classification

This project provides a web app that classifies brain tumor types from MRI images using a deep learning model.

The app is built with **Streamlit** and uses a pre-trained Keras model (`model.h5`).

---

## Features

* Upload brain MRI images (JPG, JPEG, PNG formats)
* Predict tumor type:

  * Glioma Tumor
  * Meningioma Tumor
  * No Tumor
  * Pituitary Tumor
* View prediction probabilities for each class

---

## Getting Started

### Prerequisites

* Python 3.7 or higher
* Install required packages:

  ```bash
  pip install streamlit tensorflow pillow numpy
  ```
* Place your trained model file (`model.h5`) in the project directory

### Running the App

1. Open terminal/command prompt and navigate to the project folder:

   ```bash
   cd d:\Brain_tumar
   ```
2. Start the Streamlit app:

   ```bash
   streamlit run run.py
   ```
3. The app will open in your browser. Upload an MRI image to get predictions.

---

## Project Files

* `run.py` — Main Streamlit app
* `model.h5` — Trained Keras model
* `README.md` — Project documentation
* `brain-diagnoses.ipynb` — Notebook with data analysis and model training 