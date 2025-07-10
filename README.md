# Credit-Card-Fraud-Detection

This project demonstrates various methods for detecting credit card fraud in an imbalanced dataset using Python. The workflow and results are documented based on the provided notebook.

**Data Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

## Workflow Overview

1. **Data Loading and Exploration**
   - Loads the dataset using pandas.
   - Explores the data with `info()`, `describe()`, and class distribution.
   - Visualizes feature distributions using seaborn and matplotlib.
   - **Note:** No feature engineering or transformation (such as scaling, encoding, or selection) is applied; the focus is on handling the imbalanced target label.
2. **Preprocessing and Splitting**
   - Splits the data into features (`X`) and target (`y`).
   - Performs a train-test split.

3. **Handling Imbalanced Data**
   - Uses the SMOTETomek technique from `imblearn` to balance the training data.

4. **Model Training and Evaluation**
   - Trains three logistic regression models:
     - On original data
     - On original data with class weights
     - On resampled (balanced) data
   - Trains three random forest models with similar strategies.
   - Evaluates all models using classification reports.

5. **Neural Network Approach**
   - Builds a simple neural network using TensorFlow/Keras.
   - Trains on both original and resampled data.
   - Computes class weights for further balancing.
   - Evaluates neural network predictions.

## Methods Used

- **Logistic Regression**: Baseline, class-weighted, and resampled.
- **Random Forest**: Baseline, class-weighted, and resampled.
- **SMOTETomek**: For balancing the dataset.
- **Neural Network**: Simple feedforward model with dropout.

## Results Summary

- **Imbalanced Data**: Baseline models perform poorly on the minority class.
- **Class Weights**: Improve recall for the minority class.
- **SMOTETomek Resampling**: Further improves minority class detection.
- **Neural Network**: Shows similar trends; balancing and class weights help improve fraud detection but show overfitting.

## How to Run

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Install Jupyter and run the notebook:
   ```bash
   pip install notebook ipykernel
   jupyter notebook
   ```
3. Select the `.venv` kernel in VS Code or Jupyter.
4. Run all cells in the notebook to reproduce the results.

## Notes
- The notebook demonstrates the importance of handling class imbalance in fraud detection.
- Results may vary depending on the dataset and random seed.

---

For more details, see the notebook file in this repository.
