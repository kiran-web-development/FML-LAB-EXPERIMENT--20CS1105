

<div style="background: linear-gradient(to right, #ff6b6b20, #ff6b6b40); border-left: 4px solid #dc3545; padding: 20px; border-radius: 10px; margin: 30px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h2 style="color: #dc3545; font-family: 'Poppins', 'Segoe UI', sans-serif; margin: 0 0 15px 0; font-size: 1.5em;">
  ⚠️ Important Disclaimer
</h2>
<p style="font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 1.1em; color: #dc3545; line-height: 1.6; margin: 0;">
  This repository is intended for <strong>educational purposes only</strong>. It serves as a resource to understand program structures, implementation approaches, and the usage of datasets in machine learning concepts.
  <br><br>
  <strong>IMPORTANT WARNING:</strong> Any form of code copying during examinations is strictly prohibited. The author bears no responsibility for academic misconduct or consequences resulting from the misuse of this educational resource. Students are advised to use this repository responsibly for learning and understanding concepts only.
</p>
</div>

<div align="center" style="background: linear-gradient(45deg, #1a237e, #4a148c); padding: 20px; border-radius: 15px; margin-bottom: 30px;">
<h1 style="color: white; font-family: 'Poppins', 'Roboto', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); letter-spacing: 0.5px;">
  🎓 Fundamentals of Machine Learning Lab Experiments
  <div style="font-size: 0.8em; margin-top: 10px; font-family: 'Inter', 'Roboto Mono', monospace;">20CS1105</div>
</h1>

<div style="margin: 20px 0;">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg?style=for-the-badge" alt="Python 3.x">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License: MIT">
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange.svg?style=for-the-badge" alt="Platform">



</div>

<p style="color: #e0e0e0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Open Sans', sans-serif; font-size: 1.2em; max-width: 800px; margin: 20px auto; line-height: 1.6;">
  A comprehensive collection of machine learning algorithms implemented in Python, designed for both local environments and Google Colab.
</p>
</div>

<h2 style="color:rgb(195, 0, 255); font-family: 'Segoe UI', sans-serif; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
  📚 Program Structure
</h2>

<div style="background: linear-gradient(to right, #f6f8fa, #ffffff); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
<p style="font-family: 'Segoe UI', sans-serif; font-size: 1.1em; color: #34495e;">
Each program in this repository follows a consistent structure:</p>

```python
# Aim
# Clear statement of the program's objective

# Program
# Implementation in Python

# Dataset (if required)
# Description and usage of the dataset

# Output
# Expected results and visualizations

# Result
# Confirmation of successful execution
```

<h2 style="color: #2c3e50; font-family: 'Segoe UI', sans-serif; border-bottom: 3px solid #27ae60; padding-bottom: 10px;">
  🔍 Programs Included
</h2>

<div style="background: linear-gradient(to right, #f0fff4, #ffffff); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">

1.  **Back Propagation Algorithm**
   - **Aim**: Implement neural network training using backpropagation
   - **Program Components**:
     - Neural network implementation
     - Backpropagation training logic
     - Error calculation and weight updates
   - **Libraries**: NumPy, Matplotlib
   - **Output Format**: Training progress and error visualization

2. **Bayesian Network for Medical Dataset**
   - Medical diagnosis using Bayesian networks
   - Dataset: Medical Dataset.csv
   - Libraries: pandas, pgmpy
   - Note: Upload 'Medical Dataset.csv' to your Colab workspace

3. **Classification using Multilayer Perceptron**
   - Image classification using MLP
   - Dataset: food5k (organized in training/validation sets)
   - Libraries: PyTorch, OpenCV, albumentations
   - Note: Create the same directory structure in Colab and upload your images,and the program was have some errors in the code.

4. **Decision Tree Algorithm**
   - Decision tree implementation for classification
   - Dataset: dataset.csv
   - Libraries: scikit-learn, pandas, matplotlib
   - Output: Generates decision_tree.png visualization

5. **Expectation Maximization Algorithm**
   - Clustering using EM algorithm and comparison with K-Means
   - Dataset: Iris dataset (built-in)
   - Libraries: scikit-learn, pandas, matplotlib
   - Output: Saves clustering_comparison.png

6. **K-Nearest Neighbor Algorithm**
   - KNN classification on Iris dataset
   - Dataset: iris_flower_dataset.csv
   - Libraries: scikit-learn, pandas, seaborn
   - Output: Generates confusion matrices and visualizations

7. **Locally Weighted Regression Algorithm**
   - Non-parametric regression implementation
   - Dataset: tips.csv
   - Libraries: NumPy, pandas, matplotlib
   - Output: Saves locally_weighted_regression.png

8. **Naive Bayesian Classifier**
   - Text classification using Naive Bayes
   - Dataset: dataset.csv (text data)
   - Libraries: scikit-learn, pandas

9. **Principle Component Analysis**
   - **Aim**: Implement dimensionality reduction using PCA
   - **Program Components**:
     - Matrix operations and eigenvalue decomposition
     - Dimensionality reduction logic
   - **Libraries**: NumPy, matplotlib
   - **Output**: Visualization of reduced dimensions

10. **Naive Bayesian Classifier for Text Classification**
   - **Aim**: Implement text classification using Naive Bayes algorithm
   - **Program Components**:
     - Text preprocessing and vectorization
     - Naive Bayes classifier implementation
     - Evaluation metrics calculation
   - **Dataset**: Text classification dataset (dataset.csv)
   - **Libraries**: scikit-learn, pandas
   - **Output**: Classification metrics and accuracy report

<h2 style="color: #2c3e50; font-family: 'Segoe UI', sans-serif; border-bottom: 3px solid #e74c3c; padding-bottom: 10px;">
  🚀 Setup Instructions for Google Colab
</h2>

<div style="background: linear-gradient(to right, #fff5f5, #ffffff); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
<h3 style="color: #c0392b; font-family: 'Segoe UI', sans-serif; margin-bottom: 15px;">
  Step-by-Step Guide
</h3>

1. 📤 **Upload Datasets**
   - Upload the respective .csv files to your Colab workspace
   - Modify the file paths in the code to match Colab's structure
   Example:
   ```python
   # Local path
   df = pd.read_csv("dataset.csv")
   
   # Colab path (after uploading)
   df = pd.read_csv("/content/dataset.csv")
   ```

2. 📁 **Directory Structure for Image Data**
   For the MLP classifier, create this structure in Colab:
   
   <details>
   <summary>Click to expand directory structure</summary>
   ```
   food5k/
   ├── training/
   │   ├── 0/
   │   └── 1/
   └── validation/
       ├── 0/
       └── 1/
   ```
   </details>

3. 📦 **Install Required Libraries**
   ```python
   !pip install torch torchvision
   !pip install albumentations
   !pip install pgmpy
   !pip install seaborn
   !pip install scikit-learn
   ```

## 💻 Sample Code Blocks

<details>
<summary>🔄 Loading and Preprocessing Data</summary>

```python
# Example of data loading and preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("dataset.csv")

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('target', axis=1))
y = data['target']
```
</details>

<details>
<summary>📊 Visualization Example</summary>

```python
# Example of creating visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Create plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='feature1', y='feature2', hue='target')
plt.title('Data Visualization')
plt.savefig('visualization.png')
```
</details>

<div style="background-color: #f6f8fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
<h3>💡 Tips for Google Colab</h3>

```python
# Mount Google Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install -r requirements.txt

# Set working directory
%cd /content/drive/MyDrive/your_project_folder
```
</div>

## 📝 Important Notes

<div style="background-color:lightyellow; padding: 15px; border-radius: 5px; margin: 10px 0;">

1. 📊 **Output Files**
   - Many programs generate visualization outputs (.png files)
   - In Colab, these will be saved in the runtime environment
   - Use `files.download()` to download generated images

2. **Data Preprocessing**
   - Ensure datasets are in the correct format
   - Check for proper column names and data types
   - Handle any missing values if present

3. **Memory Management**
   - For large datasets, consider using smaller subsets
   - Clear output and restart runtime if needed

## Running the Programs

1. Open the desired .py file
2. Upload required dataset
3. Modify file paths if needed
4. Run all cells

Example for modifying paths in Colab:
```python
# Original
data = pd.read_csv("Navie Bayesian Classifier/Tennisdata.csv")

# In Colab
data = pd.read_csv("/content/Tennisdata.csv")
```

## 🛠️ Requirements

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; padding: 15px;">

### Core Libraries
- ![Python](https://img.shields.io/badge/Python-3.x-blue) Python 3.x
- ![NumPy](https://img.shields.io/badge/NumPy-Latest-green) NumPy
- ![Pandas](https://img.shields.io/badge/Pandas-Latest-yellow) Pandas
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange) Scikit-learn

### Visualization
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-blue) Matplotlib
- ![Seaborn](https://img.shields.io/badge/Seaborn-Latest-pink) Seaborn

### Deep Learning & Image Processing
- ![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red) PyTorch (for MLP)
- ![OpenCV](https://img.shields.io/badge/OpenCV-Latest-blue) OpenCV
- ![Albumentations](https://img.shields.io/badge/Albumentations-Latest-green) Albumentations

### Specialized Libraries
- ![pgmpy](https://img.shields.io/badge/pgmpy-Latest-purple) pgmpy (for Bayesian networks)

</div>

## 📊 Program Outputs

Each program generates specific outputs:

<details>
<summary>Click to view output details</summary>

1. **Decision Tree Algorithm**
   - `decision_tree.png`: Visual representation of the decision tree

2. **Expectation Maximization Algorithm**
   - `clustering_comparison.png`: Comparison between EM and K-Means clustering

3. **Locally Weighted Regression**
   - `locally_weighted_regression.png`: Regression plot with data points

4. **K-Nearest Neighbor**
   - Generated confusion matrices
   - Classification report
   - Visualization plots

</details>

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📚 Improve documentation

## 📄 License

This project is licensed under the MIT License.

<div align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red.svg" alt="Made with love-kiran">
</div>

