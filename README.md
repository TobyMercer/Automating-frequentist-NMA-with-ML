# Automating Frequentist Network Meta-Analysis with Machine Learning

This project aims to streamline the process of conducting frequentist network meta-analysis (NMA) by incorporating machine learning techniques. The automated pipeline includes data preprocessing, model selection, hyperparameter tuning, and model evaluation. The goal is to provide a reusable framework for comparing multiple interventions in a more efficient and automated manner.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Introduction
Network meta-analysis is a valuable tool for comparing multiple interventions simultaneously and providing insights into their relative effectiveness. However, the traditional NMA process can be time-consuming and requires manual effort. This project explores the application of machine learning techniques to automate and streamline the NMA pipeline, making it more efficient and reproducible.

## Data
The project utilizes a dataset on smoking cessation interventions, obtained from the following source:
- Higgins JP, Jackson D, Barrett JK, Lu G, Ades AE, White IR. Consistency and inconsistency in network meta-analysis: concepts and models for multi-arm studies. Res Synth Methods. 2012 Jun;3(2):98-110. doi: [10.1002/jrsm.1044](https://onlinelibrary.wiley.com/doi/10.1002/jrsm.1044). PMID: 26062084; PMCID: PMC4433772.

The dataset includes information on various smoking cessation interventions and their effectiveness in terms of smoking abstinence rates.

## Methods
The automated NMA pipeline incorporates the following key components:
1. Data preprocessing: Handling missing data, converting data types, and preparing the dataset for analysis.
2. Feature engineering: Extracting relevant features from the dataset and creating new features as needed.
3. Model selection: Evaluating multiple machine learning models and selecting the best-performing model based on cross-validation metrics.
4. Hyperparameter tuning: Optimizing the hyperparameters of the selected model using techniques such as grid search or Bayesian optimization.
5. Model evaluation: Assessing the performance of the trained model using appropriate evaluation metrics and cross-validation techniques.

## Results
The project presents the results of the automated NMA pipeline, including:
- Comparison of different machine learning models and their performance metrics.
- Identification of the best-performing model and its hyperparameters.
- Visualization of the NMA results, including forest plots and treatment rankings.
- Assessment of inconsistency and heterogeneity in the network.

## Usage
To run the automated NMA pipeline, follow these steps:
1. Clone the repository: `git clone https://github.com/TobyMercer/Automating-frequentist-NMA-with-ML.git`
2. Navigate to the project directory: `cd Automating-frequentist-NMA-with-ML/`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Prepare your dataset in the appropriate format (refer to the data_sheet.md for details).
5. Run the Jupyter Notebook: `jupyter notebook automated_nma.ipynb`
6. Follow the instructions in the notebook to execute the automated NMA pipeline.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You are free to share and adapt the material for non-commercial purposes, as long as you give appropriate credit and indicate if changes were made. For more details, please see the [LICENSE](https://github.com/TobyMercer/Automating-frequentist-NMA-with-ML/blob/main/LICENSE.md) file.
