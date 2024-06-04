# Automating Frequentist Network Meta-Analysis with Machine Learning

## Non-technical Explanation
This project aims to streamline the process of conducting frequentist network meta-analysis (NMA) by incorporating machine learning techniques. The automated pipeline includes data preprocessing, model selection, hyperparameter tuning, and model evaluation. The goal is to provide a reusable framework for comparing multiple interventions in a more efficient and automated manner.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Model](#model)
- [Hyperparameter Optimisation](#hyperparameter-optimisation)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contact Details](#contact-details)
- [Licence](#licence)

## Introduction
Network meta-analysis is a valuable tool for comparing multiple interventions simultaneously and providing insights into their relative effectiveness. However, the traditional NMA process can be time-consuming and requires manual effort. This project explores the application of machine learning techniques to automate and streamline the NMA pipeline, making it more efficient and reproducible.

## Data
The project utilises a dataset on smoking cessation interventions, obtained from the following source:
- Higgins JP, Jackson D, Barrett JK, Lu G, Ades AE, White IR. Consistency and inconsistency in network meta-analysis: concepts and models for multi-arm studies. Res Synth Methods. 2012 Jun;3(2):98-110. doi: [10.1002/jrsm.1044](https://onlinelibrary.wiley.com/doi/10.1002/jrsm.1044). PMID: 26062084; PMCID: PMC4433772.

The dataset includes information on various smoking cessation interventions and their effectiveness in terms of smoking abstinence rates.

## Model
The automated NMA pipeline employs a range of machine learning models, including:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Elastic Net
- Decision Trees
- Random Forests
- Gradient Boosting Machines
- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)
- Gaussian Process Regression

These models were chosen based on their ability to handle complex relationships between features and target variables, as well as their interpretability and robustness. The pipeline automatically selects the best-performing model based on cross-validation metrics, ensuring optimal results for the given dataset.

## Hyperparameter Optimisation
The following hyperparameters were considered for optimization:
- Regularization strength (alpha) for Lasso, Ridge, and Elastic Net models
- Maximum depth and minimum samples per leaf for Decision Trees
- Number of trees, maximum depth, and minimum samples per split for Random Forests
- Learning rate, number of trees, and maximum depth for Gradient Boosting Machines
- Kernel type, regularization parameter (C), and kernel coefficient (gamma) for SVR
- Number of neighbors (K) and distance metric for KNN

Hyperparameter tuning was performed using a combination of grid search and random search techniques. The pipeline defined a range of values for each hyperparameter and systematically evaluated different combinations to identify the best-performing set of hyperparameters.

## Results
The automated NMA pipeline achieved the following results:
- Best-performing model: Linear Regression
- Cross-validated performance metrics:
  - Mean Squared Error (MSE): 0.0000
  - R-squared (R2): 0.9986
- Treatment rankings:
  - Best intervention: Group counseling
  - Worst intervention: No contact

The results demonstrate the effectiveness of the automated NMA pipeline in comparing multiple interventions for smoking cessation. The treatment rankings obtained from the analysis were: Group counseling > Individual counseling > Self-help > No contact. However, it is important to note that these rankings differ from the expected rankings reported in Higgins 2012, which were: Individual counseling > Group counseling > Self-help > No contact. The discrepancies can be attributed to methodological differences, model specifications, handling of multi-arm trials, and inconsistency or heterogeneity in the network.

Despite these differences, the overall conclusions regarding the relative effectiveness of the interventions are consistent, with "No contact" being identified as the worst intervention in both analyses. Further investigation using a Bayesian approach and a detailed comparison with the methodology employed in Higgins 2012 would provide a more comprehensive understanding of the ranking differences.

## Dependencies
The following dependencies are required to run the code in this project:

- Python (version 3.6 or higher)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NetworkX
- PyMC
- ArviZ
- Graphviz
- TensorFlow
- rpy2
- PyGAM

To install the dependencies, you can use the following command:

pip install -r requirements.txt

Additionally, make sure you have the following system dependencies installed:

- libudunits2-dev
- libgdal-dev
- libgeos-dev
- libproj-dev

You can install these system dependencies using the following command (for Ubuntu/Debian-based systems):

sudo apt-get install -y libudunits2-dev libgdal-dev libgeos-dev libproj-dev

## Usage
To run the automated NMA pipeline, follow these steps:
1. Clone the repository: `git clone https://github.com/TobyMercer/Automating-frequentist-NMA-with-ML.git`
2. Navigate to the project directory: `cd Automating-frequentist-NMA-with-ML/`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Prepare your dataset in the appropriate format (refer to the data_sheet.md for details).
5. Run the Jupyter Notebook: `jupyter notebook Automating_frequentist_NMA_with_ML.ipynb`
6. Follow the instructions in the notebook to execute the automated NMA pipeline.

## Contact Details
For any inquiries or collaboration opportunities, please contact:
- Name: Toby Mercer
- Email: tobias.charles.mercer@gmail.com
- GitHub: [TobyMercer](https://github.com/TobyMercer)

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You are free to share and adapt the material for non-commercial purposes, as long as you give appropriate credit and indicate if changes were made. For more details, please see the [LICENSE](https://github.com/TobyMercer/Automating-frequentist-NMA-with-ML/blob/main/LICENSE.md) file.
