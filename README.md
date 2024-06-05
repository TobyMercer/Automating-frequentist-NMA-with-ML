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
- [License](#license)

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

**Key Steps in the Automated Pipeline**

Data Preprocessing:

The first step in the automated pipeline is to prepare the input data for analysis. This involves handling missing values and converting the data into a format that can be easily processed by the machine learning algorithms. The pipeline identifies any missing information and fills in those gaps using a technique called "Multiple Imputation by Chained Equations" (MICE). This method intelligently guesses the missing values based on the available data, ensuring that the dataset is complete and ready for the next steps.

Feature Engineering:

In this step, the pipeline creates new features or variables from the existing data to provide more meaningful information for the analysis. It calculates treatment effect estimates and confidence intervals for each study, which are important indicators of the effectiveness of the interventions being compared. These engineered features help the machine learning algorithms better understand the relationships between the input data and the target variable (treatment effects).

Model Selection:

The pipeline then explores various machine learning models to find the one that best fits the data and predicts the treatment effects accurately. It considers a range of models, including linear regression, decision trees, random forests, and gradient boosting. These models are chosen based on their ability to handle complex relationships and provide interpretable results. The pipeline evaluates each model's performance using a technique called cross-validation, which involves splitting the data into multiple subsets and testing the model's accuracy on each subset. This process helps identify the model that performs the best on unseen data.

Hyperparameter Tuning:

Each machine learning model has settings called hyperparameters that control its behavior and performance. The pipeline optimises these hyperparameters to fine-tune the selected model for the specific dataset. It searches through different combinations of hyperparameters using a technique called grid search, which systematically tries out various settings to find the combination that yields the best results. This step ensures that the chosen model is well-calibrated and performs optimally for the given data.

Model Evaluation:

After selecting the best model and optimising its hyperparameters, the pipeline evaluates its performance using appropriate metrics. These metrics, such as mean squared error (MSE) or R-squared (R2), measure how well the model predicts the treatment effects compared to the actual observed values. The evaluation step provides insights into the model's accuracy and reliability, helping to assess its usefulness for comparing interventions and supporting decision-making.

By automating these key steps, the pipeline streamlines the network meta-analysis process, reducing manual effort and potential errors. It leverages machine learning techniques to efficiently handle data preprocessing, create informative features, select the best-performing model, optimise hyperparameters, and evaluate the model's performance. This automation enables researchers to focus on interpreting the results and making evidence-based decisions.

**Benefits of Using Machine Learning in Network Meta-Analysis**

Integrating machine learning (ML) techniques into the network meta-analysis (NMA) process offers several key benefits, particularly in terms of improved efficiency and automation:

1. Streamlined Workflow: By automating various steps of the NMA pipeline, such as data preprocessing, feature engineering, and model selection, ML reduces the manual effort required to conduct the analysis. This streamlined workflow allows researchers to focus more on interpreting the results and making evidence-based decisions, rather than spending time on repetitive and time-consuming tasks.

2. Efficient Handling of Large Datasets: ML algorithms are designed to handle large and complex datasets efficiently. As the volume of available studies and data continues to grow, ML techniques enable researchers to process and analyse this information more quickly and effectively. The automated pipeline can easily scale to accommodate larger datasets, ensuring that the analysis remains feasible and manageable.

3. Improved Accuracy and Reliability: ML algorithms are capable of identifying complex patterns and relationships within the data that may be difficult for humans to discern. By leveraging the power of ML, the automated NMA pipeline can potentially improve the accuracy and reliability of treatment effect estimates and rankings. The use of cross-validation and other evaluation techniques helps ensure that the selected model is robust and generalisable.
   
4. Reduced Human Bias: Traditional NMA methods often involve manual decisions and judgments, which can introduce human bias into the analysis. By automating the process using ML, the automated NMA pipeline reduces the potential for human bias, ensuring a more objective and data-driven approach. The pipeline applies consistent criteria and algorithms across all studies, minimising the influence of subjective preferences or preconceptions.
   
5. Reproducibility and Transparency: The automated NMA pipeline promotes reproducibility and transparency in the analysis process. The pipeline's code and algorithms can be easily shared and replicated, allowing other researchers to validate the results and build upon the work. This transparency enhances the credibility and trustworthiness of the findings, as the analysis steps are clearly documented and can be scrutinised by the scientific community.
   
6. Faster Iteration and Refinement: ML techniques enable researchers to quickly iterate and refine the NMA process. The automated pipeline can be easily modified and updated to incorporate new data, models, or techniques. This flexibility allows researchers to adapt to evolving research questions and incorporate the latest advancements in ML and NMA methodologies.

By harnessing the power of ML, the automated NMA pipeline offers significant benefits in terms of efficiency, accuracy, and reproducibility. It streamlines the analysis process, reduces manual effort, and enables researchers to handle large datasets effectively. The use of ML techniques also helps minimise human bias, improves the reliability of the results, and promotes transparency in the scientific process.

## Hyperparameter Optimisation
The following hyperparameters were considered for optimisation:
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

### Python Dependencies
The following Python dependencies are required to run the code in this project:

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

To install the Python dependencies, you can use the following command:

pip install -r requirements.txt

## R Dependencies
The following R packages are required to run the code in this project:

- netmeta
- dplyr
- ggplot2

To install the R dependencies, you can use the following commands in R:

install.packages("netmeta")
install.packages("dplyr")
install.packages("ggplot2")

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
