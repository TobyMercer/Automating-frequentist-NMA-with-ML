# Model Card

## Model Description

**High Level Overview**

Network meta-analysis (NMA) is a statistical method that allows for the simultaneous comparison of multiple interventions by combining direct and indirect evidence from a network of studies. The traditional NMA process involves several steps, including data extraction, network construction, model specification, parameter estimation, and results interpretation.

In this project, we integrate machine learning (ML) techniques into the NMA process to automate and streamline the analysis. The automated NMA pipeline leverages ML algorithms to perform key tasks, such as data preprocessing, feature engineering, model selection, and evaluation.

The pipeline starts by preprocessing the input data, handling missing values, and converting relevant columns to appropriate data types. It then performs feature engineering to derive additional informative features, such as treatment effect estimates and confidence intervals, from the original dataset.

To address missing data, the pipeline employs the Multiple Imputation by Chained Equations (MICE) technique, which imputes missing values based on the observed data. This step ensures that the dataset is complete and ready for analysis.

Next, the pipeline utilises various ML algorithms, including linear regression, decision trees, random forests, and gradient boosting, to model the relationship between the input features and the target variable (treatment effects). These algorithms are chosen based on their ability to handle complex relationships and their interpretability.

The pipeline performs model selection and hyperparameter tuning using cross-validation and grid search techniques. It evaluates multiple models and their hyperparameter combinations to identify the best-performing model based on specific evaluation metrics, such as mean squared error (MSE) or R-squared (R2).

Once the best model is selected, it is trained on the entire dataset using the optimised hyperparameters. The trained model is then used to estimate the relative treatment effects and generate treatment rankings for the interventions being compared.

The automated NMA pipeline streamlines the analysis process by automating data preprocessing, handling missing data, and performing model selection and evaluation. It provides a more efficient and reproducible approach to conducting NMA, reducing manual effort and potential errors.

By integrating ML techniques into the NMA process, this project aims to enhance the efficiency and reliability of treatment comparisons, ultimately supporting evidence-based decision-making in healthcare.

**Input:**
- Study-level data from 24 trials comparing different smoking cessation interventions, provided in the "smokingcessation.csv" file. The data includes:
  - Study identifier (Study)
  - Treatment comparison design (Design)
  - Number of events (dA, dB, dC, dD) and total number of participants (nA, nB, nC, nD) for each treatment arm (A, B, C, D)
  - Indicator variables (hasA, hasB, hasC, hasD) denoting which treatments are included in each study
- The treatments being compared are:
  - A: No contact
  - B: Self-help
  - C: Individual counselling
  - D: Group counselling
 - The outcome measure is the number of successful smoking cessation events out of the total number of participants in each treatment arm.

**Output:**
- Relative treatment effect estimates and 95% credible intervals for each pairwise comparison of smoking cessation interventions.
- Ranking of interventions based on their estimated effectiveness.
- Measures of inconsistency and heterogeneity in the network meta-analysis.

**Model Architecture:**

1. Data Preprocessing:

Purpose: The data preprocessing step aims to clean and prepare the input data for analysis. It handles missing values and ensures that the data is in a format suitable for the machine learning algorithms.

Functionality:

- Missing value handling: The pipeline identifies missing values in the dataset and replaces them with NaN (Not a Number) values. This step is crucial for maintaining data integrity and consistency.
  
- Data type conversion: Relevant columns containing numerical data, such as the number of events (dA, dB, dC, dD) and total participants (nA, nB, nC, nD), are converted to the float data type. This conversion ensures that the data is properly formatted for mathematical operations and analysis.
  
- Removal of incomplete rows: Rows where all event columns (dA, dB, dC, dD) are NaN are removed from the dataset. This step eliminates rows that do not contribute any meaningful information to the analysis.

Code snippet:
```
import pandas as pd
import numpy as np

# Replace '.' with NaN
data = data.replace('.', np.nan)

# Convert relevant columns to float
data[['dA', 'nA', 'dB', 'nB', 'dC', 'nC', 'dD', 'nD']] = data[['dA', 'nA', 'dB', 'nB', 'dC', 'nC', 'dD', 'nD']].astype(float)

# Remove rows where all event columns are NaN
data = data.dropna(subset=['dA', 'dB', 'dC', 'dD'], how='all')
```

2. Feature Engineering:
   
Purpose: The feature engineering step creates new features or variables from the existing data to provide more informative inputs for the machine learning models. These engineered features capture relevant information and relationships that can improve the models' predictive performance.

Functionality:

- Treatment effect estimation: The pipeline calculates treatment effect estimates for each study by computing the proportion of successful events (e.g., dA/nA for treatment A).
- Confidence interval calculation: Confidence intervals are computed for each treatment effect estimate to quantify the uncertainty associated with the estimates. The confidence intervals are typically calculated using the normal approximation method.
- Pairwise odds ratios: The pipeline calculates pairwise odds ratios and their corresponding confidence intervals for each treatment comparison within each study. These odds ratios provide a measure of the relative effectiveness of the treatments.

Code snippet:
```
# Calculate treatment effect estimates
data['A_Effect'] = data['dA'] / data['nA']
data['B_Effect'] = data['dB'] / data['nB']
data['C_Effect'] = data['dC'] / data['nC']
data['D_Effect'] = data['dD'] / data['nD']

# Calculate confidence intervals for treatment effect estimates
data['A_Lower_CI'] = data['A_Effect'] - 1.96 * np.sqrt(1 / data['nA'])
data['A_Upper_CI'] = data['A_Effect'] + 1.96 * np.sqrt(1 / data['nA'])
# ... (similar calculations for treatments B, C, and D)

# Calculate pairwise odds ratios and confidence intervals
data['OR_AB'] = (data['dA'] / (data['nA'] - data['dA'])) / (data['dB'] / (data['nB'] - data['dB']))
data['OR_AB_Lower_CI'] = np.exp(np.log(data['OR_AB']) - 1.96 * np.sqrt(1 / data['dA'] + 1 / (data['nA'] - data['dA']) + 1 / data['dB'] + 1 / (data['nB'] - data['dB'])))
data['OR_AB_Upper_CI'] = np.exp(np.log(data['OR_AB']) + 1.96 * np.sqrt(1 / data['dA'] + 1 / (data['nA'] - data['dA']) + 1 / data['dB'] + 1 / (data['nB'] - data['dB'])))
# ... (similar calculations for other treatment comparisons)
```

3. Missing Data Imputation:
   
Purpose: The missing data imputation step aims to handle missing values in the dataset by estimating and filling in the missing information. This step is crucial for ensuring that the machine learning models can utilise all available data points and avoid biased results due to missing data.

Functionality:

- Multiple Imputation by Chained Equations (MICE): The pipeline employs the MICE technique to impute missing values in the engineered features. MICE is an iterative algorithm that imputes missing values based on the observed data and the relationships between variables.
- Iterative imputation: The imputation process is performed iteratively for a specified number of iterations (e.g., max_iter=10) to refine the estimates of the missing values. In each iteration, the algorithm imputes missing values based on the current estimates of the other variables.
- Imputed data generation: The final imputed dataset is generated by combining the observed data with the imputed values. This complete dataset is then used for subsequent steps in the pipeline.

Code snippet:
```
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create an instance of the IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)

# Perform missing data imputation
imputed_data = imputer.fit_transform(data)
```
4. Data Normalisation:
   
Purpose: The data normalization step scales and standardizes the features to ensure that they have similar ranges and distributions. Normalisation helps improve the convergence and performance of many machine learning algorithms.

Functionality:

- Feature scaling: The pipeline applies feature scaling techniques, such as standardization or normalization, to bring all features to a similar scale. Standardization transforms the features to have zero mean and unit variance, while normalization scales the features to a specified range (e.g., [0, 1]).
- Handling outliers: Normalization techniques can help mitigate the impact of outliers by reducing their influence on the scaled features. This is particularly important for algorithms that are sensitive to the scale of the input features.
- Improved algorithm performance: Normalized features can lead to faster convergence and better performance of machine learning algorithms, especially those based on gradient descent optimization.

Code snippet:
```
from sklearn.preprocessing import StandardScaler

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Normalize the imputed data
normalized_data = scaler.fit_transform(imputed_data)
```

5. Model Selection:

Purpose: The model selection step involves evaluating multiple machine learning models to identify the one that best fits the data and achieves the highest predictive performance. This step ensures that the most suitable model is chosen for the specific NMA task.

Functionality:

- Model comparison: The pipeline compares various machine learning models, such as linear regression, Lasso, Ridge, decision trees, random forests, and gradient boosting. These models are selected based on their ability to handle complex relationships and provide interpretable results.
- Cross-validation: The pipeline employs cross-validation techniques, such as k-fold cross-validation, to assess the performance of each model. Cross-validation involves splitting the data into multiple subsets, training the model on a subset, and evaluating its performance on the held-out subset. This process is repeated for each subset, and the average performance across all folds is used as a measure of the model's performance.
- Performance metrics: The pipeline uses appropriate performance metrics, such as mean squared error (MSE) or R-squared (R2), to evaluate the models. These metrics provide a quantitative measure of how well the model fits the data and predicts the target variable.
- Model selection criteria: The best-performing model is selected based on the chosen performance metric. The model with the lowest MSE or the highest R2 score is typically considered the most suitable for the NMA task.

Code snippet:
```
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Define the models and their corresponding hyperparameters
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Perform model selection using cross-validation
best_model = None
best_score = float('-inf')

for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()
    
    if mean_score > best_score:
        best_model = model
        best_score = mean_score
```

6. Hyperparameter Tuning:

Purpose: The hyperparameter tuning step involves optimizing the hyperparameters of the selected model to further improve its performance. Hyperparameters are settings that are not learned from the data but are set before training the model. Tuning these hyperparameters can significantly impact the model's performance.

Functionality:

- Hyperparameter search space: The pipeline defines a search space for the hyperparameters of the selected model. This search space specifies the range of values or the set of possible values for each hyperparameter.
- Grid search or random search: The pipeline employs techniques like grid search or random search to explore different combinations of hyperparameters. Grid search exhaustively evaluates all possible combinations, while random search samples a fixed number of random combinations from the search space.
- Cross-validation: Hyperparameter tuning is performed using cross-validation to assess the model's performance for each combination of hyperparameters. The combination that yields the best performance is selected as the optimal set of hyperparameters.
- Final model training: Once the optimal hyperparameters are determined, the final model is trained on the entire dataset using these hyperparameters. This final model is then used for making predictions and generating the NMA results.

Code snippet:
```
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter search space
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Perform hyperparameter tuning using grid search
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best hyperparameters and the tuned model
best_params = grid_search.best_params_
tuned_model = grid_search.best_estimator_
```

7. Model Evaluation:

Purpose: The model evaluation step assesses the performance of the trained model on unseen data to determine its effectiveness and generalizability. It provides insights into how well the model can predict treatment effects and generate reliable NMA results.

Functionality:

- Performance metrics: The pipeline uses various performance metrics to evaluate the model's performance. These metrics may include mean squared error (MSE), mean absolute error (MAE), and R-squared (R2). MSE measures the average squared difference between the predicted and actual values, MAE measures the average absolute difference, and R2 represents the proportion of variance in the target variable that is predictable from the input features.
- Train-test split: The pipeline splits the data into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance on unseen data. This split helps assess how well the model generalizes to new data.
- Cross-validation: In addition to the train-test split, the pipeline may employ cross-validation techniques, such as k-fold cross-validation, to obtain a more robust estimate of the model's performance. Cross-validation involves splitting the data into multiple subsets, training and evaluating the model on different combinations of these subsets, and averaging the results.
- Model performance interpretation: The evaluation results are interpreted to assess the model's performance. Lower values of MSE and MAE indicate better predictive accuracy, while higher values of R2 suggest a better fit to the data. The pipeline may also compare the model's performance to baseline or reference models to determine its relative effectiveness.

Code snippet:
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the tuned model on the training set
tuned_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = tuned_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")
```
Interpretation:

The model evaluation results provide insights into the model's performance and its ability to generate reliable NMA results. Lower values of MSE and MAE indicate that the model's predictions are close to the actual values, suggesting good predictive accuracy. A high R2 value indicates that a significant proportion of the variance in the target variable can be explained by the model, suggesting a good fit to the data.

However, it is important to interpret the evaluation results in the context of the specific NMA problem and the limitations of the data. The model's performance may be affected by factors such as the quality and representativeness of the input data, the presence of outliers or influential points, and the assumptions made during the modeling process.

8. Treatment Effect Estimation:
   
The trained model is used to predict the treatment effects for each study in the dataset. These predicted treatment effects represent the model's estimates of the relative effectiveness of the interventions based on the input features. The predicted treatment effects, along with the observed effects and study information, are stored in a results DataFrame for further analysis and interpretation.

9. Visualization and Interpretation:
    
To facilitate understanding and interpretation of the NMA results, several visualization techniques are employed. A forest plot is used to display the predicted treatment effects and their confidence intervals for each study. This plot provides a visual summary of the relative effectiveness of the interventions across studies and allows for easy comparison of the results.

Treatment rankings are calculated based on the mean predicted effects, providing an ordering of the interventions from most effective to least effective. These rankings can be compared with the expected rankings from reference studies, such as Higgins 2012, to assess the consistency and validity of the automated NMA approach.

Inconsistency assessment is performed using the node-splitting approach, which evaluates the consistency of the network meta-analysis results by comparing the direct and indirect evidence for each treatment comparison. This assessment helps identify potential sources of inconsistency in the network and provides insights into the reliability of the NMA findings.

The visualization and interpretation of the NMA results should be done in collaboration with domain experts to ensure that the findings are contextually relevant and aligned with existing knowledge in the field. The results should be interpreted cautiously, considering the limitations of the data and the assumptions made during the modeling process.

It is important to note that while the automated frequentist NMA pipeline provides a valuable tool for exploring and analyzing data, it should not be used as the sole basis for policy-making or healthcare decision-making. Further validation, particularly using Bayesian methods, and comparison with established guidelines and expert judgment are necessary to establish the robustness and reliability of the automated NMA approach.

Summary:

The automated frequentist NMA pipeline presented in this model card combines various machine learning techniques and statistical methods to compare the effectiveness of smoking cessation interventions across multiple studies. The pipeline incorporates the following key components:

1. Data Preprocessing: Handling missing values, converting data types, and removing incomplete rows to ensure data quality and consistency.
2. Feature Engineering: Creating informative features, such as treatment effect estimates, confidence intervals, and pairwise odds ratios, to capture relevant information for the analysis.
3. Missing Data Imputation: Employing the Multiple Imputation by Chained Equations (MICE) technique to estimate and fill in missing values, allowing the models to utilize all available data.
4. Data Normalization: Scaling and standardizing the features to improve the convergence and performance of the machine learning algorithms.
5. Model Selection: Evaluating multiple machine learning models using cross-validation to identify the best-performing model for the NMA task.
6. Hyperparameter Tuning: Optimizing the hyperparameters of the selected model to further improve its performance and generalizability.
7. Model Evaluation: Assessing the trained model's performance using various metrics, such as mean squared error (MSE), mean absolute error (MAE), and R-squared (R2), to determine its effectiveness and reliability.
8. Treatment Effect Estimation: Utilizing the trained model to predict treatment effects for each study and storing the results along with the observed effects and study information.
9. Visualization and Interpretation: Visualizing the NMA results using forest plots, calculating treatment rankings, and assessing inconsistency to facilitate understanding and interpretation of the findings.

This automated NMA pipeline provides a research-oriented approach to comparing the effectiveness of smoking cessation interventions, leveraging machine learning techniques to streamline the analysis process and generate insights from the available data. However, it is important to interpret the results with caution and consider the limitations of the frequentist approach.

While this pipeline offers a valuable tool for exploring and automating the NMA process, it should not be used directly for policy-making or healthcare decision-making without further validation and comparison with Bayesian methods. The National Institute for Health and Care Excellence (NICE) recommends using Bayesian approaches for NMA, as they provide a more comprehensive framework for handling uncertainty and incorporating prior knowledge.

Researchers and practitioners should view this automated frequentist NMA pipeline as a complementary tool to traditional methods, providing an efficient and reproducible way to analyze data and generate hypotheses. However, the results should be interpreted in the context of the specific research question, the quality and limitations of the input data, and the underlying assumptions of the models.

Further research and validation, particularly using Bayesian methods, are necessary to establish the robustness and reliability of the automated NMA approach. Collaboration between machine learning experts and domain specialists is crucial to refine the pipeline, incorporate domain knowledge, and ensure the validity and interpretability of the results.

## Performance

- The best-performing model is Linear Regression with the following performance metrics (refer to Cell 14.05 and Cell 15.03):
  - Mean Squared Error (MSE): 0.0000
  - Mean Absolute Error (MAE): 0.0044 (refer to Cell 16.01)
  - R-squared (R2): 0.9237 (refer to Cell 16.01)
- Performance is evaluated using cross-validation with different splitting strategies to ensure robustness and generalisability (refer to Cell 15.04).
- Model performance is assessed using bootstrap resampling (refer to Cell 16.01).
- Visualisations:
  - Network diagram showing the network of treatments:
    ![Network Diagram](https://github.com/TobyMercer/Automating-frequentist-NMA-with-ML/blob/main/images/Cell_3.01.png)
  - Forest plot of relative treatment effect estimates and 95% credible intervals for each pairwise comparison before the NMA:
    ![Forest plot before NMA](https://github.com/TobyMercer/Automating-frequentist-NMA-with-ML/blob/main/images/Cell_3.02.png)
  - Heatmap of treatment rankings in each study:
    ![Heatmap of treatment rankings](https://github.com/TobyMercer/Automating-frequentist-NMA-with-ML/blob/main/images/Cell_3.03.png)
  - Forest plot of relative treatment effect estimates and 95% credible intervals for each pairwise comparison after the NMA:
    ![Forest plot after NMA](https://github.com/TobyMercer/Automating-frequentist-NMA-with-ML/blob/main/images/Cell_20.01.png)

## Limitations

- Potential biases in the included studies, such as publication bias or lack of representation of certain patient subgroups, which may affect the generalisability of the findings.
- Assumes transitivity (i.e., consistency of intervention effects across different study designs and populations), which may not always hold. The model includes inconsistency parameters to assess this assumption, but the presence of inconsistency may limit the reliability of the results.
- Limited to study-level covariates available in the provided dataset, which does not include potentially relevant effect modifiers such as patient demographics, intervention details (e.g., dose, duration), or specific study design characteristics (e.g., randomisation, blinding).
- Relies on aggregate study-level data rather than individual patient data, which may lead to ecological bias and limit the ability to explore patient-level heterogeneity.
- The model does not account for the risk of bias or quality of individual studies, which may influence the validity of the findings.
- The use of a frequentist approach may limit the ability to fully incorporate prior knowledge and quantify uncertainty compared to Bayesian methods.
- The model's performance and validity may be limited by the relatively small number of studies included in the analysis (24 trials).

## Trade-offs

- Ensemble modeling approach improves predictive performance but increases computational complexity and may reduce interpretability compared to simpler models.
- Inclusion of a wide range of studies increases generalisability but may also introduce heterogeneity and potential biases.
- The use of frequentist methods, such as maximum likelihood estimation and restricted maximum likelihood (REML), allows for the estimation of model parameters and the assessment of model fit.
- The model provides rankings of interventions but should not be used to make definitive conclusions about effectiveness without considering uncertainty and limitations.
- The analysis uses a random-effects model to account for heterogeneity between studies, which assumes that the true treatment effects are drawn from a common distribution.
- The inconsistency models allow for the assessment of potential conflicts between direct and indirect evidence in the network meta-analysis.
- The use of network meta-regression techniques enables the exploration of potential effect modifiers and sources of heterogeneity or inconsistency.

## Intended Use

- Intended for research purposes to compare the effectiveness of smoking cessation interventions across multiple studies using a frequentist network meta-analysis approach.
- Not intended to guide policy decisions or treatment recommendations, as the National Institute for Health and Care Excellence (NICE) recommends the use of Bayesian techniques for network meta-analysis, which are likely superior.
- This project serves as a stepping stone towards potentially automating Bayesian network meta-analysis using machine learning techniques in the future.
- Not intended for making individual-level treatment recommendations or causal inferences without considering study design and potential confounding.
- Intended users include researchers interested in exploring the automation of frequentist network meta-analysis methods and comparing them with Bayesian approaches.
- The results should be interpreted with caution and should not be used directly for policy-making or healthcare decision-making without further validation and comparison with Bayesian methods.

## Ethical Considerations

- Potential for model misuse if rankings are interpreted as definitive conclusions about intervention effectiveness without considering uncertainty and limitations.
- It is important to emphasise that this model is intended for research purposes and should not be used directly for policy-making or treatment recommendations without further validation and comparison with Bayesian methods.
- Model development process included outlier detection and sensitivity analyses to mitigate potential biases.
- Transparency about model limitations, assumptions, and the exploratory nature of this research is crucial for responsible use and interpretation of results.
- Users should be aware that this model is part of a larger research effort aimed at potentially automating Bayesian network meta-analysis using machine learning techniques in the future.
- The results should be communicated clearly, emphasising the limitations of the frequentist approach and the need for further validation and comparison with Bayesian methods before any practical application.

## Caveats and Recommendations

- Model rankings are estimates with uncertainty and should be interpreted cautiously, especially when using frequentist methods.
- Users should consider the confidence intervals and measures of inconsistency/heterogeneity when interpreting the rankings.
- Results should be used in conjunction with other sources of evidence, such as expert judgment and mechanistic understanding, and should not be relied upon solely for decision-making.
- Limitations of relying on aggregate study-level data should be considered when applying the model, as individual-level covariates and potential effect modifiers may not be captured.
- The model should be viewed as an exploratory tool for researching the automation of network meta-analysis using machine learning techniques, rather than as a definitive guide for policy or treatment recommendations.
- Future research should focus on extending this work to Bayesian methods, which are preferred by organisations such as NICE for network meta-analysis, and validating the automated approach against established manual methods.
- Users should be transparent about the limitations and caveats of the model when presenting or publishing results, emphasising the need for further validation and the transition to Bayesian methods in the future.
