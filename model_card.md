# Model Card

## Model Description

**High Level Overview**
Network meta-analysis (NMA) is a statistical method that allows for the simultaneous comparison of multiple interventions by combining direct and indirect evidence from a network of studies. The traditional NMA process involves several steps, including data extraction, network construction, model specification, parameter estimation, and results interpretation.
In this project, we integrate machine learning (ML) techniques into the NMA process to automate and streamline the analysis. The automated NMA pipeline leverages ML algorithms to perform key tasks, such as data preprocessing, feature engineering, model selection, and evaluation.
The pipeline starts by preprocessing the input data, handling missing values, and converting relevant columns to appropriate data types. It then performs feature engineering to derive additional informative features, such as treatment effect estimates and confidence intervals, from the original dataset.
To address missing data, the pipeline employs the Multiple Imputation by Chained Equations (MICE) technique, which imputes missing values based on the observed data. This step ensures that the dataset is complete and ready for analysis.
Next, the pipeline utilizes various ML algorithms, including linear regression, decision trees, random forests, and gradient boosting, to model the relationship between the input features and the target variable (treatment effects). These algorithms are chosen based on their ability to handle complex relationships and their interpretability.
The pipeline performs model selection and hyperparameter tuning using cross-validation and grid search techniques. It evaluates multiple models and their hyperparameter combinations to identify the best-performing model based on specific evaluation metrics, such as mean squared error (MSE) or R-squared (R2).
Once the best model is selected, it is trained on the entire dataset using the optimized hyperparameters. The trained model is then used to estimate the relative treatment effects and generate treatment rankings for the interventions being compared.
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
   - Missing values in the dataset are replaced with NaN (Not a Number) values.
   - Relevant columns containing numerical data are converted to float data type.
   - Rows where all event columns (dA, dB, dC, dD) are NaN are removed.
2. Feature Engineering:
   - Treatment effect estimates and confidence intervals are calculated for each study.
   - Additional features, such as pairwise odds ratios and confidence intervals, are derived from the dataset.
3. Missing Data Imputation:
   - Missing values in the engineered features are imputed using the Multiple Imputation by Chained Equations (MICE) technique, implemented through the IterativeImputer class from the scikit-learn library.
   - The imputation process is performed iteratively for a specified number of iterations (max_iter=10) to fill in missing values based on the observed data.
4. Data Normalisation:
   - The imputed features are normalised or standardised using the StandardScaler class from scikit-learn to ensure that all features have zero mean and unit variance.
5. Model Selection and Hyperparameter Tuning:
   - Multiple machine learning models, including linear regression, Lasso, Ridge, decision tree, random forest, and gradient boosting, are evaluated using grid search with cross-validation.
   - The best-performing model is selected based on the lowest cross-validated mean squared error (MSE).
   - Hyperparameter tuning is performed using grid search to find the optimal hyperparameters for the selected model.
6. Model Training and Evaluation:
   - The best-performing model is trained on the entire dataset using the imputed and scaled features.
   - The trained model's performance is evaluated using various metrics, including mean squared error (MSE), mean absolute error (MAE), and R-squared (R2).
   - Bootstrap resampling is employed to assess the model's performance and obtain confidence intervals for the evaluation metrics.
7. Treatment Effect Estimation:
   - The trained model is used to predict the treatment effects for each study in the dataset.
   - The predicted treatment effects, along with the observed effects and study information, are stored in a results DataFrame.
8. Visualisation and Interpretation:
   - The NMA results are visualised using a forest plot, displaying the predicted treatment effects and their confidence intervals for each study.
   - Treatment rankings are calculated based on the mean predicted effects and compared with the expected rankings from the Higgins 2012 study.
   - Inconsistency assessment is performed using the node-splitting approach to evaluate the consistency of the network meta-analysis results.

The architecture of this automated frequentist NMA pipeline combines data preprocessing, feature engineering, missing data imputation, model selection, and evaluation techniques to provide a research-oriented approach for comparing the effectiveness of smoking cessation interventions across multiple studies. However, the results should be interpreted with caution and not used directly for policy-making or healthcare decision-making without further validation and comparison with Bayesian methods, as recommended by the National Institute for Health and Care Excellence (NICE).

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
