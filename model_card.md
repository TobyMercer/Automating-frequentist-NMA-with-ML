# Model Card

## Model Description

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
- The analysis uses an ensemble of multiple machine learning algorithms, including Linear Regression, Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, SVR, KNeighbors, and Gaussian Proces.
- The models are implemented in Python using scikit-learn and TensorFlow libraries.
- Data preprocessing steps:
  - Handling missing data with iterative imputation (refer to Cell 18.04)
  - Feature scaling using standardisation
  - One-hot encoding of categorical variables
- Hyperparameters are optimised using grid search cross-validation (refer to Cell 9.01 and Cell 15.01).

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

- Potential biases in the training data, such as publication bias or lack of representation of certain patient subgroups.
- Assumes transitivity (i.e., consistency of intervention effects across different study designs and populations), which may not always hold.
- Limited to study-level covariates and may not capture all relevant effect modifiers.
- Relies on aggregate study-level data rather than individual patient data, which may lead to ecological bias.

## Trade-offs

- Ensemble modeling approach improves predictive performance but increases computational complexity and may reduce interpretability compared to simpler models.
- Inclusion of a wide range of studies increases generalizability but may also introduce heterogeneity and potential biases.
- Use of Bayesian methods allows for incorporation of prior knowledge and uncertainty quantification but requires careful specification of prior distributions.
- Model provides rankings of interventions but should not be used to make definitive conclusions about effectiveness without considering uncertainty and limitations.

## Intended Use

- Intended for research purposes to compare the effectiveness of smoking cessation interventions across multiple studies.
- Not intended for making individual-level treatment recommendations or causal inferences without considering study design and potential confounding.
- Intended users include researchers, policymakers, and healthcare decision-makers interested in summarizing evidence on smoking cessation interventions.

## Ethical Considerations

- Potential for model misuse if rankings are interpreted as definitive conclusions about intervention effectiveness without considering uncertainty and limitations.
- Model development process included outlier detection and sensitivity analyses to mitigate potential biases.
- Transparency about model limitations and assumptions is crucial for responsible use.

## Caveats and Recommendations

- Model rankings are estimates with uncertainty and should be interpreted cautiously.
- Users should consider the credible intervals and measures of inconsistency/heterogeneity when interpreting the rankings.
- Results should be used in conjunction with other sources of evidence, such as expert judgment and mechanistic understanding.
- Limitations of relying on aggregate study-level data should be considered when applying the model.
