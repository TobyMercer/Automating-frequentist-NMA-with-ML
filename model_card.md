# Model Card

## Model Description

**Input:**
- Study-level data from a network of studies comparing different smoking cessation interventions.
- Key input features include:
  - Sample size
  - Study population demographics
  - Intervention details (e.g., dose, duration)
  - Study design (e.g., randomized controlled trial, observational study)

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
- Performance is evaluated using cross-validation with different splitting strategies to ensure robustness and generalizability (refer to Cell 15.04).
- Model performance is assessed using bootstrap resampling (refer to Cell 16.01).
- Visualisations:
  - Network diagram showing the comparisons between interventions (refer to the output of Cell 3.01).
  - Forest plot of relative treatment effect estimates and 95% credible intervals for each pairwise comparison (refer to the output of Cell 3.02 and Cell 20.01).
  - Heatmap of treatment rankings (refer to the output of Cell 3.03).
  - Feature importance plot (refer to the output of Cell 16.03).

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
