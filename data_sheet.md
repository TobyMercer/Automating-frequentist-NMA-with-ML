# Data Sheet

## Data

The data used in this analysis comes from a network meta-analysis of smoking cessation interventions, originally published in the following paper:

Higgins JP, Jackson D, Barrett JK, Lu G, Ades AE, White IR. Consistency and inconsistency in network meta-analysis: concepts and models for multi-arm studies. Res Synth Methods. 2012 Jun;3(2):98-110. doi: 10.1002/jrsm.1044. PMID: 26062084; PMCID: PMC4433772.

The data was obtained from Appendix A of the paper: Table A.1 Smoking data set. I copied and pasted it to a csv file, which I called smokingcessation.csv. I include the smokingcessation.csv file. It contains information from 24 trials comparing different smoking cessation interventions. The columns in the dataset are:

- Study: Study identifier 
- Design: Treatments compared in the study (A = no contact, B = self-help, C = individual counselling, D = group counselling)
- dA, nA, dB, nB, dC, nC, dD, nD: Number of observed events (d) and total number of participants (n) for each treatment arm
- hasA, hasB, hasC, hasD: Indicator variables for whether each treatment was included in the study

Some data preprocessing and imputation was performed in the provided analysis code to handle missing data. The specific steps are:

1. MICE (Multivariate Imputation by Chained Equations) was used to impute missing values in the dataset. The `IterativeImputer` from the `sklearn.impute` module was employed for this purpose.

2. After imputation, the imputed dataset was used for further analysis and modeling.

No other modifications were made to the original data. The full processed dataset used in the analysis is printed in the provided code output.

## Motivation for Dataset Creation

This dataset was compiled to illustrate statistical methods for assessing consistency and inconsistency in network meta-analysis with multi-arm trials. Network meta-analysis allows simultaneous comparison of multiple treatments by combining direct and indirect evidence. Checking the consistency of different sources of evidence is an important component. The smoking cessation data, which contains a number of multi-arm trials, provides a useful test case for the proposed methods.

## Dataset Composition

The dataset contains results from 24 trials evaluating smoking cessation interventions. The interventions compared were: 
A - no contact
B - self-help 
C - individual counselling
D - group counselling

The trials vary in size from 20 to 2138 participants. 1 trial compared all four interventions, 1 trial compared BCD, 3 compared AB, 14 compared AC, 1 compared AD, 1 compared BC, 1 compared BD, and 2 compared CD.

## Data Collection Process

The data was obtained from the published literature on randomized controlled trials of smoking cessation interventions. Trials were identified through searches of electronic databases and reference lists of relevant systematic reviews. Data on the number of participants who quit smoking and the total number randomized was extracted for each treatment arm. 

## Data Preprocessing

MICE (Multivariate Imputation by Chained Equations) was used to handle missing data in the dataset. The `IterativeImputer` from the `sklearn.impute` module was employed for this purpose. After imputation, the imputed dataset was used for further analysis and modeling. The data is otherwise as extracted from the original trial reports.

## Ethical Considerations

The data comes from published randomized controlled trials. The conduct of these trials would have been approved by the relevant institutional ethics boards and informed consent would have been obtained from participants. No individual participant data is included in this dataset.

## Funding Sources

Not applicable. This dataset makes use of previously published data.

## Existing Uses of the Dataset

The dataset was used in the research paper cited above to illustrate new statistical methods for network meta-analysis. To the best of my knowledge it has not been used in other publications.
