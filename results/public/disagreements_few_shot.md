# Disagreements: few_shot

12 of 64 papers differ from my label.

## #3 Interpretability and fairness evaluation of DL models on MIMIC-IV
- My label: B. Model: A
- Checks (original model / clinical outcome / fairness in results): yes / yes / yes
- Model reason: Trains deep learning models for in-hospital mortality prediction on MIMIC-IV and reports fairness and bias results across demographic subgroups.

## #6 Peeking into a black box — fairness/generalizability of MIMIC-III benchmark
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): no / yes / yes
- Model reason: Audits an existing MIMIC benchmark model rather than developing an original prediction model as a contribution.

## #19 Improving Equity in DL Medical Applications, Gerchberg-Saxton Algorithm
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): unclear / no / no
- Model reason: Abstract discusses bias reduction in deep learning applications but does not clearly present an original predictive model for a clinical outcome with fairness results.

## #27 AEquity — bias detection/mitigation via subgroup learnability
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): no / yes / yes
- Model reason: Presents a bias-mitigation metric/strategy (AEquity) rather than developing an original clinical prediction model as a contribution.

## #32 Interpretable ML survival model, older adults w/ multimorbidity
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): yes / yes / no
- Model reason: Develops a survival model for late-life mortality, but mentions subgroup fairness only in passing without reporting actual fairness metric results across demographic groups.

## #38 Multi-condition (ICU mortality + psych readmission)
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): no / yes / yes
- Model reason: Applies an existing algorithm as a case study/audit to examine bias rather than developing or training a new model as a contribution.

## #40 Enhancing Multi-Attribute Fairness in Healthcare Predictive Modeling
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): yes / unclear / yes
- Model reason: Develops a multi-attribute fairness optimization approach and reports fairness results, but the abstract does not specify a concrete real patient-level clinical outcome.

## #42 Fairfedtransnet: Rare Disease
- My label: B. Model: A
- Checks (original model / clinical outcome / fairness in results): yes / yes / yes
- Model reason: Develops a transformer-based model for rare disease diagnosis on eICU data and reports fairness gaps between clients.

## #56 Fair ML with Structured and Unstructured Features
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): yes / yes / no
- Model reason: Develops a machine learning pipeline for patient mortality prediction with bias mitigation, but reports no subgroup fairness results in the abstract.

## #61 Cross-Modality Bias Auditing
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): no / yes / yes
- Model reason: The study frames its contribution as auditing and evaluating mitigation strategies on an existing framework rather than developing a new original predictive model as a core contribution.

## #62 Adaptive Disparity-Aware Loss
- My label: B. Model: A
- Checks (original model / clinical outcome / fairness in results): yes / yes / yes
- Model reason: Develops a federated multimodal model with an adaptive disparity-aware loss and reports fairness and disparity results across demographic groups.

## #65 Shining Light on Dark Skin, Pulse Oximetry
- My label: B. Model: A
- Checks (original model / clinical outcome / fairness in results): yes / yes / yes
- Model reason: Trains an XGBoost regression model to correct pulse oximetry inaccuracies and evaluates performance among Black patients.
