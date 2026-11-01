# Disagreements: rules

9 of 64 papers differ from my label.

## #6 Peeking into a black box — fairness/generalizability of MIMIC-III benchmark
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): no / yes / yes
- Model reason: The paper performs an audit of an existing benchmarking model rather than developing a new one.

## #38 Multi-condition (ICU mortality + psych readmission)
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): no / yes / yes
- Model reason: The paper examines case studies rather than developing and claiming an original predictive model.

## #40 Enhancing Multi-Attribute Fairness in Healthcare Predictive Modeling
- My label: A. Model: needs_check
- Checks (original model / clinical outcome / fairness in results): yes / unclear / yes
- Model reason: The paper develops an original model and reports fairness results, but the abstract does not specify if the predicted target is a real patient-level clinical outcome.

## #43 FairnessAware GNN, ICU LOS
- My label: B. Model: needs_check
- Checks (original model / clinical outcome / fairness in results): yes / yes / unclear
- Model reason: The paper introduces a novel predictive model for a clinical outcome (length of stay), but it is unclear from the abstract whether fairness metrics are reported in the results as broken down by demographic subgroups.

## #53 When to Use Demographic Data, Bias-Responsible Approach
- My label: B. Model: needs_check
- Checks (original model / clinical outcome / fairness in results): unclear / yes / unclear
- Model reason: It is unclear from the abstract if a new predictive model was developed and if fairness metrics are reported as results.

## #56 Fair ML with Structured and Unstructured Features
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): yes / yes / no
- Model reason: Fairness or bias metrics are not reported as results in the abstract, meaning fairness_in_results is 'no'.

## #61 Cross-Modality Bias Auditing
- My label: A. Model: B
- Checks (original model / clinical outcome / fairness in results): no / yes / yes
- Model reason: The paper focuses on auditing an existing multimodal architecture rather than developing a new predictive model as a core contribution.

## #62 Adaptive Disparity-Aware Loss
- My label: B. Model: A
- Checks (original model / clinical outcome / fairness in results): yes / yes / yes
- Model reason: The paper develops a new predictive model (ADL-FMT) for healthcare decisions using benchmark datasets and reports fairness metrics across demographic groups in the results.

## #65 Shining Light on Dark Skin, Pulse Oximetry
- My label: B. Model: A
- Checks (original model / clinical outcome / fairness in results): yes / yes / yes
- Model reason: The study develops a predictive model for a clinical outcome (SaO2/hypoxemia) and reports fairness/performance metrics across racial subgroups.
