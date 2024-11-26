# Data Science Ethics Checklist

[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

## A. Data Collection
 - **A.1 Informed consent**: If there are human subjects, have they given informed consent, where subjects affirmatively opt-in and have a clear understanding of the data uses to which they consent?
    - Source datasets (Ma, Wang, Hou, etc.) are from public repositories with appropriate licenses
    - Data usage terms are clearly documented
    - Future consideration: Verify redistribution rights for all datasets
 - **A.2 Collection bias**: Have we considered sources of bias that could be introduced during data collection and survey design and taken steps to mitigate those?
    - Chemical Space Coverage: Those dataset only contained commercially available drug small molecules compounds. Therefore, they might have underrepresentation of novel chemical scaffolds and limited coverage of large molecules and peptides
    - Publication Bias: Bias toward positive/successful results (high bioavailability,  low permeability, good absorption,.. ), underreporting of negative results, focus on "interesting" compounds
    - Experimental Condition Varied
 - **A.3 Limit PII exposure**: Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?
    - No personal information is collected or stored
    - User inputs (SMILES strings) are not stored
    - Prediction results are not linked to user sessions
 - **A.4 Downstream bias mitigation**: Have we considered ways to enable testing downstream results for biased outcomes (e.g., collecting data on protected group status like race or gender)?
    - In the future, we should include chemical diversity metrics and enable analysis of model performance across different structural groups

## B. Data Storage
 - **B.1 Data security**: Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?
    - The data used for this model is open source and publically available 
 - **B.2 Right to be forgotten**: Do we have a mechanism through which an individual can request their personal information be removed?
    - There are no user data is stored. Temporary calculations are cleared after predictions
 - **B.3 Data retention plan**: Is there a schedule or plan to delete the data after it is no longer needed?
    - Training data versions are archived with model versions

## C. Analysis
 - **C.1 Missing perspectives**: Have we sought to address blindspots in the analysis through engagement with relevant stakeholders (e.g., checking assumptions and discussing implications with affected communities and subject matter experts)?
    - We have not sought for any outsider perspective on this model. Future plan: Consult medicinal chemists, drug development researchers, computational chemistry experts for validation
 - **C.2 Dataset bias**: Have we examined the data for possible sources of bias and taken steps to mitigate or address these biases (e.g., stereotype perpetuation, confirmation bias, imbalanced classes, or omitted confounding variables)?
    - Future consideration: analyze distribution of chemical properties, check for overrepresented chemical scaffolds and evaluate experimental condition bias
 - **C.3 Honest representation**: Are our visualizations, summary statistics, and reports designed to honestly represent the underlying data?
    - We have clear visualization of predictions and transparent presentation of limitations
 - **C.4 Privacy in analysis**: Have we ensured that data with PII are not used or displayed unless necessary for the analysis?
    - The data used to train this model is publically available .We don't have any storage of proprietary chemical structures. We don't track of prediction patterns
 - **C.5 Auditability**: Is the process of generating the analysis well documented and reproducible if we discover issues in the future?
    - All of the step for training model is properly recorded with their metrics and evaluation

## D. Modeling
 - **D.1 Proxy discrimination**: Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?
    - The model is using molecules data. Therefore, there is no discrimination. We only use chemically relevant descriptor as the input 
 - **D.2 Fairness across groups**: Have we tested model results for fairness with respect to different affected groups (e.g., tested for disparate error rates)?
    - I have not test this model for a wide varity of API. Future plan: test performance across different chemical classes and evaluate prediction bias for various molecular sizes
 - **D.3 Metric selection**: Have we considered the effects of optimizing for our defined metrics and considered additional metrics?
    - The metrics I used for the model is R2, MAE, RMSE value for regression and F1 score and precision for classification. This have been documented in README file. In the future, I will validate  metric relevance with expert
 - **D.4 Explainability**: Can we explain in understandable terms a decision the model made in cases where a justification is needed?
    - I provided molecular descriptor/ properties explanations + feature importance analysis and documented model decision process
 - **D.5 Communicate bias**: Have we communicated the shortcomings, limitations, and biases of the model to relevant stakeholders in ways that can be generally understood?
    - We have create a disclaimer in the website for our model limitation. In the future after our analysis. We are hope to document known chemical space limitations.

## E. Deployment
 - **E.1 Redress**: Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?
    - I have a disclaimer stating all the limitation of the model. Future plan: I will have protocol for model updates based on feedback
 - **E.2 Roll back**: Is there a way to turn off or roll back the model in production if necessary?
    - There have only been 1 model created. But in the future, system for quick model switching will be created
 - **E.3 Concept drift**: Do we test and monitor for concept drift to ensure the model remains fair over time?
    -  Not yet. Monitor prediction accuracy over time and  tracking of chemical space coverage
 will be implemented
 - **E.4 Unintended use**: Have we taken steps to identify and prevent unintended uses and abuse of the model and do we have a plan to monitor these once the model is deployed?
    - We have clear disclaimers about intended use
    - Warnings against medical decision-making


*Data Science Ethics Checklist generated with [deon](http://deon.drivendata.org).*