# Drug Properties Prediction Machine Learning Project

Machine learning system for predicting drug delivery properties using molecular descriptors.

## Author
**Author**: Nerissa Vu  
**Email**: vuthunga2002@gmail.com  
Chemical Engineering Student, Department of Chemical Engineering, Bucknell University

## Course Information
**Course**: Data Science for Chemical Engineering  
**Course Code**: CHEME 472 
**Instructor**: Prof. Jude Okolie 
**Quarter**: Fall 2024  
**Project Type**: Individual ML Project  

## Acknowledgments
This work builds upon the Therapeutic Data Commons (TDC) dataset and methodology.  

Special thanks to:
- The TDC team for providing the comprehensive drug discovery datasets
- The ChemE Data Science teaching team for their guidance
- The open-source community behind RDKit and other tools used in this project

## App Usage - Installation

Follow these steps to get [Your Project Name] up and running on your local machine:

### 1. Clone the repository

Clone the repository to your local machine using the following command:

    git clone https://github.com/nerissavu/ML_drug_discovery.git

### 2. Install required dependencies

Install the required Python libraries specified in `requirements.txt`:

    pip install -r requirements.txt

### 3. Download model folders
Download the model folder file from https://drive.google.com/drive/folders/1EQjgeVYYiPT_zpukC6HdqyVpEYPBD9Oo?usp=sharing

### 4. Start the application
run 'streamlit run app.py' on terminal

## Overview

### Target Variables - Properties

#### Regression Tasks

| Dataset Name | Property | Unit | Range | Reference |
|--------------|----------|------|--------|-----------|
| `Caco2_Wang` | Cell permeability | cm/s | Log scale | Wang et al. Advanced Drug Delivery Reviews 2016 |
| `Lipophilicity` | Fat solubility | Log ratio | -2 to 7 | AstraZeneca dataset, Experiments 1995-2015 |
| `Solubility` | Water solubility | Log mol/L | Continuous | AqSolDB - Sorkun et al. Scientific Data 2019 |

#### Classification Tasks

| Property | Description | Unit | Classes | Reference |
|----------|-------------|------|----------|-----------|
| `HIA_Hou` | Human intestinal absorption | % | Binary (>80% = 1) | Hou et al. J Chemical Information and Computer Sciences 2007 |
| `Bioavailability` | Drug availability | % | Binary (>50% = 1) | Ma et al. Journal of Chemical Information and Modeling 2008 |

**Data Source**: Therapeutic Data Commons (TDC)  
Huang, K., Fu, T., Glass, L.M. et al. "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development." Nat Commun 13, 3229 (2022)  
DOI: https://doi.org/10.1038/s41467-022-30887-3

### Input Features - Molecular Descriptors

Molecular descriptors calculated using RDKit:

| Feature | Description | Predicted Relevance |
|---------|-------------|-------------------|
| `MolWeight` | Molecular weight | Size and mass |
| `LogP` | Partition coefficient | Lipophilicity and permeability |
| `TPSA` | Topological Polar Surface Area | Absorption prediction |
| `NumRotatableBonds` | Count of rotatable bonds | N/A |
| `NumRings` | Total number of rings | N/A |
| `NumAromatic` | Count of aromatic rings | N/A |
| `NumHAcceptors` | H-bond acceptors | Drug-target interactions |
| `NumHDonors` | H-bond donors | Solubility and bioavailability |

## Model Performance

| Property | Train Metric | Test Metric |
|----------|--------------|-------------|
| Bioavailability | F1: 1.000 | F1: 0.626 |
| Cell permeability | R²: 0.940 | R²: 0.677 |
| Absorption | F1: 0.965 | F1: 0.901 |
| Lipophilicity | R²: 0.916 | R²: 0.422 |
| Solubility | R²: 0.931 | R²: 0.765 |

## Development Pipeline

### Data Analysis

- `eda.ipynb`: Feature engineering and property-specific EDA
  - Creates processed_data folder (property-specific CSV files)
  - Generates cleaned_drug_data.pkl (consolidated clean data)

### Model Development

- `train_model.py`: Initial model training and evaluation
- `train_model_cross_validation.py`: Cross-validation with three models
  - Results stored in train_model_cross_validation_result folder
  - Random forest showed best performance across all architectures
- `hyperparameter.py`: Hyperparameter optimization and metric storage
  - Results stored in rf_drug_analysis_results folder
- `interpretable_analysis.py`: Random forest feature interpretation
  - Results stored in rf_drug_analysis_interpretable_analysis folder
- `app.py`: Web interface for SMILES-based predictions

## Results Directory Structure

### A. train_model_cross_validation_result/

Contains comprehensive analysis using multiple models (Random Forest, Decision Tree, XGBoost)

1. **model_metrics/**
   - `{property}_metrics.csv` contains:
     - Model names
     - Training scores (R²/F1, RMSE/Precision)
     - Testing scores
     - Cross-validation statistics
     - Model-specific metrics

2. **model_plots/**
   - Feature Importance Plots (`{property}_feature_importance.png`)
     - Visual ranking of molecular descriptor influence
     - Horizontal bar charts of relative importance
   - Model Comparison Plot (`model_comparison.png`)
     - Boxplot comparing all three models
     - Train/test performance distribution
     - Color-coded: Blue (RF), Orange (DT), Green (XGBoost)

3. **stored_results/**
   - `{property}_results.json` contains:
     - Raw cross-validation results
     - Training/testing score arrays
     - Per-fold metrics

### B. rf_drug_analysis_results/final_results/

Random Forest model analysis results:

1. **Feature Importance Plots**
   - `{property}_feature_importance.png`
   - Visual ranking of descriptor influence
   - Horizontal bar format

2. **Metrics Files**
   - `{property}_metrics.json`
   - Performance statistics
   - Training/test metrics
   - Optimal hyperparameters

3. **Model Files**
   - `{property}_model.pkl`
   - Serialized Random Forest model
   - Complete preprocessing pipeline

### C. rf_drug_analysis_interpretable_analysis/final_results/

Extended analysis with SHAP interpretability:

1. **SHAP Plots**
   - `shap_global_{property}.png`
   - Global feature importance visualization
   - Feature interaction details

2. **SHAP Plot Structure**
   - Vertical feature ranking
   - SHAP value impact axis
   - Point-based visualization
   - Color gradient: Blue (low) to Red (high)

3. **Interpretation Guide**
   - Position indicates feature importance
   - Color + Position combinations:
     - Red right: High values increase prediction
     - Red left: High values decrease prediction
     - Blue shows inverse patterns
   - Spread indicates impact consistency
