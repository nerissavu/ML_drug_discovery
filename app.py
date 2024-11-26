import streamlit as st
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import matplotlib.pyplot as plt
import seaborn as sns

class DrugPredictor:
    def __init__(self, model_dir='rf_drug_analysis_results/final_results'):
        self.model_dir = model_dir
        self.models = self.load_models()
        
    def load_models(self):
        models = {}
        properties = ['Bioavailability_Ma', 'Caco2_Wang', 'HIA_Hou', 
                     'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB']
        
        for prop in properties:
            try:
                with open(f'{self.model_dir}/{prop}_model.pkl', 'rb') as f:
                    models[prop] = pickle.load(f)
            except FileNotFoundError:
                st.error(f"Model for {prop} not found")
        return models
    
    def calculate_descriptors(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Only include the 8 basic descriptors used in training
            features = {
                'MolWeight': Descriptors.ExactMolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRings': Descriptors.RingCount(mol),
                'NumAromatic': sum(1 for ring in mol.GetRingInfo().AtomRings() 
                                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
            }
                
            return pd.DataFrame([features])
        except Exception as e:
            st.error(f"Error calculating descriptors: {str(e)}")
            return None

    def predict_properties(self, smiles):
        descriptors = self.calculate_descriptors(smiles)
        if descriptors is None:
            return None
        
        predictions = {}
        for prop, model in self.models.items():
            try:
                predictions[prop] = model.predict(descriptors)[0]
            except Exception as e:
                st.error(f"Error predicting {prop}: {str(e)}")
        
        return predictions

def display_property_details(predictions):
    st.subheader("Property Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Binary Properties:")
        if 'Bioavailability_Ma' in predictions:
            bio_status = "High" if predictions['Bioavailability_Ma'] == 1 else "Low"
            st.metric("Bioavailability", bio_status)
        
        if 'HIA_Hou' in predictions:
            hia_status = "High" if predictions['HIA_Hou'] == 1 else "Low"
            st.metric("Human Intestinal Absorption", hia_status)
    
    with col2:
        st.write("Numerical Properties:")
        if 'Solubility_AqSolDB' in predictions:
            st.metric("Solubility (log mg/mL)", f"{predictions['Solubility_AqSolDB']:.2f}")
        
        if 'Caco2_Wang' in predictions:
            st.metric("Cell Permeability (cm/s)", f"{predictions['Caco2_Wang']:.2e}")
        
        if 'Lipophilicity_AstraZeneca' in predictions:
            st.metric("Lipophilicity (LogP)", f"{predictions['Lipophilicity_AstraZeneca']:.2f}")

def show_info_page():
    st.title("About Drug Property Predictor")
    
    # Disclaimer Section
    st.header("⚠️ Important Disclaimer")
    st.error("""
    - This tool is for research and educational purposes only.
    - Predictions are based on machine learning models and should not be used as the sole basis for any medical or drug development decisions.
    - Always consult with qualified professionals and perform proper experimental validation.
    - The accuracy and reliability of predictions may vary depending on the input structure's similarity to the training data.
    """)
    
    # Properties Section
    st.header("🔬 Predicted Properties")

    with st.expander("Bioavailability (Ma)", expanded=True):
        st.write("""
        **Description:** Binary classification of oral bioavailability based on Ma's rules.
        
        **Importance:**
        - Indicates the fraction of an administered drug that reaches systemic circulation
        - Critical for determining oral drug effectiveness
        - Influences dosing requirements and drug efficacy
        
        **Interpretation:**
        - High (1): ≥20% bioavailability 
        - Low (0): <20% bioavailability
        - Model outputs probability score between 0-1
        """)

    with st.expander("Caco-2 Permeability (Wang)", expanded=True):
        st.write("""
        **Description:** Numerical prediction of Caco-2 cell permeability in log scale.
        
        **Importance:**
        - Measures the ability of a drug to cross intestinal epithelial cells
        - Key indicator for oral drug absorption potential
        - Used in early-stage drug development screening
        
        **Interpretation:**
        - Values are in log cm/s
        - Higher values indicate better membrane permeability
        - Typical range: -8.0 to -4.0 log cm/s
        - >-5.15 log cm/s: High permeability
        - <-5.15 log cm/s: Low permeability
        """)

    with st.expander("Human Intestinal Absorption (HIA)", expanded=True):
        st.write("""
        **Description:** Binary classification based on Hou's model.
        
        **Importance:**
        - Predicts the extent of drug absorption in the human intestine
        - Essential for oral drug delivery assessment
        - Influences bioavailability and drug effectiveness
        
        **Interpretation:**
        - High (1): ≥30% absorption
        - Low (0): <30% absorption
        - Model outputs probability score between 0-1
        """)

    with st.expander("Lipophilicity (AstraZeneca)", expanded=True):
        st.write("""
        **Description:** Numerical prediction of LogP value.
        
        **Importance:**
        - Measures the drug's ability to dissolve in fats, oils, and non-polar solvents
        - Affects drug absorption, distribution, and membrane penetration
        - Key factor in drug-likeness assessment
        
        **Interpretation:**
        - LogP scale: Typically -2 to 5
        - Optimal range for oral drugs: 0 to 5
        - <0: Low lipophilicity
        - 0-5: Moderate to high lipophilicity
        - >5: Very high lipophilicity (potential issues)
        """)

    with st.expander("Solubility (AqSolDB)", expanded=True):
        st.write("""
        **Description:** Numerical prediction of aqueous solubility.
        
        **Importance:**
        - Indicates how well the drug dissolves in water
        - Critical for drug absorption and bioavailability
        - Influences formulation strategies
        
        **Interpretation:**
        - Values in log mg/mL
        - Typical range: -12 to 2 log mg/mL
        - >-4 log mg/mL: Acceptable solubility
        - <-4 log mg/mL: Poor solubility
        """)
    
    # Molecular Descriptors Section
    st.header("🧬 Molecular Descriptors Used")
    
    descriptors_df = {
        "Descriptor": ["Molecular Weight", "LogP", "Number of Rotatable Bonds", 
                      "Number of H-Bond Acceptors", "Number of H-Bond Donors",
                      "Topological Polar Surface Area (TPSA)", "Number of Rings",
                      "Number of Aromatic Rings"],
        "Description": [
            "Mass of the molecule",
            "Octanol-water partition coefficient",
            "Number of bonds that can rotate freely",
            "Count of H-bond accepting atoms",
            "Count of H-bond donating atoms",
            "Surface area of all polar atoms",
            "Total number of rings in the molecule",
            "Number of aromatic rings in the molecule"
        ],
        "Relevance": [
            "Affects absorption and drug-likeness",
            "Indicates fat solubility and membrane penetration",
            "Influences molecular flexibility and binding",
            "Important for protein binding and solubility",
            "Affects solubility and membrane permeation",
            "Predicts drug absorption and penetration",
            "Related to molecular complexity",
            "Influences protein binding and metabolism"
        ]
    }
    
    st.dataframe(pd.DataFrame(descriptors_df), use_container_width=True)
    
    # Model Information
    st.header("🤖 Model Information")
    st.write("""
    This tool uses Random Forest models trained on various datasets:
    - Bioavailability: Based on Ma's dataset of oral bioavailability
    - Caco-2: Wang's permeability dataset
    - HIA: Hou's human intestinal absorption dataset
    - Lipophilicity: AstraZeneca experimental logP measurements
    - Solubility: AqSolDB aqueous solubility database
    
    The models use molecular descriptors calculated using RDKit and were trained using standard machine learning practices including cross-validation and hyperparameter optimization.
    """)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Property Predictor", "Information"])
    
    if page == "Property Predictor":
        st.title("Drug Property Predictor")
        predictor = DrugPredictor()
        
        example_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        smiles = st.text_input("Enter SMILES string:", example_smiles)
        
        if st.button("Predict Properties"):
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    st.error("Invalid SMILES string")
                else:
                    predictions = predictor.predict_properties(smiles)
                    if predictions:
                        descriptors = predictor.calculate_descriptors(smiles)
                        if descriptors is not None:
                            st.subheader("Molecular Descriptors")
                            st.dataframe(descriptors)
                        display_property_details(predictions)
    else:
        show_info_page()

if __name__ == "__main__":
    main()