import streamlit as st
import pickle
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="AI-Based Chemical Toxicity Predictor",
    layout="wide"
)


# ---------- HEADER ----------

st.markdown("""
<div style='text-align: center;'>
<h1 style='color:#4DA3FF;'>AI-Based Chemical Toxicity Predictor</h1>
<p style='font-size:18px; color:gray;'>
AI-based molecular toxicity and drug-likeness analysis tool
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div.stButton > button {
    background-color: #4DA3FF;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    height: 50px;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------

@st.cache_resource
def load_model():
    with open("toxicity_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()


# ---------- SESSION HISTORY ----------

if "history" not in st.session_state:
    st.session_state.history = []


# ---------- EXAMPLE MOLECULES ----------

examples = {
    "None": "",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
}

example_choice = st.selectbox("Example Molecule", list(examples.keys()))

smiles = st.text_input("Enter SMILES String", value=examples[example_choice])


# ---------- PREDICT BUTTON ----------

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3,2,3])

with col2:
    predict = st.button("Predict Toxicity", use_container_width=True)


# ---------- PREDICTION ----------

if predict:

    if smiles:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            st.error("Invalid SMILES string")

        else:

            # Morgan Fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            fp_array = np.array(fp).reshape(1, -1)

            prediction = model.predict(fp_array)[0]
            prob = model.predict_proba(fp_array)[0][1]

            # Molecular properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Molecule (SMILES)")
                st.code(smiles)

            with col2:
                st.subheader("Molecular Properties")
                st.write(f"Molecular Weight: {mw:.2f}")
                st.write(f"LogP: {logp:.2f}")
                st.write(f"H-Bond Donors: {hbd}")
                st.write(f"H-Bond Acceptors: {hba}")

            st.divider()


            # ---------- LIPINSKI RULE ----------

            st.subheader("Drug-Likeness (Lipinski Rule of 5)")

            lipinski = (
                mw < 500 and
                logp < 5 and
                hbd <= 5 and
                hba <= 10
            )

            if lipinski:
                st.success("Lipinski Compliant (Drug-Like)")
            else:
                st.warning("Not Drug-Like")

            st.divider()


            # ---------- PREDICTION RESULT ----------

            st.subheader("Toxicity Prediction")

            if prediction == 1:
                st.error("Prediction: Toxic")
                pred_label = "Toxic"
            else:
                st.success("Prediction: Non-Toxic")
                pred_label = "Non-Toxic"

            st.write(f"Toxicity Probability: {prob:.4f}")


            # ---------- RISK BAR ----------

            st.subheader("Toxicity Risk Level")

            st.progress(float(prob))

            if prob < 0.3:
                st.success("Low Toxicity Risk")
            elif prob < 0.6:
                st.warning("Moderate Toxicity Risk")
            else:
                st.error("High Toxicity Risk")

            st.divider()


            # ---------- APPLICATIONS ----------

            st.subheader("Applications")

            st.write("""
• Early drug discovery screening  
• Toxicity filtering of chemical compounds  
• Cheminformatics research  
• Molecular property analysis
""")


            # ---------- DOWNLOAD RESULT ----------

            result_text = f"""
SMILES,{smiles}
Prediction,{pred_label}
Probability,{prob:.4f}
Molecular Weight,{mw:.2f}
LogP,{logp:.2f}
H-Bond Donors,{hbd}
H-Bond Acceptors,{hba}
"""

            st.download_button(
                "Download Result",
                result_text,
                file_name="toxicity_result.csv"
            )


            # ---------- SAVE HISTORY ----------

            st.session_state.history.append({
                "SMILES": smiles,
                "Prediction": pred_label,
                "Probability": round(prob,4)
            })


# ---------- HISTORY ----------

if st.session_state.history:

    st.divider()

    st.subheader("Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    history_df["Probability"] = history_df["Probability"].astype(str)

    st.dataframe(history_df)


# ---------- FOOTER ----------

st.markdown("---")
st.markdown(
"Developed using **Python, RDKit, Machine Learning, and Streamlit**"
)