import pandas as pd
import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("tox21.csv")

# Drop rows with missing values
data = data.dropna()

# Convert SMILES to fingerprints
fingerprints = []

for smiles in data["smiles"]:
    mol = Chem.MolFromSmiles(smiles)
    
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        fingerprints.append(list(fp))
    else:
        fingerprints.append([0]*512)

X = np.array(fingerprints)
y = data["NR-AR"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model.fit(X_train, y_train)

# Save model
with open("toxicity_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as toxicity_model.pkl")