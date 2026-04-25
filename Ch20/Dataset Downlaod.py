import pubchempy as pcp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def get_aromatic_proportion(smiles):
    if not smiles or pd.isna(smiles):
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        aromatic_atoms = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol.GetNumAtoms())]
        heavy_atoms = mol.GetNumHeavyAtoms()
        return sum(aromatic_atoms) / heavy_atoms if heavy_atoms > 0 else 0
    return 0


def create_solubility_dataset(count=150):
    # Requesting multiple SMILES types to be safe
    props = ['IsomericSMILES', 'CanonicalSMILES', 'ConnectivitySMILES', 'XLogP', 'MolecularWeight',
             'RotatableBondCount']

    print(f"Downloading data for {count} molecules...")
    df = pcp.get_properties(props, list(range(1, count + 1)), as_dataframe=True)

    # Logic to find WHICH Smiles column we actually got
    smiles_col = None
    for col in ['CanonicalSMILES', 'ConnectivitySMILES', 'IsomericSMILES']:
        if col in df.columns:
            smiles_col = col
            break

    if not smiles_col:
        print("Error: No SMILES column found. Available columns:", df.columns.tolist())
        return None

    print(f"Using '{smiles_col}' for RDKit calculations...")

    # Calculate Aromatic Proportion
    df['AromaticProportion'] = df[smiles_col].apply(get_aromatic_proportion)

    # Handle XLogP (using RDKit MolLogP as fallback if XLogP is NaN)
    def fill_logp(row):
        if pd.isna(row.get('XLogP')):
            mol = Chem.MolFromSmiles(row[smiles_col])
            return Descriptors.MolLogP(mol) if mol else None
        return row['XLogP']

    df['Final_LogP'] = df.apply(fill_logp, axis=1)

    df.to_csv('solubility_data.csv', index=False)
    print("Success! Saved 150 rows to solubility_data.csv")
    return df


if __name__ == "__main__":
    dataset = create_solubility_dataset(150)