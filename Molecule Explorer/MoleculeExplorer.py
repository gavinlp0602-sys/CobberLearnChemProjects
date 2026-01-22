import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import os
import webbrowser


def analyze_and_name_smiles(smiles_input):
    smiles_list = [s.strip() for s in smiles_input.split(',')]
    folder_name = "named_structures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Added 3 newlines to push output 2 lines below the input
    # Expanded header to include H-Donors and H-Acceptors
    print(f"\n\n\n{'Name':<18} | {'MolWt':<7} | {'LogP':<6} | {'H-Don':<5} | {'H-Acc':<5} | {'SMILES'}")
    print("-" * 95)

    for smiles in smiles_list:
        try:
            # 1. Fetch Name from PubChem
            results = pcp.get_compounds(smiles, namespace='smiles')
            if results and results[0].cid:
                c = results[0]
                name = c.synonyms[0] if c.synonyms else c.iupac_name
            else:
                name = "Unknown_Compound"

            # 2. Process with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Calculate all requested properties
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                h_donors = Descriptors.NumHDonors(mol)
                h_acceptors = Descriptors.NumHAcceptors(mol)

                # Format the name for the table
                display_name = name.split(';')[0][:17]

                # Print the data row
                print(f"{display_name:<18} | {mw:<7.2f} | {logp:<6.2f} | {h_donors:<5} | {h_acceptors:<5} | {smiles}")

                # 3. Generate and save image
                safe_name = name.split(';')[0].replace(' ', '_').lower()
                filename = f"{safe_name}.png"
                img_path = os.path.join(folder_name, filename)
                Draw.MolToFile(mol, img_path, size=(400, 400), legend=name.title())

                # Open the image
                webbrowser.open(os.path.abspath(img_path))
            else:
                print(f"\n\n\nError: '{smiles}' is not a valid SMILES string.")
        except Exception as e:
            print(f"\n\n\nError processing {smiles}: {e}")


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 70)
        user_input = input("Enter SMILES (or type 'exit' to stop): ").strip()

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting program...")
            break

        if not user_input:
            continue

        analyze_and_name_smiles(user_input)