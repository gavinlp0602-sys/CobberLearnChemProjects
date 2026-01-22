# Code asks for compounds and then returns with formula, mass, smiles string, and an image of the 2d structure

import pubchempy as pcp
import urllib.request
import os
import webbrowser


def get_chemical_info(compounds):
    # Ensure compounds is always a list
    compound_list = [compounds] if isinstance(compounds, str) else compounds

    # Create a directory for images if it doesn't exist
    folder_name = "structure_images"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Updated header to include SMILES
    print(f"{'Name':<12} | {'Formula':<10} | {'Weight':<8} | {'SMILES'}")
    print("-" * 80)

    for name in compound_list:
        try:
            results = pcp.get_compounds(name, 'name')
            if not results:
                print(f"Result for '{name}': Not Found")
                continue

            c = results[0]

            # Using isomeric_smiles as it is the most standard structural string
            smiles = c.isomeric_smiles if c.isomeric_smiles else "N/A"

            # Print the data table
            print(f"{name.capitalize():<12} | {c.molecular_formula:<10} | {c.molecular_weight:<8} | {smiles}")

            # 1. Download the image into the folder
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{c.cid}/PNG"
            filename = os.path.join(folder_name, f"{name.lower()}_structure.png")
            urllib.request.urlretrieve(img_url, filename)

            # 2. Open the image
            webbrowser.open(os.path.abspath(filename))

        except Exception as e:
            print(f"Error fetching '{name}': {e}")


if __name__ == "__main__":
    user_input = input("Enter compound(s) separated by commas: ")
    processed_input = [item.strip() for item in user_input.split(',')]

    get_chemical_info(processed_input)