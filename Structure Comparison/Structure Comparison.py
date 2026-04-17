import tkinter as tk
from tkinter import filedialog
from Bio.PDB import PDBParser, PDBIO, Superimposer
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
import os


def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select TWO PDB files (1: Experimental, 2: Predicted)",
        filetypes=[("PDB files", "*.pdb")]
    )
    root.destroy()
    return file_paths


def get_ca_data(structure):
    """Returns CA atoms, pLDDT scores, and coords."""
    ca_atoms, plddt, coords, res_ids = [], [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atom = residue["CA"]
                    ca_atoms.append(atom)
                    plddt.append(atom.get_bfactor())
                    coords.append(atom.get_coord())
                    res_ids.append(residue.get_id()[1])
    return ca_atoms, np.array(plddt), np.array(coords), res_ids


def plot_correlation(plddt, distances):
    """Generates a scatter plot of pLDDT vs Alignment Distance."""
    plt.figure(figsize=(8, 6))
    plt.scatter(plddt, distances, alpha=0.5, color='purple', edgecolors='none')

    # Add a trend line
    if len(plddt) > 1:
        z = np.polyfit(plddt, distances, 1)
        p = np.poly1d(z)
        plt.plot(plddt, p(plddt), "r--", alpha=0.8, label="Trendline")

    plt.title("Correlation: Prediction Confidence vs. Geometric Error")
    plt.xlabel("pLDDT (Model Confidence)")
    plt.ylabel("Distance to Experimental ($Å$)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()


def main():
    files = select_files()
    if len(files) != 2: return

    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("Ref", files[0])
    mov_struct = parser.get_structure("Model", files[1])

    # Extract Data
    ref_ca, _, ref_coords, _ = get_ca_data(ref_struct)
    mov_ca, mov_plddt, mov_coords, _ = get_ca_data(mov_struct)

    # Truncate to match length for 1-to-1 comparison
    min_len = min(len(ref_ca), len(mov_ca))

    # 1. Align using Superimposer
    sup = Superimposer()
    sup.set_atoms(ref_ca[:min_len], mov_ca[:min_len])
    sup.apply(mov_struct.get_atoms())

    # 2. Calculate post-alignment per-residue distance
    # We update mov_coords because 'apply' moved the atoms in the object
    aligned_mov_coords = np.array([atom.get_coord() for atom in mov_ca[:min_len]])
    distances = np.linalg.norm(ref_coords[:min_len] - aligned_mov_coords, axis=1)

    print(f"Global RMSD: {sup.rms:.4f} Å")

    # 3. Correlation Plot
    plot_correlation(mov_plddt[:min_len], distances)

    # 4. Save visualization (using the stable py3Dmol approach)
    io = PDBIO()
    aln_path = "aligned_predicted.pdb"
    io.set_structure(mov_struct)
    io.save(aln_path)

    # ... (Add calls to generate_pymol_script or save_py3dmol_html here)


if __name__ == "__main__":
    main()