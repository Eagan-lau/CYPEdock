import os
from Bio.PDB import PDBParser
import numpy as np

# File paths
protein_dir = '/media/yugengliu/DATA1/erbeiti_pocket/pdbbind2020/PDBbind'
pocket_dir = '/media/yugengliu/DATA1/erbeiti_pocket/pdbbind2020/pocket'
# tokens_dir = '/media/yugengliu/Computer2/zzy/pocket-prediction/data/PDBbind/train/tokens'
labels_dir = '/media/yugengliu/Computer2/zzy/pocket-prediction/data/PDBbind/train/labels'

threshold = 1.0  # Distance threshold
parser = PDBParser()


# Compute residue centers
def get_residue_centers(structure):
    residue_centers = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    residue_centers[(chain.id, residue.id)] = residue["CA"].coord
    return residue_centers


# Compute residue labels
def get_residue_labels(protein_residues, pocket_residues, threshold):
    labels = {}
    for residue_id, coord in protein_residues.items():
        is_in_pocket = False
        for pocket_coord in pocket_residues.values():
            distance = np.linalg.norm(coord - pocket_coord)
            if distance <= threshold:
                is_in_pocket = True
                break
        labels[residue_id] = 1 if is_in_pocket else 0
    return labels


# Iterate over protein and pocket directories
for protein_file in os.listdir(protein_dir):
    if not protein_file.endswith(".pdb"):
        continue

    # if not protein_file.startswith("12gs"):
    #     continue

    # Derive the pocket path from the protein filename
    pocket_file = protein_file.replace("_protein", "_pocket")
    token_file = protein_file.replace(".pdb", "-tokens.npy")
    protein_path = os.path.join(protein_dir, protein_file)
    pocket_path = os.path.join(pocket_dir, pocket_file)
    # token_path = os.path.join(tokens_dir, token_file)

    # Verify that the corresponding pocket file exists
    if not os.path.exists(pocket_path):
        print(f"Pocket file for {protein_file} not found. Skipping.")
        continue

    # if not os.path.exists(token_path):
    #     print(f"token file for {token_file} not found. Skipping.")
    #     continue

    # Parse protein and pocket structures
    protein_structure = parser.get_structure("protein", protein_path)
    pocket_structure = parser.get_structure("pocket", pocket_path)

    # Compute residue-center coordinates
    protein_residues = get_residue_centers(protein_structure)
    pocket_residues = get_residue_centers(pocket_structure)

    # Compute labels
    labels = get_residue_labels(protein_residues, pocket_residues, threshold)

    # Convert the label dictionary to an ordered array
    sorted_keys = sorted(labels.keys())
    sorted_labels = [labels[key] for key in sorted_keys]
    labels_array = np.array(sorted_labels)

    # Save as a .npy file
    label_filename = protein_file.replace("_protein.pdb", "_labels.npy")
    label_path = os.path.join(labels_dir, label_filename)
    np.save(label_path, labels_array)
    print(f"Labels saved for {protein_file} as {label_filename}.")
