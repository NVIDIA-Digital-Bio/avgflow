import os
import os.path as osp
import sys, pickle, json, compress_pickle
from typing import Callable, List, Optional

from operator import itemgetter

from rdkit import Chem
import numpy as np
import copy

from data_utils import get_atom_features, get_bond_features, get_laplacian_pe, one_hot_encoding
from functools import partial
from multiprocessing import Pool

def smiles2mol(smiles: str, add_hydrogen: bool = True) -> Chem.Mol:
    """
    Convert a SMILES string to a RDKiTmolecule object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if add_hydrogen:
        mol = Chem.AddHs(mol)
    return mol

def extract_conformers(mol: Chem.Mol, centering: bool = True) -> np.ndarray:
    """
    Extract the first conformer of a molecule.
    Args:
        mol: a RDKiT molecule object
        centering: whether to center the conformer
    Returns:
        pos: a numpy array of the coordinates of the atoms in the conformer
    """
    try:
        conf = mol.GetConformer(0)
    except:
        raise ValueError(f"Molecule has no conformers")
    pos = conf.GetPositions()
    if centering:
        pos = pos - np.mean(pos, axis=0)
    return pos

def mol2features(mol: Chem.Mol, dataset: str, lap_pe_dim: int = 32) -> dict:
    """
    Convert a molecule to a dictionary of data.
    Serveral steps:
        1. Convert the molecule to a graph (not fully-connected)
        2. Calculate the Laplacian eigenvector PE with the not fully-connected graph
        3. featurize atoms 
        4. one-hot encode bonds (now consider fully-connected graph, no-bond is also a bond type)
    Output: 
        data: dict {
            'node_feats': atom features,
            'edge_types': bond types,
            'senders': sender nodes,
            'receivers': receiver nodes,
            'pe': Laplacian eigenvector PE,
        }
    """
    assert dataset in ['qm9', 'drugs'], f"Feature extraction is only supported for qm9 or drugs style"
    # Get the atom features
    atom_feats = get_atom_features(mol, dataset)
    # Get the bond features, edge_attrs are one-hot encoded bond types (4 types in total)
    senders, receivers, edge_attr = get_bond_features(mol) 
    # Get the Laplacian eigenvector PE from the not fully-connected graph
    eig_vals, lap_pe = get_laplacian_pe(mol.GetNumAtoms(), senders, receivers, lap_pe_dim)

    # Build the fully-connected graph
    num_nodes = mol.GetNumAtoms()
    bond_matrix = np.zeros((num_nodes, num_nodes))
    bond_type_with_no_bond = np.argmax(edge_attr, axis=1) + 1
    bond_matrix[senders, receivers] = bond_type_with_no_bond

    rows, cols, edge_types = [], [], []
    for i in range(bond_matrix.shape[0]):
        for j in range(i + 1, bond_matrix.shape[1]):
            rows += [i, j]
            cols += [j, i]
            edge_types += 2 * [bond_matrix[i, j]]
    # Use fully-connected graph senders and receivers
    senders, receivers = np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)
    # one-hot encode bond types
    bond_attrs = np.array(edge_types).astype(int)
    bond_attrs = one_hot_encoding(bond_attrs, 5)

    
    return {
        'node_feats': atom_feats,
        'edge_types': bond_attrs,
        'senders': senders,
        'receivers': receivers,
        'pe': lap_pe,
    }

