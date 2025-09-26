import jax.numpy as jnp
import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType as BT
from scipy.linalg import sqrtm
from scipy.sparse.linalg import eigs

chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}


def one_hot_encoding(x, num_classes):
    """
    One-hot encoding of a categorical variable.
    """
    return np.eye(num_classes)[x]

def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def get_atom_features(mol: Chem.Mol, dataset: str):
    """
    Get the atom features of a molecule.
    Part of the code is borrowed from Tor. Diff. repo.
    """
    if dataset=='qm9':
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    elif dataset=='drugs':
        types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
                'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
                'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
                'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
    else:
        raise Exception("dataset type not supported")

    type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    ring = mol.GetRingInfo()

    for i, atom in enumerate(mol.GetAtoms()):
        type_idx.append(types[atom.GetSymbol()])
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                            1 if atom.GetIsAromatic() else 0])
        atom_features.extend([atom.GetTotalNumHs(includeNeighbors=True)])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                            int(ring.IsAtomInRingOfSize(i, 4)),
                            int(ring.IsAtomInRingOfSize(i, 5)),
                            int(ring.IsAtomInRingOfSize(i, 6)),
                            int(ring.IsAtomInRingOfSize(i, 7)),
                            int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))
    
    N = mol.GetNumAtoms()
    x1 = one_hot_encoding(np.array(type_idx), num_classes=len(types)) # one-hot encoding of atom types
    x2 = np.array(atom_features).reshape(N, -1) # other atom features
    x = np.concatenate([x1, x2], axis=-1) # concatenate the two
    return x

def get_bond_features(mol: Chem.Mol):
    """
    Get the bond features of a molecule.
    We use one-hot encoding for bond types.
    Part of the code is borrowed from Tor. Diff. repo.
    """
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [bonds[bond.GetBondType()]]

    edge_index = np.array([rows, cols])
    senders = edge_index[0, :]
    receivers = edge_index[1, :]
    edge_type = np.array(edge_types)
    edge_attr = one_hot_encoding(edge_type, num_classes=len(bonds))
    return senders, receivers, edge_attr

def build_adj_matrix(
    num_nodes: int,
    s: np.ndarray,
    r: np.ndarray,
    weights: np.ndarray,
):
    adj = np.zeros((num_nodes, num_nodes))
    adj[s, r] = weights
    return adj

def add_self_loops(
    senders: np.ndarray,
    receivers: np.ndarray,
    weights: np.ndarray,
    num_nodes: int,
    fill_value: float=1.0,
):
    appendix = []
    for i in range(num_nodes):
        appendix.append(i)
    senders = np.concatenate([senders, appendix])
    receivers = np.concatenate([receivers, appendix])
    weights = np.concatenate([weights, fill_value * np.ones(num_nodes)])
    return senders, receivers, weights

def fix_PE_shape(pe, k=32):
    if pe.shape[1]<k:
        pe = np.hstack([pe, np.zeros((pe.shape[0], k-pe.shape[1]))])
    elif pe.shape[1]>k:
        pe = pe[:, :k]
    return pe

def get_laplacian_pe(
    num_nodes: int, 
    s: np.ndarray, 
    r: np.ndarray,
    k: int, # final laplacian_pe dimension
):  
    """
    Get the Laplacian positional encodings of a molecule.
    We use the symmetric normalization of laplacian.

    Part of the code is borrowed from torch_geometric.transforms.AddLaplacianEigenvectorPE

    Args:
        num_nodes: number of nodes in the molecule
        s: sender nodes
        r: receiver nodes
        k: final laplacian_pe dimension
    
    Returns:
        eig_vals: eigenvalues of the laplacian
        pe: positional encodings
    """
    SPARSE_THRESHOLD = 100
    n_lap = min(num_nodes-1, k) # for small molecules, we only use the first few eigenvectors
    edge_weight = np.ones_like(s, dtype=np.float32)
    A = build_adj_matrix(num_nodes, s, r, edge_weight) # adjacency matrix
    A = np.asarray(A, dtype=np.float32)
    D = np.diag(np.sum(A, axis=1)) # degree matrix

    # Symmetric normalization laplacian
    # L_sym = I - (D*)^(1/2) @ A @ (D*)^(1/2), D* is pinv(D)
    D_inv_sqrt = np.linalg.pinv(sqrtm(D))

    L = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    L = np.asmatrix(L, dtype=np.float32)
    L = scipy.sparse.coo_matrix(L)

    if num_nodes < SPARSE_THRESHOLD:
        L = L.todense()
        eig_vals, eig_vecs = np.linalg.eig(L)
    else:
        eig_vals, eig_vecs = eigs(
            L, k=n_lap+1, 
            which='SR', return_eigenvectors=True
        )
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = eig_vecs[:, 1:n_lap+1]
    pe = np.asarray(pe)
    sign = -1 + 2 * np.random.randint(0, 2, (n_lap, ))
    pe *= sign

    # pad or truncate to the final dimension
    pe = fix_PE_shape(pe, k)

    return eig_vals, pe

