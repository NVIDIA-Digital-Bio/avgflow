import os
import os.path as osp
import sys, pickle, json
import compress_pickle
from typing import Callable, List, Optional

import torch
from torch import Tensor
from tqdm import tqdm
from p_tqdm import p_map
from operator import itemgetter

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import one_hot, scatter
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType as BT
import jraph
import numpy as np
import copy

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from old_averaged_flow import graph_permutations, graph_permutations_Hmask
from functools import partial
from multiprocessing import Pool

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

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

def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]


def mol2Data(mol, smiles, dataset='qm9'):
    if dataset=='qm9':
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    elif dataset=='drugs':
        types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
                'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
                'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
                'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
    else:
        raise Exception("dataset type not supported")
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    N = mol.GetNumAtoms()

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

    z = torch.tensor(atomic_number, dtype=torch.long)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1, x2], dim=-1)


    data = Data(
        x=x,
        z=z,
        edge_index=edge_index,
        smiles=smiles,
        edge_attr=edge_attr,
    )
    return data

def fix_PE_shape(pe, k=4):
    if pe.shape[1]<k:
        pe = np.hstack([pe, np.zeros((pe.shape[0], k-pe.shape[1]))])
    elif pe.shape[1]>k:
        pe = pe[:, :k]
    return pe

def pyg2jraph(data: Data, fully_connected=True, laplacian_pe=True):
    if laplacian_pe:
        transform = AddLaplacianEigenvectorPE(k=min(data.x.shape[0]-1, 32), attr_name='laplacian_pe')
        data = transform(data)
        # try:
        #     transform = AddLaplacianEigenvectorPE(k=data.x.shape[0]-1, attr_name='laplacian_pe')
        #     data = transform(data)
        # except:
        #     print(data.x.shape, data.smiles)
        #     transform = AddLaplacianEigenvectorPE(k=32, attr_name='laplacian_pe')
        #     data = transform(data)
    node_feat = data.x.numpy()
    # node_feat = node_feat[:, :5]

    num_nodes = node_feat.shape[0]
    senders = data.edge_index[0, :].numpy()
    receivers = data.edge_index[1, :].numpy()
    bond_attrs = data.edge_attr.numpy()
    pe = data.laplacian_pe.numpy()
    # set maximum number of dimension of PE to 32 
    pe = fix_PE_shape(pe, k=32)

    MAX_PERMUTATIONS = 1024
    permutations = graph_permutations(
        node_feat, senders, receivers, bond_attrs, max_permutations=MAX_PERMUTATIONS
    )

    if fully_connected:
        bond_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(len(senders)):
            u, v = senders[i], receivers[i]
            bond_type = np.argmax(bond_attrs[i]) + 1
            bond_matrix[u, v] = bond_type
            bond_matrix[v, u] = bond_type
        rows, cols, edge_types = [], [], []
        for i in range(bond_matrix.shape[0]):
            for j in range(i + 1, bond_matrix.shape[1]):
                rows += [i, j]
                cols += [j, i]
                edge_types += 2 * [bond_matrix[i, j]]
        senders, receivers = np.array(rows), np.array(cols)
        bond_attrs = edge_types
    bond_attrs = np.array(bond_attrs).astype(int)

    # Padding and masking
    num_perm = len(permutations)
    permutations = np.concatenate(
        [permutations, np.zeros((MAX_PERMUTATIONS - num_perm, num_nodes), int)], axis=0
    )
    permutations_mask = np.concatenate(
        [np.ones(num_perm, int), np.zeros(MAX_PERMUTATIONS - num_perm, int)], axis=0
    )

    graph = jraph.GraphsTuple(
        nodes={
            "features": node_feat,
            "permutations": permutations.T,
            "laplacian_pe": pe,
        },
        edges={
            "features": np.eye(5)[bond_attrs],
        },
        senders=senders,
        receivers=receivers,
        globals={
            "permutations_mask": permutations_mask[None, :],
        },
        n_node=np.array([num_nodes]),
        n_edge=np.array([len(senders)]),
    )
    return graph, num_perm

def extract_conformers(molecule, dataset, topk=30):
    # datadir = '/home/zhonglinc/research/conformer/data/geom/%s/confs/'%dataset
    positions = []
    ranked = sorted(molecule['conformers'], key=itemgetter('boltzmannweight'), reverse=True)
    base = copy.deepcopy(ranked[0]['rd_mol'])
    raw_boltzmann = [ranked[0]['boltzmannweight']]
    if '.' in Chem.MolToSmiles(base):
        return None, None, None, base
    total_num_confs = 1
    for mol in ranked[1:]:
        rd_mol =mol['rd_mol']
        # Filter disconnected conformers
        if '.' in Chem.MolToSmiles(rd_mol):
            continue
        conf = rd_mol.GetConformer(0)
        try:
            base.AddConformer(conf, assignId=True)
        except RuntimeError:
            continue
        total_num_confs += 1
        raw_boltzmann.append(mol['boltzmannweight'])
        if total_num_confs>=topk:
            break
    for i in range(base.GetNumConformers()):
        conf = base.GetConformer(i)
        pos = conf.GetPositions()
        pos = pos - np.mean(pos, axis=0)
        positions.append(pos)
    raw_boltzmann = np.array(raw_boltzmann)
    scaled_boltzmann = raw_boltzmann / sum(raw_boltzmann)
    return positions, raw_boltzmann, scaled_boltzmann, base

def process_drug_smiles(pinput):
    smi, drugs_meta = pinput
    datadir = '/home/zhonglinc/research/conformer/data/geom/drugs/'
    try:
        picklepath = drugs_meta[smi]['pickle_path']
    except:
        print(smi)
        return
    with open(picklepath, "rb") as input_file:
        molecule = pickle.load(input_file)
    clean_smi = osp.basename(picklepath).replace('.pickle', '')
    mol = molecule['conformers'][0]['rd_mol']
    data = mol2Data(mol, smi, dataset='drugs')
    graph_tuple = pyg2jraph(data)
    positions, raw_boltzmann, scaled_boltzmann = extract_conformers(molecule, dataset='drugs', topk=30)
    # save graph_tuple
    pickle.dump(graph_tuple, open(datadir+'mols/%s.pkl'%clean_smi, "wb"))
    # save conformers
    np.savez(
        datadir+'confs/%s.npz'%clean_smi, 
        pos=positions, raw_boltzmann=raw_boltzmann, scaled_boltzmann=scaled_boltzmann
    )
    return

if __name__ == '__main__':
    # with open('summary_qm9.json') as json_file:
    #     qm9_meta = json.load(json_file)
    # qm9_smiles = list(qm9_meta.keys())

    with open('summary_drugs.json') as json_file:
        drugs_meta = json.load(json_file)
    drugs_smiles = list(drugs_meta.keys())


    # for smi in tqdm(qm9_smiles):
    #     datadir = '/home/zhonglinc/research/conformer/data/geom/qm9_correct/'
    #     try:
    #         picklepath = qm9_meta[smi]['pickle_path']
    #     except:
    #         print(smi)
    #         continue
    #     with open(picklepath, "rb") as input_file:
    #         molecule = pickle.load(input_file)
    #     # Seems like clean_smi just replaced '/' in the raw SMILES with '_' 
    #     clean_smi = osp.basename(picklepath).replace('.pickle', '')
    #     mol = molecule['conformers'][0]['rd_mol']
    #     data = mol2Data(mol, smi, dataset='qm9')
    #     graph_tuple, num_perm = pyg2jraph(data)
    #     positions, raw_boltzmann, scaled_boltzmann = extract_conformers(molecule, dataset='qm9', topk=30)
    #     # save graph_tuple
    #     # pickle.dump(graph_tuple, open(datadir+'mols/%s.pkl'%clean_smi, "wb"))
    #     compress_pickle.dump(graph_tuple, open(datadir+'mols/%s.pkl'%clean_smi, "wb"), compression='gzip')
    #     # save conformers
    #     np.savez(
    #         datadir+'confs/%s.npz'%clean_smi, 
    #         pos=positions, raw_boltzmann=raw_boltzmann, scaled_boltzmann=scaled_boltzmann
    #     )

    # drugs_out = p_map(process_drug_smiles, drugs_smiles, drugs_meta, num_cpus=8)


    pbar = tqdm(range(len(drugs_smiles)), dynamic_ncols=True)
    successful_preprocess = 0
    datadir = '/home/zhonglinc/research/conformer/data/geom/drugs_new/'
    os.makedirs(datadir+'mols/', exist_ok=True)
    os.makedirs(datadir+'confs/', exist_ok=True)
    for i in pbar:
        smi = drugs_smiles[i]
        try:
            picklepath = drugs_meta[smi]['pickle_path']
        except:
            print(smi)
            print('No pickle path')
            print('-'*50)
            continue
        with open(picklepath, "rb") as input_file:
            molecule = pickle.load(input_file)
        clean_smi = osp.basename(picklepath).replace('.pickle', '')
        positions, raw_boltzmann, scaled_boltzmann, base= extract_conformers(molecule, dataset='drugs', topk=30)
        if positions is None:
            print(smi)
            print(Chem.MolToSmiles(base))
            print('Failed to extract conformers')
            print('-'*50)
            continue
        data = mol2Data(base, smi, dataset='drugs')
        graph_tuple, num_perm = pyg2jraph(data)
        compress_pickle.dump(graph_tuple, open(datadir+'mols/%s.pkl'%clean_smi, "wb"), compression='gzip')
        # save conformers
        np.savez(
            datadir+'confs/%s.npz'%clean_smi, 
            pos=positions, raw_boltzmann=raw_boltzmann, scaled_boltzmann=scaled_boltzmann
        )
        pbar.set_description(f"Number of permutations: {num_perm}, Number of conformers: {len(raw_boltzmann)}")
        successful_preprocess += 1
    print(f"Successfully preprocessed {successful_preprocess}/{len(drugs_smiles)} drugs")