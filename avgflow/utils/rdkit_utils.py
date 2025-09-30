# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rdkit import Chem
from rdkit.Geometry import Point3D
import copy
import numpy as np


def add_conformer_to_mol(molecule, atom_coordinates):
    atom_coordinates = np.array(atom_coordinates, dtype=np.double)
    mol = copy.deepcopy(molecule)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x,y,z = atom_coordinates[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    mol.AddConformer(conf)
    return mol

def mirror_3d_mol(molecule):
    mol = copy.deepcopy(molecule)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(-x,-y,-z))
    return mol

def post_hoc_flip_mol(gt_smis, gen_mol):
    """
    Post-hoc flip the generated conformers if SMILES does not match ground truth SMILES.
    Warning: For the gt_smis, make sure to use the canonical SMILES without hydrogen.
    """
    # Make a copy of the generated molecule
    mol = copy.deepcopy(gen_mol)
    # Assign atom chiral tags
    Chem.AssignAtomChiralTagsFromStructure(mol)
    # Get the SMILES of the generated molecule
    gen_smis = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    # If the SMILES does not match the ground truth SMILES, flip the molecule
    if gen_smis != gt_smis:
        mol = mirror_3d_mol(mol)
    return mol  

def SMILES_to_safeSMILES(smiles: str) -> str:
    """
    Replace '/' with '_' in the SMILES string to avoid file name errors.
    """
    assert '.' not in smiles, 'Does not support SMILES string contains "."'
    return smiles.replace('/', '_')

def safeSMILES_to_SMILES(safe_smiles: str) -> str:
    """
    Replace '_' with '/' in the SMILES string to avoid file name errors.
    """
    assert '.' not in safe_smiles, 'Does not support SMILES string contains "."'
    return safe_smiles.replace('_', '/')