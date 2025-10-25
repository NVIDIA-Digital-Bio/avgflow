# Model Overview

## Description:  
AvgFlow as released in this repository is a diffusion transformer model with pairwise biased attention trained on 3-dimensional conformer data of small molecules. It can be used as a highly efficient molecular conformer generator.
 
_This model is ready for commercial use._  


### License/Terms of Use:   
GOVERNING TERMS: Use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). AvgFlow source code is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### Deployment Geography:  
Global

### Use Case:  
AvgFlow is a model for efficient molecular conformer generation. The model can be used in the pharmaceutical and chemical industries and acadamic research to generate the 3D conformers of small organic molecules for downstream tasks such as property prediction and virtual screening. 

### Release Date:   
**Github:** 10/15/2025 via https://github.com/NVIDIA/avgflow   

**NGC:** 10/15/2025 via https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/collections/avgflow

## Reference(s):
Research paper: "Efficient Molecular Conformer Generation with
SO(3)-Averaged Flow Matching and Reflow", https://arxiv.org/abs/2507.09785

## Model Architecture:   
**Architecture Type:** Transformer with pair-biased attention

**Network Architecture:** Diffusion Transformer architecture with pair-biased attention, adaptive layernorm, and gating. 

**Number of model parameters:** Two variants: 52 million and 64 million   

 
## Input:  
**Input Type(s):** Text and 3D atom coordinates<br>

**Input Format(s):** SMILES string and Cartesian coordinates<br>

**Input Parameters:** 1D (SMILES string) and 3D (coordinates)  

**Other Properties Related to Input:** Trained with maximum number of token (atom) as 200. 

## Output:  
**Output Type(s):** 3D atom coordinates 

**Output Format:** Cartesian coordinates  

**Output Parameters:** 3D

**Other Properties Related to Output:** None Applicable   
   

## Software Integration:  
**Runtime Engine(s):**  
* JAX

**Supported Hardware Microarchitecture Compatibility:**  
* NVIDIA Ampere  
* NVIDIA Ada

**[Preferred/Supported] Operating System(s):**  
* Linux  


## Model Version(s): 
AvgFlow v1.0

## Training, Testing, and Evaluation Datasets:  
The released model is trained on the public GEOM-Drugs (https://github.com/learningmatter-mit/geom)
 

## Training Dataset:
GEOM-Drugs
**Link:** https://github.com/learningmatter-mit/geom

**Data Modality:**  
* Molecule SMILES string
* Molecule 3D coordinates 
 
**Non-Audio, Image, Text Training Data Size (If Applicable):**  
* Approximately 300k molecules.   

**Data Collection Method by dataset:**  
* Semi-empirical quamtum mechanics calculation

**Labeling Method by dataset:**  
* Automated  

**Dataset License(s):** [MIT License](https://mit-license.org/)  

## Evaluation Dataset:
GEOM-Drugs

**Link:** https://github.com/learningmatter-mit/geom  

**Data Modality:**  
* Molecule SMILES string
* Molecule 3D coordinates 
 
**Non-Audio, Image, Text Training Data Size (If Applicable):**  
* Approximately 300k molecules.   

**Data Collection Method by dataset:**  
* Semi-empirical quamtum mechanics calculation

**Labeling Method by dataset:**  
* Automated  

**Dataset License(s):** [MIT License](https://mit-license.org/)  
  

## Inference: 
**Engine:** JAX

**Test Hardware** Ampere (NVIDIA A100) / Ada (Nvidia A5880/A6000)


## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards  None.  

Please report model quality, risk, security vulnerabilities or concerns https://developer.nvidia.com/support.   
