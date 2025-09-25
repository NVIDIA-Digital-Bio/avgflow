<h1 align="center"> Efficient Molecular Conformer Generation with SO(3) <em>Averaged Flow</em>-Matching and Reflow</h1>


This is the official code repository for the paper titled [Efficient Molecular Conformer Generation with SO(3) Averaged Flow-Matching and Reflow](https://arxiv.org/abs/2507.09785) (ICML 2025).

## Contribution
+ We propose SO(3)-*Averaged Flow*: A novel flow-matching objective that analytically compute the probability flow from noise to all rotations of the data. When the "correctness" of samples is rotational invariant (such as conformer generation), SO(3)-*Averaged Flow* improves training efficiency by eliminating the need of rotational data augmentation and further improves model performance.
+ We propose to use reflow+distillation to reduce the number of sampling steps of flow-based model for conformer generation and maintain high generation quality.
+ We provide a JAX implementation of the diffusion transformer with pairwise biased attention architecture. It is powerful and scalable for generative modeling of molecules. 
## Instalation

## Pretrained Checkpoint

## Training

## Sampling

## License
Copyright @ 2025, NVIDIA Corporation. All rights reserved.<br>
The source code is made available under Apache-2.0.<br>
The model weights are made available under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
## Citation
If you find this repository and our paper useful, please cite our work through:
```BibTex
@article{cao2025efficient,
  title   = {Efficient Molecular Conformer Generation with SO (3)-Averaged Flow Matching and Reflow},
  author  = {Cao, Zhonglin and Geiger, Mario and Costa, Allan Dos Santos and Reidenbach, Danny and Kreis, Karsten and Geffner, Tomas and Pellegrini, Franco and Zhou, Guoqing and Kucukbenli, Emine},
  journal = {arXiv preprint arXiv:2507.09785},
  year    = {2025}
}
```
## Disclaimer
This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.