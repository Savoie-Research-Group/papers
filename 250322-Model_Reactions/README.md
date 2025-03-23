# Graphically-Defined Model Reactions are Extensible, Accurate, and Systematically Improvable

This directory contains data to reproduce main text result figures from the article. DOI: [10.26434/chemrxiv-2025-pp3xl](https://doi.org/10.26434/chemrxiv-2025-pp3xl)

## Author(s)

- Veerupaksh Singla | [GitHub](https://github.com/veerupaksh)
- Qiyuan Zhao | [GitHub](https://github.com/zhaoqy1996)
- Hsuan-Hao Hsu
- Brett M. Savoie (Corresponding Author) | [bsavoie2@nd.edu](mailto:bsavoie2@nd.edu) | [GitHub](https://github.com/Savoie-Research-Group)

## Directory tree *(description in parenthesis)*
```bash
.
├── fig1
│   ├── IRC_xyz  # (List of IRC xyz images)
│   ├── React_Prod_TS_xyz  # (List of Reactant, Product, and DFT optimized TS xyz (in that order))
│   └── DA_barrier.csv  # (Gibbs energies of activations)
├── fig3
│   ├── actual_reaction_React_Prod_TS_xyz  # (List of Reactant, Product, and DFT optimized TS xyz (in that order) for actual reaction)
│   ├── model_reaction_React_Prod_TS_xyz  # (List of Reactant, Product, and DFT optimized TS xyz (in that order) for model reaction)
│   └── YARP2_benchmark.csv  # (Gibbs energies of activations of actual reactions and their corresponding model reactions)
├── fig4
│   ├── different_scheme_results.txt  # (Gibbs energies of activations for all reactions across all sampling schemes)
│   ├── React_Prod_xyz  # (List of Reactant and Product xyz (in that order))
│   ├── sampling1_TS_xyz  # (DFT optimized TS xyz for sampling scheme 1)
│   ├── sampling2_TS_xyz  # (DFT optimized TS xyz for sampling scheme 2)
│   ├── sampling3_TS_xyz  # (DFT optimized TS xyz for sampling scheme 3)
│   └── sampling4_TS_xyz  # (DFT optimized TS xyz for sampling scheme 4)
├── fig5
│   ├── BEP
│   │   ├── alkanes.txt  # (List of alkanes used to generate degradation reactions)
│   │   ├── for_bep.csv  # (Detailed activation and reaction energy results for actual and model reactions for BEP)
│   │   ├── radicals.txt  # (List of radicals generated from alkanes)
│   │   ├── smiles_list.txt  # (Concatenated list of alkanes and radicals)
│   │   └── TCIT_Hf.json  # (reaction energies obtained from TCIT)
│   ├── MR_IRC_xyz  # (List of IRC xyz images for model reactions)
│   ├── MR_React_Prod_TS_xyz  # (List of Reactant, Product, and DFT optimized TS xyz (in that order) for model reactions)
│   └── model_reaction_barriers.csv  # (Gibbs energies of activations for model reactions)
└── README.md
```