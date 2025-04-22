# Actual Reaction Generation and Benchmarking from Model Reactions

This directory contains data to reproduce main text result figures from the article. DOI: TBD on submission.

## Author(s)
- Veerupaksh Singla | [GitHub](https://github.com/veerupaksh)
- Ethan G. Sterbis
- Kyra L. Kapuscinski
- Brett M. Savoie (Corresponding Author) | [bsavoie2@nd.edu](mailto:bsavoie2@nd.edu) | [GitHub](https://github.com/Savoie-Research-Group)

## Directory tree *(description in parenthesis)*
```bash
.
├── analyses_and_plots  # (contains all the plots and tables in the article. generated with plotting scripts in the code/data_analysis_code directory)
├── code
│   ├── data_analysis_code
│   │   ├── make_activation_barrier_dicts.py  # (example script to generate activation barrier (free energies of activation) dictionaries)
│   │   ├── make_data_accuracy_plots.py  # (contains main script to generate data accuracy plots, fig 4)
│   │   ├── make_data_stats_plots.py  # (contains main script to generate data accuracy and statistics plots, fig 3)
│   │   ├── make_umap.py  # (contains main script to generate UMAP plots, fig 3)
│   │   ├── train_xgb_classifier_plot_confusion_matrix.py  # (contains main script to train XGBoost classifier, analyze results, and plot confusion matrix. table 1, fig 5)
│   │   └── utils.py  # (contains functions to generate plots, used in all the scripts in this directory)
│   └── data_gen_code
│       ├── data_gen_utils.py  # (contains functions to generate data, used in generate_actual_rxns_from_model_rxns.py)
│       ├── generate_actual_rxns_from_model_rxns.py  # (contains main script to generate actual reactions from model reactions)
│       └── yarp_utils  # (utilities extracted from YARP 2.0: https://github.com/zhaoqy1996/YARP/tree/254855d0381da647f8fe8a4da0da05bd4026b22a/version2.0)
├── data
│   ├── ar_atom_mapped_smi_dict.json  # (contains atom-mapped SMILES for actual reactions. AR index: AM SMILES)
│   ├── ar_ea_fwd_dict.json  # (contains forward activation barrier (free energy of activation) for actual reactions. AR index: EA fwd)
│   ├── ar_ea_rev_dict.json  # (contains reverse activation barrier (free energy of activation) for actual reactions. AR index: EA rev)
│   ├── ar_fn_groups_added_dict.json  # (contains functional groups added to actual reactions. AR index: FN groups added list)
│   ├── ar_geometries_unoptimized.tar.gz  # (contains unoptimized geometries (xyz files) for actual reactions)
│   ├── ar_mr_dict.json  # (contains mapping of actual reactions to model reactions. AR index: MR index)
│   ├── ar_smi_dict.json  # (contains SMILES for actual reactions. AR index: SMILES)
│   ├── ar_transition_state_energy_list.csv  # (contains transition state h,f,zpe (enthalpy, free energy, zero point energy) for actual reactions. csv header: ar index,h,f,spe)
│   ├── ar_transition_state_geometries_dft_optimized.tar.gz  # (contains optimized (B3LYP-D3/TZVP) transition state geometries (xyz files) for actual reactions)
│   ├── fn_groups_class_to_ref_nums_dict.json  # (contains functional group classes to reference numbers dictionary. fn groups class name: ref nums of functional groups)
│   ├── fn_groups_ref_nums_to_smi_dict.json  # (contains functional group reference numbers to SMILES dictionary. ref nums of functional groups: SMILES)
│   ├── mr_ar_list_dict.json  # (contains mapping of model reactions to actual reactions. MR index: AR indices list)
│   ├── mr_atom_mapped_smi_dict.json  # (contains atom-mapped SMILES for model reactions. MR index: AM SMILES)
│   ├── mr_ea_fwd_dict.json  # (contains forward activation barrier (free energy of activation) for model reactions. MR index: EA fwd)
│   ├── mr_ea_rev_dict.json  # (contains reverse activation barrier (free energy of activation) for model reactions. MR index: EA rev)
│   ├── mr_geometries_unoptimized.tar.gz  # (contains unoptimized geometries (xyz files) for model reactions)
│   ├── mr_smi_dict.json  # (contains SMILES for model reactions. MR index: SMILES)
│   ├── mr_transition_state_energy_list.csv  # (contains transition state h,f,zpe (enthalpy, free energy, zero point energy) for model reactions. csv header: mr index,h,f,spe)
│   ├── mr_transition_state_geometries_dft_optimized.tar.gz  # (contains optimized (B3LYP-D3/TZVP) transition state geometries (xyz files) for model reactions)
│   └── react_prod_smi_energy_list.csv  # (contains reactant and product SMILES and energies (h,f,zpe) for both model and actual reactions. csv header: smi,h,f,spe)
└── README.md
```
