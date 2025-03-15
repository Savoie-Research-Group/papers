# More Bang for Your Bond: Small Molecule Kinetics as Predictors of Polymer Stability

This directory contains all the data, python scripts, pretrained models, and figures to reproduce the results from the article. DOI: [10.26434/chemrxiv-2025-4hlt8](https://doi.org/10.26434/chemrxiv-2025-4hlt8)


## Author(s)
- Veerupaksh Singla | [GitHub](https://github.com/veerupaksh)
- Brett M. Savoie (Corresponding Author) | [bsavoie2@nd.edu](mailto:bsavoie2@nd.edu) | [GitHub](https://github.com/Savoie-Research-Group)


## Installation
To set up RMG, and the RMG Database, clone the corresponding GitHub Repositories and follow the instructions. a: [github.com/ReactionMechanismGenerator/RMG-Py](https://github.com/ReactionMechanismGenerator/RMG-Py). b: [github.com/ReactionMechanismGenerator/RMG-database](https://github.com/ReactionMechanismGenerator/RMG-database).

It is recommended to use a [`Conda` environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and to [install `RDKit` via `Conda`](https://www.rdkit.org/docs/Install.html) rather than `PyPI`. Other python libraries used are listed in `python_requirements.txt` and may be installed through `PyPI`.


## Directory tree *(description in parenthesis)*

```bash
.
├── code
│   ├── data_ops_analysis_code  # (Analyze and plot data. All figures from the articles have been generated from this code.)
│   │   ├── alfabet_lowest_bde_bdfe_plots.py  # (Plot results for ALFABET model.)
│   │   ├── alkane_stab_score_data_plots.py  # (Plot results for alkane stability score model.)
│   │   ├── expt_small_molecule_data_plots.py  # (Plot HA & deg temp distibution for the experimental small molecule data.)
│   │   ├── rmg_active_iters_data_plots.py  # (Plot results for stability score models from actively learned literature kinetics.)
│   │   └── smi_list_chord_plots.py  # (Plot the connectivity chord diagrams.)
│   ├── ml_code  # (Train stability score models and/or generate all the results.)
│   │   ├── ml_utils.py  # (Utility functions used across the codebase.)
│   │   ├── predict_alfabet_lowest_bde_bdfe.py  # (Generate predictions using ALFABET and obtain accuracy results.)
│   │   ├── train_predict_alkane_stability_score_models.py  # (Train alkane stability score models, generate predictions and obtain accuracy results.)
│   │   └── train_predict_rmg_models.py  # (Train stability score models from actively learned literature kinetics, generate predictions and obtain accuracy results.)
│   └── rmg_acive_learning_data_gen_code  # (Generate computational small molecule thermal degradation half life data using active learning and RMG.)
│       ├── gen_rmg_hl_data_active_learn.py  # (Active sampling and RMG simulations to generate half life data. Automatically calls all other scripts in this directory.)
│       ├── gen_rmg_hl_input_from_template.py  # (Generate RMG input files from RMG input template. May be used for standalone testing. Called automatically by gen_rmg_hl_data_active_learn.py.)
│       ├── job_submit_template_slurm.sub  # (Job submit template for SLURM. Modify for your cluster.)
│       ├── read_rmg_logs.py  # (Parse RMG output log file to obtain half life data. Called by gen_rmg_hl_data_active_learn.py.)
│       ├── rmg_input_template.py  # (RMG input template.)
│       └── submit_rmg_hl_jobs.py  # (Submit jobs using SLURM. Mofidy for your cluster. Called by gen_rmg_hl_data_active_learn.py.)
├── data
│   ├── alkane_stab_score_paper_data  # (alkane stability score pretrained models and predicted data.)
│   │   ├── alk_smi_hl_dict_secs_hl_prune_till_c17_32421_vals.json  # (pyrolysis half life data (in seconds) for alkanes pulled from doi: 10.1039/D4DD00036F.)
│   │   ├── trained_model_till_c15  # (final state dicts of models trained using alkanes with upto 15 HA.)
│   │   │   ├── model_fold_<i>_best_state_dict.pth  # (final trained model state dict checkpoints for 10 folds, i.e. i in range (0, 10).)
│   │   │   └── model_fold_<i>_info.json  # (final trained model info for 10 folds, i.e. i in range (0, 10).)
│   │   └── trained_model_till_c17  # (final state dicts of models trained using alkanes with upto 17 HA.)
│   │       ├── model_fold_<i>_best_state_dict.pth
│   │       └── model_fold_<i>_info.json
│   ├── expt_polymer_decomp_temp_data  # (Experimental polymer decomposition data and corresponding predictions from different models.)
│   │   ├── expt_polymer_data_main.csv  # (Ground truth experimental data pulled from literature as described in the article.)
│   │   ├── polymer_abbr_expt_td_dict.json  # ({Polymer abbreviation: Td} generated from expt_polymer_data_main.csv)
│   │   ├── polymer_abbr_expt_tp_dict.json  # ({Polymer abbreviation: Tp} generated from expt_polymer_data_main.csv)
│   │   ├── polymer_abbr_expt_tp_td_mean_dict.json  # ({Polymer abbreviation: (Td+Tp)/2} generated from expt_polymer_data_main.csv)
│   │   ├── polymer_abbr_linear_dimer_smi_dict.json  # ({Polymer abbreviation: linear dimer SMILES} generated from expt_polymer_data_main.csv)
│   │   ├── polymer_abbr_linear_tetramer_smi_dict.json  # ({Polymer abbreviation: linear tetramer SMILES} generated from expt_polymer_data_main.csv)
│   │   ├── polymer_abbr_linear_trimer_smi_dict.json  # ({Polymer abbreviation: linear trimer SMILES} generated from expt_polymer_data_main.csv)
│   │   ├── alfabet_min_bde_bdfe_preds  # (Predictions using the ALFABET model.)
│   │   │   ├── dimer_smi_min_bde_dict.json  # (Using linear dimer as model input, min BDE prediction as stability surrogate.)
│   │   │   ├── dimer_smi_min_bdfe_dict.json  # (Using linear dimer as model input, min BDFE prediction as stability surrogate.)
│   │   │   ├── tetramer_smi_min_bde_dict.json
│   │   │   ├── tetramer_smi_min_bdfe_dict.json
│   │   │   ├── trimer_smi_min_bde_dict.json
│   │   │   └── trimer_smi_min_bdfe_dict.json
│   │   ├── dimer_preds_alkane_stab_score_models  # (Predictions for linear dimer using the alkane stability score model.)
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl  # (Predictions for all 10 folds using models trained with alkanes with upto 15 HA.)
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json  # (Pairwise accuracy dict using polymer Td as ground truth.)
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json # (Pairwise accuracy dict using polymer Tp as ground truth.)
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json # (Pairwise accuracy dict using polymer (Td+Tp)/2 as ground truth.)
│   │   │   └── tp_td_mean_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   ├── dimer_preds_rmg_active_learning_iterations_models  # (Predictions for linear dimer using the stability score models from actively learned literature kinetics.)
│   │   │   ├── k_fold_smi_preds_dict_iteration_<j>.pkl  # (Predictions for all 10 folds using models trained on cummulative data generated through the j-th active iteration; j in range (0, 10).)
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_<j>.json  # (Pairwise accuracy dicts using polymer Td as ground truth for all the models trained above. j in range (0, 10).)
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_<j>.json  # (Pairwise accuracy dicts using polymer Tp as ground truth for all the models trained above. j in range (0, 10).)
│   │   │   └── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_<j>.json  # (Pairwise accuracy dicts using polymer (Td+Tp)/2 as ground truth for all the models trained above. j in range (0, 10).)
│   │   ├── tetramer_preds_alkane_stab_score_models
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   └── tp_td_mean_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   ├── tetramer_preds_rmg_active_learning_iterations_models
│   │   │   ├── k_fold_smi_preds_dict_iteration_<j>.pkl
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_<j>.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_<j>.json
│   │   │   └── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_<j>.json
│   │   ├── trimer_preds_alkane_stab_score_models
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   └── tp_td_mean_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   └── trimer_preds_rmg_active_learning_iterations_models
│   │       ├── k_fold_smi_preds_dict_iteration_<j>.pkl
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_<j>.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_<j>.json
│   │       └── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_<j>.json
│   ├── expt_small_molecule_decomp_temp_data  # (Experimental small molecule decomposition data and corresponding predictions from different models.)
│   │   ├── smi_expt_decomp_temp_dict_chon_f_cl.json
│   │   ├── smi_alfabet_min_bde_dict.json
│   │   ├── smi_alfabet_min_bdfe_dict.json
│   │   ├── preds_alkane_stab_score_models  # (Predictions using the alkane stability score model.)
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl  # (Predictions for all 10 folds using models trained with alkanes with upto 15 HA.)
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl
│   │   │   ├── k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json  # (Pairwise accuracy dict.)
│   │   │   └── k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   └── preds_rmg_active_learning_iterations_models  # (Predictions for using the stability score models from actively learned literature kinetics.)
│   │       ├── k_fold_smi_preds_dict_iteration_<j>.pkl  # (Predictions for all 10 folds using models trained on cummulative data generated through the j-th active iteration; j in range (0, 10).)
│   │       └── k_fold_pairwise_accuracy_dict_iteration_<j>.json  # (Pairwise accuracy dicts for all the models trained above. j in range (0, 10).)
│   ├── plots  # (All the plots in both the min text and the SI of the article.)
│   │   ├── chord_plot_expt_small_molec_data.html
│   │   ├── chord_plot_rmg_active_learning_data.html
│   │   ├── expt_small_molec_distributions.pdf
│   │   ├── expt_small_molec_ha_decomp_temp_histograms.pdf
│   │   ├── polymer_category_accuracy_alfabet.pdf
│   │   ├── polymer_category_k_fold_accuracy_alkane_stab_score_model.pdf
│   │   ├── polymer_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf
│   │   ├── SI_dimer_category_accuracy_alfabet_bde.pdf
│   │   ├── SI_dimer_category_accuracy_alfabet_bdfe.pdf
│   │   ├── SI_dimer_category_k_fold_accuracy_alkane_stab_score_model_till_c15.pdf
│   │   ├── SI_dimer_category_k_fold_accuracy_alkane_stab_score_model_till_c17.pdf
│   │   ├── SI_dimer_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf
│   │   ├── SI_small_molec_category_accuracy_alfabet_bde.pdf
│   │   ├── SI_small_molec_category_accuracy_alfabet_bdfe.pdf
│   │   ├── SI_small_molec_category_k_fold_accuracy_alkane_stab_score_model_till_c15.pdf
│   │   ├── SI_small_molec_category_k_fold_accuracy_alkane_stab_score_model_till_c17.pdf
│   │   ├── SI_small_molec_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf
│   │   ├── SI_tetramer_category_accuracy_alfabet_bde.pdf
│   │   ├── SI_tetramer_category_accuracy_alfabet_bdfe.pdf
│   │   ├── SI_tetramer_category_k_fold_accuracy_alkane_stab_score_model_till_c15.pdf
│   │   ├── SI_tetramer_category_k_fold_accuracy_alkane_stab_score_model_till_c17.pdf
│   │   ├── SI_tetramer_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf
│   │   ├── SI_trimer_category_accuracy_alfabet_bde.pdf
│   │   ├── SI_trimer_category_accuracy_alfabet_bdfe.pdf
│   │   ├── SI_trimer_category_k_fold_accuracy_alkane_stab_score_model_till_c15.pdf
│   │   ├── SI_trimer_category_k_fold_accuracy_alkane_stab_score_model_till_c17.pdf
│   │   ├── SI_trimer_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf
│   │   ├── small_molec_category_accuracy_alfabet.pdf
│   │   ├── small_molec_category_k_fold_accuracy_alkane_stab_score_model.pdf
│   │   └── small_molec_category_k_fold_accuracy_rmg_active_learning_iterations_models.pdf
│   └── rmg_active_learning_data  # (Data used for and generated during active sampling literature kinetics using RMG.)
│       ├── active_iterations_smi_lists  # (Lists of SMILES of molecules sampled in each iteration. Iteration 0 was supplied manually to start the process.)
│       │   └── iteration_<j>_smi_list.txt  # (j in range (0, 10).)
│       ├── sampling_space_pubchem_chon_f_cl_max_15_ha_clean_acyclic.txt  # (Space of all molecule SMILES extracted from PubChem from which active sampling is done.)
│       ├── sampled_smi_list.txt  # (List of SMILES of all the molecules sampled across all active iterations.)
│       ├── sampled_smi_hl_dict.json  # (RMG half lives of all the molecules sampled across all active iterations.)
│       └── trained_models_active_iterations_cummulative  # (10-fold stability score models trained using cummulative data sampled across all iterations.)
│           └── iteration_<j>  # (Final state dicts of all 10-fold trained models using cummulative data through iteration j; j in range (0, 10).)
│               ├── model_fold_<i>_best_state_dict.pth  # (final trained model state dict checkpoints for 10 folds, i.e. i in range (0, 10).)
│               └── model_fold_<i>_info.json  # (final trained model info for 10 folds, i.e. i in range (0, 10).)
├── python_requirements.txt
└── README.md
```
