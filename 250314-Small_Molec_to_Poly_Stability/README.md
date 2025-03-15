# More Bang for Your Bond: Small Molecule Kinetics as Predictors of Polymer Stability

This directory contains all the data, python scripts, pretrained models, and figures to reproduce the results from the article: DOI: [10.26434/chemrxiv-2025-4hlt8](https://doi.org/10.26434/chemrxiv-2025-4hlt8)


## Author(s)
- Veerupaksh Singla | [GitHub](https://github.com/veerupaksh)
- Brett M. Savoie (Corresponding Author) | [bsavoie2@nd.edu](mailto:bsavoie2@nd.edu) | [GitHub](https://github.com/Savoie-Research-Group)


## Installation
To set up RMG-Py, and the RMG-Py Database, clone the corresponding GitHub Repositories and follow the instructions: a. [github.com/ReactionMechanismGenerator/RMG-Py](https://github.com/ReactionMechanismGenerator/RMG-Py). b. [github.com/ReactionMechanismGenerator/RMG-database](https://github.com/ReactionMechanismGenerator/RMG-database).


## Directory tree description

```bash
.
├── code
│   ├── data_ops_analysis_code
│   │   ├── alfabet_lowest_bde_bdfe_plots.py
│   │   ├── alkane_stab_score_data_plots.py
│   │   ├── expt_small_molecule_data_plots.py
│   │   ├── rmg_active_iters_data_plots.py
│   │   └── smi_list_chord_plots.py
│   ├── ml_code
│   │   ├── ml_utils.py
│   │   ├── predict_alfabet_lowest_bde_bdfe.py
│   │   ├── train_predict_alkane_stability_score_models.py
│   │   └── train_predict_rmg_models.py
│   └── rmg_acive_learning_data_gen_code
│       ├── gen_rmg_hl_data_active_learn.py
│       ├── gen_rmg_hl_input_from_template.py
│       ├── job_submit_template_slurm.sub
│       ├── read_rmg_logs.py
│       ├── rmg_input_template.py
│       └── submit_rmg_hl_jobs.py
├── data
│   ├── alkane_stab_score_paper_data
│   │   ├── alk_smi_hl_dict_secs_hl_prune_till_c17_32421_vals.json
│   │   ├── trained_model_till_c15
│   │   │   ├── model_fold_0_best_state_dict.pth
│   │   │   ├── model_fold_0_info.json
│   │   │   ├── model_fold_1_best_state_dict.pth
│   │   │   ├── model_fold_1_info.json
│   │   │   ├── model_fold_2_best_state_dict.pth
│   │   │   ├── model_fold_2_info.json
│   │   │   ├── model_fold_3_best_state_dict.pth
│   │   │   ├── model_fold_3_info.json
│   │   │   ├── model_fold_4_best_state_dict.pth
│   │   │   ├── model_fold_4_info.json
│   │   │   ├── model_fold_5_best_state_dict.pth
│   │   │   ├── model_fold_5_info.json
│   │   │   ├── model_fold_6_best_state_dict.pth
│   │   │   ├── model_fold_6_info.json
│   │   │   ├── model_fold_7_best_state_dict.pth
│   │   │   ├── model_fold_7_info.json
│   │   │   ├── model_fold_8_best_state_dict.pth
│   │   │   ├── model_fold_8_info.json
│   │   │   ├── model_fold_9_best_state_dict.pth
│   │   │   └── model_fold_9_info.json
│   │   └── trained_model_till_c17
│   │       ├── model_fold_0_best_state_dict.pth
│   │       ├── model_fold_0_info.json
│   │       ├── model_fold_1_best_state_dict.pth
│   │       ├── model_fold_1_info.json
│   │       ├── model_fold_2_best_state_dict.pth
│   │       ├── model_fold_2_info.json
│   │       ├── model_fold_3_best_state_dict.pth
│   │       ├── model_fold_3_info.json
│   │       ├── model_fold_4_best_state_dict.pth
│   │       ├── model_fold_4_info.json
│   │       ├── model_fold_5_best_state_dict.pth
│   │       ├── model_fold_5_info.json
│   │       ├── model_fold_6_best_state_dict.pth
│   │       ├── model_fold_6_info.json
│   │       ├── model_fold_7_best_state_dict.pth
│   │       ├── model_fold_7_info.json
│   │       ├── model_fold_8_best_state_dict.pth
│   │       ├── model_fold_8_info.json
│   │       ├── model_fold_9_best_state_dict.pth
│   │       └── model_fold_9_info.json
│   ├── expt_polymer_decomp_temp_data
│   │   ├── alfabet_min_bde_bdfe_preds
│   │   │   ├── dimer_smi_min_bde_dict.json
│   │   │   ├── dimer_smi_min_bdfe_dict.json
│   │   │   ├── tetramer_smi_min_bde_dict.json
│   │   │   ├── tetramer_smi_min_bdfe_dict.json
│   │   │   ├── trimer_smi_min_bde_dict.json
│   │   │   └── trimer_smi_min_bdfe_dict.json
│   │   ├── dimer_preds_alkane_stab_score_models
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   └── tp_td_mean_k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   ├── dimer_preds_rmg_active_learning_iterations_models
│   │   │   ├── k_fold_smi_preds_dict_iteration_0.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_10.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_1.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_2.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_3.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_4.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_5.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_6.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_7.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_8.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_9.pkl
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_9.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_9.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │   │   └── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_9.json
│   │   ├── expt_polymer_data_main.csv
│   │   ├── polymer_abbr_expt_td_dict.json
│   │   ├── polymer_abbr_expt_tp_dict.json
│   │   ├── polymer_abbr_expt_tp_td_mean_dict.json
│   │   ├── polymer_abbr_linear_dimer_smi_dict.json
│   │   ├── polymer_abbr_linear_tetramer_smi_dict.json
│   │   ├── polymer_abbr_linear_trimer_smi_dict.json
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
│   │   │   ├── k_fold_smi_preds_dict_iteration_0.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_10.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_1.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_2.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_3.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_4.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_5.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_6.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_7.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_8.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_9.pkl
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │   │   ├── td_k_fold_pairwise_accuracy_dict_iteration_9.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │   │   ├── tp_k_fold_pairwise_accuracy_dict_iteration_9.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │   │   ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │   │   └── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_9.json
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
│   │       ├── k_fold_smi_preds_dict_iteration_0.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_10.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_1.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_2.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_3.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_4.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_5.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_6.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_7.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_8.pkl
│   │       ├── k_fold_smi_preds_dict_iteration_9.pkl
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │       ├── td_k_fold_pairwise_accuracy_dict_iteration_9.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │       ├── tp_k_fold_pairwise_accuracy_dict_iteration_9.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_0.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_10.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_1.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_2.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_3.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_4.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_5.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_6.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_7.json
│   │       ├── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_8.json
│   │       └── tp_td_mean_k_fold_pairwise_accuracy_dict_iteration_9.json
│   ├── expt_small_molecule_decomp_temp_data
│   │   ├── preds_alkane_stab_score_models
│   │   │   ├── k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c15.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_alkane_stab_score_model_till_c17.json
│   │   │   ├── k_fold_smi_preds_dict_alkane_stab_score_model_till_c15.pkl
│   │   │   └── k_fold_smi_preds_dict_alkane_stab_score_model_till_c17.pkl
│   │   ├── preds_rmg_active_learning_iterations_models
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_0.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_10.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_1.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_2.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_3.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_4.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_5.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_6.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_7.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_8.json
│   │   │   ├── k_fold_pairwise_accuracy_dict_iteration_9.json
│   │   │   ├── k_fold_smi_preds_dict_iteration_0.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_10.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_1.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_2.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_3.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_4.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_5.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_6.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_7.pkl
│   │   │   ├── k_fold_smi_preds_dict_iteration_8.pkl
│   │   │   └── k_fold_smi_preds_dict_iteration_9.pkl
│   │   ├── smi_alfabet_min_bde_dict.json
│   │   ├── smi_alfabet_min_bdfe_dict.json
│   │   └── smi_expt_decomp_temp_dict_chon_f_cl.json
│   ├── plots
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
│   └── rmg_active_learning_data
│       ├── active_iterations_smi_lists
│       │   ├── iteration_0_smi_list.txt
│       │   ├── iteration_10_smi_list.txt
│       │   ├── iteration_1_smi_list.txt
│       │   ├── iteration_2_smi_list.txt
│       │   ├── iteration_3_smi_list.txt
│       │   ├── iteration_4_smi_list.txt
│       │   ├── iteration_5_smi_list.txt
│       │   ├── iteration_6_smi_list.txt
│       │   ├── iteration_7_smi_list.txt
│       │   ├── iteration_8_smi_list.txt
│       │   └── iteration_9_smi_list.txt
│       ├── sampled_smi_hl_dict.json
│       ├── sampled_smi_list.txt
│       ├── sampling_space_pubchem_chon_f_cl_max_15_ha_clean_acyclic.txt
│       └── trained_models_active_iterations_cummulative
│           ├── iteration_0
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_1
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_10
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_2
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_3
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_4
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_5
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_6
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_7
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           ├── iteration_8
│           │   ├── model_fold_0_best_state_dict.pth
│           │   ├── model_fold_0_info.json
│           │   ├── model_fold_1_best_state_dict.pth
│           │   ├── model_fold_1_info.json
│           │   ├── model_fold_2_best_state_dict.pth
│           │   ├── model_fold_2_info.json
│           │   ├── model_fold_3_best_state_dict.pth
│           │   ├── model_fold_3_info.json
│           │   ├── model_fold_4_best_state_dict.pth
│           │   ├── model_fold_4_info.json
│           │   ├── model_fold_5_best_state_dict.pth
│           │   ├── model_fold_5_info.json
│           │   ├── model_fold_6_best_state_dict.pth
│           │   ├── model_fold_6_info.json
│           │   ├── model_fold_7_best_state_dict.pth
│           │   ├── model_fold_7_info.json
│           │   ├── model_fold_8_best_state_dict.pth
│           │   ├── model_fold_8_info.json
│           │   ├── model_fold_9_best_state_dict.pth
│           │   └── model_fold_9_info.json
│           └── iteration_9
│               ├── model_fold_0_best_state_dict.pth
│               ├── model_fold_0_info.json
│               ├── model_fold_1_best_state_dict.pth
│               ├── model_fold_1_info.json
│               ├── model_fold_2_best_state_dict.pth
│               ├── model_fold_2_info.json
│               ├── model_fold_3_best_state_dict.pth
│               ├── model_fold_3_info.json
│               ├── model_fold_4_best_state_dict.pth
│               ├── model_fold_4_info.json
│               ├── model_fold_5_best_state_dict.pth
│               ├── model_fold_5_info.json
│               ├── model_fold_6_best_state_dict.pth
│               ├── model_fold_6_info.json
│               ├── model_fold_7_best_state_dict.pth
│               ├── model_fold_7_info.json
│               ├── model_fold_8_best_state_dict.pth
│               ├── model_fold_8_info.json
│               ├── model_fold_9_best_state_dict.pth
│               └── model_fold_9_info.json
├── python_requirements.txt
└── README.md
```
