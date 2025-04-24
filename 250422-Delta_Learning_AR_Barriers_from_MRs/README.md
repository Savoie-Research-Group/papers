# Delta Learning Large Actual Reaction Barriers from Model Reactions

This directory contains data to reproduce main text result figures from the article. DOI: TBD on submission.

## Author(s)
- Veerupaksh Singla | [GitHub](https://github.com/veerupaksh)
- Brett M. Savoie (Corresponding Author) | [bsavoie2@nd.edu](mailto:bsavoie2@nd.edu) | [GitHub](https://github.com/Savoie-Research-Group)

## Directory tree *(description in parenthesis)*
```bash
.
├── analyses_and_plots  # (plots from `code/analyses_code`)
│   ├── case_study_alkane_pyrolysis_analysis  # (figure 4 plots. code from `code/analyses_code/case_study_alkane_pyrolysis_analysis.py`)
│   ├── detailed_accuracy_plots  # (figure 3 plots. code from `code/analyses_code/detailed_accuracy_plots.py`)
│   ├── k_fold_accuracy_analysis  # (figure 2 plots. code from `code/analyses_code/k_fold_accuracy_analysis.py`)
│   └── parity_plots # (figure 1 plots. code from `code/analyses_code/parity_plots.py`)
├── code  # (all code)
│   ├── analyses_code  # (code to reproduce figures in `analyses_and_plots`)
│   │   ├── case_study_alkane_pyrolysis_analysis.py  # (figure 4)
│   │   ├── detailed_accuracy_plots.py  # (figure 3)
│   │   ├── k_fold_accuracy_analysis.py  # (figure 2)
│   │   └── parity_plots.py  # (figure 1)
│   ├── ml_code  # (code to split data, and train+test models)
│   │   ├── case_study_alkane_pyrolysis_mr_benchmark.py  # (case study alkane pyrolysis benchmark. fig 4. see 'data/case_study_alkane_pyrolysis_mr_benchmark' for output)
│   │   ├── create_fingerprints.py  # (creates drfp from atom mapped SMILES. see `data/fps` for output)
│   │   ├── create_splits.py  # (creates train/test splits for models. see `data/splits` for output)
│   │   ├── train_chemprop_delta.py  # (train+pred ChemProp delta model)
│   │   ├── train_chemprop_delta_transfer.py  # (train+pred ChemProp delta transfer models)
│   │   ├── train_chemprop_direct.py  # (train+pred ChemProp direct model)
│   │   ├── train_xgb_delta.py  # (train+pred XGBoost delta model)
│   │   └── train_xgb_direct.py  # (train+pred XGBoost direct model)
│   └── utils.py  # (contains functions used in `ml_code` and `analyses_code`)
├── data  # (all data)
│   ├── case_study_alkane_pyrolysis_mr_benchmark  # (alkane pyrolysis benchmark data. code from `code/ml_code/case_study_alkane_pyrolysis_mr_benchmark.py`)
│   │   ├── alkane_pyrolysis_mr_benchmark.csv  # (raw data extracted from pyrolysis article: https://doi.org/10.1039/D4DD00036F)
│   │   ├── chemprop_delta_preds  # (ChemProp delta model predictions on the alkane pyrolysis benchmark)
│   │   ├── chemprop_delta_transfer_preds_b2plypd3  # (ChemProp delta transfer model (b2plypd3/cc-pvtz) predictions on the alkane pyrolysis benchmark)
│   │   ├── chemprop_delta_transfer_preds_wb97xd  # (ChemProp delta transfer model (wb97xd/def2-tzvp) predictions on the alkane pyrolysis benchmark)
│   │   ├── mr_ar_am_smi_dict.json  # (atom mapped SMILES dictionary for alkane pyrolysis benchmark)
│   │   └── mr_ar_ea_dict.json  # (barrier dictionary for alkane pyrolysis benchmark)
│   ├── high_level_single_pt_refinement  # (high level single point refinement data for alkane pyrolysis benchmark)
│   │   ├── mr_ar_ea_combined_dict_b2plypd3.json  # (barrier (b2plypd3/cc-pvtz) dictionary for alkane pyrolysis benchmark using b3lyp-d3/tzvp geometries)
│   │   ├── mr_ar_ea_combined_dict_wb97xd.json  # (barrier (wb97xd/def2-tzvp) dictionary for alkane pyrolysis benchmark using b3lyp-d3/tzvp geometries)
│   │   ├── react_prod_b2plypd3.csv  # (reactants/products thermochemistry (b2plypd3/cc-pvtz) data for alkane pyrolysis benchmark using b3lyp-d3/tzvp geometries)
│   │   ├── react_prod_wb97xd.csv  # (reactants/products thermochemistry (wb97xd/def2-tzvp) data for alkane pyrolysis benchmark using b3lyp-d3/tzvp geometries)
│   │   ├── ts_b2plypd3.csv  # (transition state thermochemistry (b2plypd3/cc-pvtz) data for alkane pyrolysis benchmark using b3lyp-d3/tzvp geometries)
│   │   └── ts_wb97xd.csv  # (transition state thermochemistry (wb97xd/def2-tzvp) data for alkane pyrolysis benchmark using b3lyp-d3/tzvp geometries)
│   ├── fps  # (drfp for train/test data for xgb models generated from `code/ml_code/create_fingerprints.py`)
│   ├── models  # (pretrained models. download from: . and extract here)
│   ├── preds  # (k-fold predictions from models across all splits and train data percents. see `code/ml_code/train_chemprop_delta.py`, `train_chemprop_delta_transfer.py`, `train_chemprop_direct.py`, `train_xgb_delta.py`, and `train_xgb_direct.py` for details)
│   └── splits  # (all train/test k-fold splits with different train percents for models generated from `code/ml_code/create_splits.py`)
└── README.md
```