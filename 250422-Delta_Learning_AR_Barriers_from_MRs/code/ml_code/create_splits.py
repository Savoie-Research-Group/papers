"""
    Last Modified: 2025/04/04
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: divide ar names into different types of 10-fold splits based on mr classes corresponding to ars.
"""


import os
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, ShuffleSplit, train_test_split


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


import sys
sys.path.append(os.path.join(this_script_dir, ".."))


from utils import *


def create_stratified_k_fold_splits(ar_mr_dict, mr_ar_list_dict, n_splits=10, shuffle=True, random_state=random_seed):
    """
    Create stratified k-fold splits for given reaction data.

    Args:
    ar_mr_dict (dict): Dictionary mapping ar names to corresponding mr classes.
    mr_ar_list_dict (dict): Dictionary mapping mr classes to lists of ar names.
    n_splits (int, optional): Number of folds. Default is 10.
    shuffle (bool, optional): Whether to shuffle each class's samples before splitting them. Default is True.
    random_state (int, optional): Random seed for reproducibility. Default is random_seed.

    Returns:
    dict: A dictionary where keys are fold numbers and values are lists containing train, val, and test ar names.
    """

    ## Remove ARs which don't have at least two MRs
    ## This is necessary for StratifiedKFold to work
    x = list(ar_mr_dict.keys())
    x = [i for i in x if len(mr_ar_list_dict[ar_mr_dict[i]]) > 2]
    y = [ar_mr_dict[ar] for ar in x]
    x = np.array(x)
    y = np.array(y)

    ## Create StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)  # to create train and test

    ## Split the data into train and test sets
    splits = skf.split(x, y)

    ## Create StratifiedShuffleSplit object for train:val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0/9.0, train_size=8.0/9.0, random_state=random_state)  # to create train and val from train
    
    # Create StratifiedShuffleSplit objects for varying train set sizes
    sss_80_from_100 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=random_state)  ## 80% of the full train set
    sss_60_from_80 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, train_size=0.75, random_state=random_state)  ## 60% of the full train set is 75% of the 80% of the full train set
    sss_40_from_60 = StratifiedShuffleSplit(n_splits=1, test_size=0.3333, train_size=0.6667, random_state=random_state)  ## 40% of the full train set is ~66.67% of the 60% of the full train set
    sss_20_from_40 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=0.5, random_state=random_state)  ## 20% of the full train set is 50% of the 40% of the full train set
    
    ## Create a dictionary to store the splits
    fold_split_dict = {}  ## {fold: [[[train_ar_names_20%], [train_ar_names_40%], [train_ar_names_60%], [train_ar_names_80%], [train_ar_names_100%]], [val_ar_names], [test_ar_names]]}
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        fold = str(fold)
        
        # get the final test list
        test_ar_list = x[test_idx].tolist()
        
        ## Split the train set into train and val sets
        train_ar_temp = x[train_idx]
        train_idx_new, val_idx = next(sss.split(train_ar_temp, y[train_idx]))
        
        # get the final val list
        val_ar_list = train_ar_temp[val_idx].tolist()
        
        # get the final train lists
        train_ar_list_100 = train_ar_temp[train_idx_new]  ## full train set
        train_ar_list_100_mrs = np.array([ar_mr_dict[ar] for ar in train_ar_list_100])
        
        train_idx_80, _ = next(sss_80_from_100.split(train_ar_list_100, train_ar_list_100_mrs))  ## 80% of the full train set
        train_ar_list_80 = train_ar_list_100[train_idx_80]
        train_ar_list_80_mrs = np.array([ar_mr_dict[ar] for ar in train_ar_list_80])
        
        train_idx_60, _ = next(sss_60_from_80.split(train_ar_list_80, train_ar_list_80_mrs))  ## 60% of the full train set
        train_ar_list_60 = train_ar_list_80[train_idx_60]
        train_ar_list_60_mrs = np.array([ar_mr_dict[ar] for ar in train_ar_list_60])
        
        train_idx_40, _ = next(sss_40_from_60.split(train_ar_list_60, train_ar_list_60_mrs))  ## 40% of the full train set
        train_ar_list_40 = train_ar_list_60[train_idx_40]
        train_ar_list_40_mrs = [ar_mr_dict[ar] for ar in train_ar_list_40]
        
        train_ar_list_40_for_20 = np.array([j for i, j in enumerate(train_ar_list_40) if train_ar_list_40_mrs.count(train_ar_list_40_mrs[i]) > 1])  ## to ensure proper stratification in the 20% train set
        train_ar_list_40_mrs_for_20 = np.array([ar_mr_dict[ar] for ar in train_ar_list_40_for_20])
        
        train_idx_20, _ = next(sss_20_from_40.split(train_ar_list_40_for_20, train_ar_list_40_mrs_for_20))  ## 20% of the full train set
        train_ar_list_20 = train_ar_list_40[train_idx_20]

        train_ar_list_100 = train_ar_list_100.tolist() + list(set(y.tolist()))  # fold mr into train
        train_ar_list_80 = train_ar_list_80.tolist() + list(set(y.tolist()))  # fold mr into train
        train_ar_list_60 = train_ar_list_60.tolist() + list(set(y.tolist()))  # fold mr into train
        train_ar_list_40 = train_ar_list_40.tolist() + list(set(y.tolist()))  # fold mr into train
        train_ar_list_20 = train_ar_list_20.tolist() + list(set(y.tolist()))  # fold mr into train
        random.shuffle(train_ar_list_100)
        random.shuffle(train_ar_list_80)
        random.shuffle(train_ar_list_60)
        random.shuffle(train_ar_list_40)
        random.shuffle(train_ar_list_20)
        
        ## Store the splits in the fold_split_dict = {fold: [[[train_ar_names_20%], [train_ar_names_40%], [train_ar_names_60%], [train_ar_names_80%], [train_ar_names_100%]], [val_ar_names], [test_ar_names]]}
        fold_split_dict[fold] = [[train_ar_list_20, train_ar_list_40, train_ar_list_60, train_ar_list_80, train_ar_list_100], val_ar_list, test_ar_list]
        
        
    ## Print some stats about the splits
    for fold, (train_ar_list_list, val_ar_list, test_ar_list) in fold_split_dict.items():
        train_len_20 = len(train_ar_list_list[0])
        train_len_40 = len(train_ar_list_list[1])
        train_len_60 = len(train_ar_list_list[2])
        train_len_80 = len(train_ar_list_list[3])
        train_len_100 = len(train_ar_list_list[4])
        val_len = len(val_ar_list)
        test_len = len(test_ar_list)
        total_len = train_len_100 + val_len + test_len
        print(f"fold: {fold}, total_len: {total_len}, train_frac_20: {train_len_20/total_len}, train_frac_40: {train_len_40/total_len}, train_frac_60: {train_len_60/total_len}, train_frac_80: {train_len_80/total_len}, train_frac_100: {train_len_100/total_len}, val_frac: {val_len/total_len}, test_frac: {test_len/total_len}")
    
    return fold_split_dict


def create_random_k_fold_splits(ar_mr_dict, n_splits=10, shuffle=True, random_state=random_seed):
    """
    Creates random k-fold cross-validation splits for AR names.

    Args:
        ar_mr_dict (dict): A dictionary where keys are AR names and values are MR classes.
        n_splits (int): Number of folds for cross-validation. Default is 10.
        shuffle (bool): Whether to shuffle the data before splitting. Default is True.
        random_state (int): Random seed for reproducibility. Default is `random_seed`.

    Returns:
        dict: A dictionary where keys are fold numbers (as strings) and values are lists 
              containing train AR names, validation AR names, and test AR names for each fold.
    """

    # Extract AR names and their corresponding MR classes
    x = list(ar_mr_dict.keys())
    y = [ar_mr_dict[ar] for ar in x]
    
    # Convert lists to numpy arrays for efficient indexing
    x = np.array(x)
    y = np.array(y)
    
    # Initialize KFold to create train and test splits
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Generate KFold splits
    splits = kf.split(x)
    
    # Initialize ShuffleSplit to create train and val splits from train
    ss = ShuffleSplit(n_splits=1, test_size=1.0/9.0, train_size=8.0/9.0, random_state=random_state)
    
    # ShuffleSplits for varying train sizes
    ss_80_from_100 = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=random_state)
    ss_60_from_80 = ShuffleSplit(n_splits=1, test_size=0.25, train_size=0.75, random_state=random_state)
    ss_40_from_60 = ShuffleSplit(n_splits=1, test_size=0.3333, train_size=0.6667, random_state=random_state)
    ss_20_from_40 = ShuffleSplit(n_splits=1, test_size=0.5, train_size=0.5, random_state=random_state)
    
    # Dictionary to store the train, val, and test splits for each fold
    fold_split_dict = {}  # {fold: [[[train_ar_names_20%], [train_ar_names_40%], [train_ar_names_60%], [train_ar_names_80%], [train_ar_names_100%]], [val_ar_names], [test_ar_names]]}
    
    # Iterate over each fold generated by KFold
    for fold, (train_idx, test_idx) in enumerate(splits):
        fold = str(fold)
        
        # Get the test AR names for the current fold
        test_ar_list = x[test_idx].tolist()
        
        # Temporarily store train AR names
        train_ar_temp = x[train_idx]
        
        # Split the train set into new train and val sets
        train_idx_new, val_idx = next(ss.split(train_ar_temp, x[train_idx]))
        
        # Get the validation AR names for the current fold
        val_ar_list = train_ar_temp[val_idx].tolist()
        
        # Get the final train AR names and ensure all unique MR classes are included
        train_ar_list_100 = train_ar_temp[train_idx_new]
        
        train_idx_80, _ = next(ss_80_from_100.split(train_ar_list_100))
        train_ar_list_80 = train_ar_list_100[train_idx_80]
        
        train_idx_60, _ = next(ss_60_from_80.split(train_ar_list_80))
        train_ar_list_60 = train_ar_list_80[train_idx_60]
        
        train_idx_40, _ = next(ss_40_from_60.split(train_ar_list_60))
        train_ar_list_40 = train_ar_list_60[train_idx_40]
        
        train_idx_20, _ = next(ss_20_from_40.split(train_ar_list_40))
        train_ar_list_20 = train_ar_list_40[train_idx_20]
        
        train_ar_list_100 = train_ar_list_100.tolist() + list(set(y.tolist()))
        train_ar_list_80 = train_ar_list_80.tolist() + list(set(y.tolist()))
        train_ar_list_60 = train_ar_list_60.tolist() + list(set(y.tolist()))
        train_ar_list_40 = train_ar_list_40.tolist() + list(set(y.tolist()))
        train_ar_list_20 = train_ar_list_20.tolist() + list(set(y.tolist()))
        
        random.shuffle(train_ar_list_100)
        random.shuffle(train_ar_list_80)
        random.shuffle(train_ar_list_60)
        random.shuffle(train_ar_list_40)
        random.shuffle(train_ar_list_20)
        
        # Store the train, val, and test lists in the fold_split_dict = {fold: [[[train_ar_names_20%], [train_ar_names_40%], [train_ar_names_60%], [train_ar_names_80%], [train_ar_names_100%]], [val_ar_names], [test_ar_names]]}
        fold_split_dict[fold] = [[train_ar_list_20, train_ar_list_40, train_ar_list_60, train_ar_list_80, train_ar_list_100], val_ar_list, test_ar_list]

        
    ## Print statistics about the split proportions for each fold
    for fold, (train_ar_list_list, val_ar_list, test_ar_list) in fold_split_dict.items():
        train_len_20 = len(train_ar_list_list[0])
        train_len_40 = len(train_ar_list_list[1])
        train_len_60 = len(train_ar_list_list[2])
        train_len_80 = len(train_ar_list_list[3])
        train_len_100 = len(train_ar_list_list[4])
        val_len = len(val_ar_list)
        test_len = len(test_ar_list)
        total_len = train_len_100 + val_len + test_len
        print(f"fold: {fold}, total_len: {total_len}, train_frac_20: {train_len_20/total_len}, train_frac_40: {train_len_40/total_len}, train_frac_60: {train_len_60/total_len}, train_frac_80: {train_len_80/total_len}, train_frac_100: {train_len_100/total_len}, val_frac: {val_len/total_len}, test_frac: {test_len/total_len}")
    
    
    return fold_split_dict


def create_mr_scaffold_k_fold_splits(ar_mr_dict, mr_ar_list_dict, n_splits=10, shuffle=True, random_state=random_seed):
    """
    Creates random k-fold cross-validation splits for AR names, treating MRs as scaffolds.

    Args:
        ar_mr_dict (dict): A dictionary where keys are AR names and values are MR classes.
        mr_ar_list_dict (dict): A dictionary where keys are MR classes and values are lists of AR names.
        n_splits (int): Number of folds for cross-validation. Default is 10.
        shuffle (bool): Whether to shuffle the data before splitting. Default is True.
        random_state (int): Random seed for reproducibility. Default is `random_seed`.

    Returns:
        dict: A dictionary where keys are fold numbers (as strings) and values are lists 
              containing train AR names, validation AR names, and test AR names for each fold.
    """
    
    # Get the MR classes as an array
    x = list(mr_ar_list_dict.keys())
    x = np.array(x)
    
    # Create a KFold object to create train and test splits
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = kf.split(x)
    
    # Create a ShuffleSplit object to create train and val splits from train
    ss = ShuffleSplit(n_splits=1, test_size=1.0/9.0, train_size=8.0/9.0, random_state=random_state)
    
    # ShuffleSplits for varying train sizes
    ss_80_from_100 = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=random_state)
    ss_60_from_80 = ShuffleSplit(n_splits=1, test_size=0.25, train_size=0.75, random_state=random_state)
    ss_40_from_60 = ShuffleSplit(n_splits=1, test_size=0.3333, train_size=0.6667, random_state=random_state)
    ss_20_from_40 = ShuffleSplit(n_splits=1, test_size=0.5, train_size=0.5, random_state=random_state)
    
    # Initialize a dictionary to store the train, val, and test splits for each fold
    fold_split_dict = {}  ## {fold: [[[train_ar_names_20%], [train_ar_names_40%], [train_ar_names_60%], [train_ar_names_80%], [train_ar_names_100%]], [val_ar_names], [test_ar_names]]}
    
    # Iterate over each fold generated by KFold
    for fold, (train_idx, test_idx) in enumerate(splits):
        fold = str(fold)
        
        # Get the test MR classes for the current fold
        test_mr_list = x[test_idx].tolist()
        
        # Temporarily store the train MR classes
        train_mr_temp = x[train_idx]
        
        # Split the train set into new train and val sets
        train_idx_new, val_idx = next(ss.split(train_mr_temp, x[train_idx]))
        
        # Get the final train and val MR classes
        train_mr_list = train_mr_temp[train_idx_new].tolist()
        val_mr_list = train_mr_temp[val_idx].tolist()
        
        # Initialize lists to store the train, val, and test AR names
        test_ar_list = []
        train_ar_list = []
        val_ar_list = []
        
        # Populate the lists with AR names corresponding to the MR classes
        for mr in test_mr_list:
            test_ar_list.extend(mr_ar_list_dict[mr])
        for mr in train_mr_list:
            train_ar_list.extend(mr_ar_list_dict[mr])
        for mr in val_mr_list:
            val_ar_list.extend(mr_ar_list_dict[mr])
        
        test_ar_list.extend(test_mr_list)
        val_ar_list.extend(val_mr_list)
        random.shuffle(test_ar_list)
        random.shuffle(val_ar_list)
        
        train_mr_list_100 = np.array(train_mr_list)
        train_ar_list_100 = deepcopy(train_ar_list)
        
        train_mr_idx_80, _ = next(ss_80_from_100.split(train_mr_list_100))
        train_mr_list_80 = train_mr_list_100[train_mr_idx_80]
        train_ar_list_80 = []
        for mr_80 in train_mr_list_80:
            train_ar_list_80.extend(mr_ar_list_dict[mr_80])
        
        train_mr_idx_60, _ = next(ss_60_from_80.split(train_mr_list_80))
        train_mr_list_60 = train_mr_list_80[train_mr_idx_60]
        train_ar_list_60 = []
        for mr_60 in train_mr_list_60:
            train_ar_list_60.extend(mr_ar_list_dict[mr_60])
        
        train_mr_idx_40, _ = next(ss_40_from_60.split(train_mr_list_60))
        train_mr_list_40 = train_mr_list_60[train_mr_idx_40]
        train_ar_list_40 = []
        for mr_40 in train_mr_list_40:
            train_ar_list_40.extend(mr_ar_list_dict[mr_40])
        
        train_mr_idx_20, _ = next(ss_20_from_40.split(train_mr_list_40))
        train_mr_list_20 = train_mr_list_40[train_mr_idx_20]
        train_ar_list_20 = []
        for mr_20 in train_mr_list_20:
            train_ar_list_20.extend(mr_ar_list_dict[mr_20])
        
        # Add the MR classes to the lists
        train_ar_list_100.extend(train_mr_list_100.tolist())
        train_ar_list_80.extend(train_mr_list_80.tolist())
        train_ar_list_60.extend(train_mr_list_60.tolist())
        train_ar_list_40.extend(train_mr_list_40.tolist())
        train_ar_list_20.extend(train_mr_list_20.tolist())
        
        # Shuffle the lists
        random.shuffle(train_ar_list_100)
        random.shuffle(train_ar_list_80)
        random.shuffle(train_ar_list_60)
        random.shuffle(train_ar_list_40)
        random.shuffle(train_ar_list_20)
        
        # Store the lists in the fold_split_dict = {fold: [[[train_ar_names_20%], [train_ar_names_40%], [train_ar_names_60%], [train_ar_names_80%], [train_ar_names_100%]], [val_ar_names], [test_ar_names]]}
        fold_split_dict[fold] = [[train_ar_list_20, train_ar_list_40, train_ar_list_60, train_ar_list_80, train_ar_list_100], val_ar_list, test_ar_list]
        
    ## Print statistics about the split proportions for each fold
    for fold, (train_ar_list_list, val_ar_list, test_ar_list) in fold_split_dict.items():
        train_len_20 = len(train_ar_list_list[0])
        train_len_40 = len(train_ar_list_list[1])
        train_len_60 = len(train_ar_list_list[2])
        train_len_80 = len(train_ar_list_list[3])
        train_len_100 = len(train_ar_list_list[4])
        val_len = len(val_ar_list)
        test_len = len(test_ar_list)
        total_len = train_len_100 + val_len + test_len
        print(f"fold: {fold}, total_len: {total_len}, train_frac_20: {train_len_20/total_len}, train_frac_40: {train_len_40/total_len}, train_frac_60: {train_len_60/total_len}, train_frac_80: {train_len_80/total_len}, train_frac_100: {train_len_100/total_len}, val_frac: {val_len/total_len}, test_frac: {test_len/total_len}")
    
    return fold_split_dict


def main():
    # Check if splits dir in data_dir. Else create it.
    splits_dir = os.path.join(data_dir, "splits")
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)
    else:
        print("splits dir already exists.")
    
    ar_mr_dict_path = os.path.join(data_paper_data_dir, "ar_mr_dict.json")
    ar_mr_dict = json.load(open(ar_mr_dict_path, "r"))
    
    mr_ar_list_dict_path = os.path.join(data_paper_data_dir, "mr_ar_list_dict.json")
    mr_ar_list_dict = json.load(open(mr_ar_list_dict_path, "r"))
    
    # Create and save the 10-fold stratified splits
    stratified_k_fold_splits_dict = create_stratified_k_fold_splits(ar_mr_dict, mr_ar_list_dict, n_splits=10, shuffle=True, random_state=random_seed)
    json.dump(stratified_k_fold_splits_dict, open(os.path.join(splits_dir, "stratified_k_fold_splits_dict.json"), "w"), indent=4)
    
    # Create and save the 10-fold random splits
    random_k_fold_splits_dict = create_random_k_fold_splits(ar_mr_dict, n_splits=10, shuffle=True, random_state=random_seed)
    json.dump(random_k_fold_splits_dict, open(os.path.join(splits_dir, "random_k_fold_splits_dict.json"), "w"), indent=4)
    
    # Create and save the MR scaffold-based splits
    mr_scaffold_k_fold_splits_dict = create_mr_scaffold_k_fold_splits(ar_mr_dict, mr_ar_list_dict, n_splits=10, shuffle=True, random_state=random_seed)
    json.dump(mr_scaffold_k_fold_splits_dict, open(os.path.join(splits_dir, "mr_scaffold_k_fold_splits_dict.json"), "w"), indent=4)
    
    return


if __name__ == "__main__":
    main()
