"""
    Date Modified: 2024/01/23
    Author: Veerupaksh (Veeru) Singla (singla2@purdue.edu)
    Description: additions to Chemprop.
"""


import numpy as np
from typing import List


# all changes on "442a1602b670f173de166f987eab64396571ee98" hash of Chemprop (https://github.com/chemprop/chemprop)


# chemprop/train/loss_functions.py needs to be updated in the following way:
'''
my_hinge_loss function needs to be added
the "regression" subdict in the supported_loss_functions dict needs to be updated to include my_hinge_loss: "my_hinge": my_hinge_loss,
'''


# numpy based my_hinge_loss to find loss in evaluation mode on the cpu. needs to be added to chemprop/train/metrics.py
def my_hinge_loss(targets_: List[float], preds_: List[float]):
    
    targets_ = np.array(targets_)
    preds_ = np.array(preds_)
    
    f_0_ = targets_ - np.roll(targets_, -1, axis=0)
    f_1_ = preds_ - np.roll(preds_, -1, axis=0)
    
    # for i_ in range(2, targets_.shape[0]):
    #     f_0_ = np.concatenate((f_0_, targets_ - np.roll(targets_, -1 * i_, axis=0)), axis=0)
    #     f_1_ = np.concatenate((f_1_, preds_ - np.roll(preds_, -1 * i_, axis=0)), axis=0)
    
    return np.sum(np.maximum(np.multiply(f_0_ - f_1_, np.sign(f_0_)), 0.), axis=-1)


# get_metric_func in chemprop/train/metrics.py also needs to be updated to include my_hinge_loss
    '''
    if metric=='my_hinge':
        return my_hinge_loss
    '''
    

# chemprop/args.py needs to be updated in the following way:
'''
line 19: add 'my_hinge to the Metric list
line 235: add 'my_hinge' to loss_function: list
line 462: add 'my_hinge' in the return set
line 588: add 'my_hinge' to the metric list
'''

# chemprop/utils.py needs to be updated in the following way:
'''
line 794: add 'my_hinge' to the nonscale_dependent_metrics list
'''
