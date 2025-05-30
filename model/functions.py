import torch
import torch.nn as nn
import numpy as np

from .clip.weight_init import trunc_normal_


def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

def freeze(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        classname.weight.requires_grad_(False)
    

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.002, mean=0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def KL_divergence_normalized(p, q):
    """
    Kullback-Leibler divergence of two (empirical) probability distributions.
    
    Parameters
    ----------
    p : numpy.ndarray
        Vector of the values of the first (discrete) probability distribution.
    q : numpy.ndarray
        Vector of the values of the second (discrete) probability distribution.
    
    Returns
    -------
    res : numpy.float64
        Result of the computation of the Kullback-Leibler divergence of p from q.
    """
    
    res = np.sum(np.where(p!=0, p*np.log(p/q), 0))
    n = len(p)
    
    return res/n


def print_weight(m):
    # if not isinstance(m, torch.nn.LayerNorm):
    if hasattr(m, 'weight') and m.weight is not None:
        print(m, m.weight.data.sum())
        # m.weight.requires_grad_(False)
    if hasattr(m, 'bias') and m.bias is not None:
        # m.bias.requires_grad_(False)
        print(m, m.bias.data.sum())
