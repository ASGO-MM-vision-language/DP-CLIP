import torch
from torch import nn
import torch.nn.functional as F

import pdb

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def pair_cosine_dist(x, y):

    dist = 1. - F.cosine_similarity(x, y)
    # dist = dist / 2.0
    # print('dist.shape', dist.shape)
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    # shape [N, N]
    # is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
  
    for i in range(N):
        ap = dist_mat[i,i]
        if i == 0 :
            dist_ap = ap.view(1,1)
        else:
            dist_ap = torch.cat([dist_ap, ap.view(1,1)], dim=0)

    dist_ap = dist_ap.squeeze(1)

    dist_an= None
    for i in range(N):
        an = torch.min(dist_mat[i, is_neg[i]])
        if i == 0 :
            dist_an = an.view(1,1)
        else:
            dist_an = torch.cat([dist_an, an.view(1,1)], dim=0)
    dist_an = dist_an.squeeze(1)   


    return dist_ap, dist_an

class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=0.3, hard_factor=0.0, if_hard=True, distance_type='cos'):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin > 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.if_hard = if_hard
        self.distance_type = distance_type


    def __call__(self, sk_feat, img_feat, neg_feat=None, labels=None, normalize_feature=False):
        if normalize_feature:
            sk_feat = normalize(sk_feat, axis=-1)
            img_feat = normalize(img_feat, axis=-1)
        
        if self.if_hard:
            if self.distance_type == 'eud':
                dist_mat = euclidean_dist(sk_feat, img_feat)
            elif self.distance_type == 'cos':
                dist_mat = cosine_dist(sk_feat, img_feat)
            dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        else:
            dist_ap = pair_cosine_dist(sk_feat, img_feat)
            dist_an = pair_cosine_dist(sk_feat, neg_feat)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin > 0:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
            # print('soft margin loss', loss)
        return loss #, dist_ap, dist_an


