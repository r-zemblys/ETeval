#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rz
"""
import copy, itertools
import numpy as np
import pandas as pd

from sklearn import metrics

#%%
def calc_k(gt, pr):
    '''
    Handles error if all samples are from the same class
    '''
    k = 1. if (gt == pr).all() else metrics.cohen_kappa_score(gt, pr)
    return k

#%%

def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

#%%
def calc_evt_overlap(etdata_gt, etdata_pr):
    '''Calculates event overlaps.
    Parameters:
        etdata_gt   --  instance of ETData containing ground truth data
        etdata_pr   --  instance of ETData containing comparison data
    Returns:

    '''
    #calculate event level matches
    gt_evt_index = [ind for i, n in enumerate(np.diff(etdata_gt.evt[['s', 'e']]).squeeze(1))
                        for ind in itertools.repeat(i, n)]
    pr_evt_index = [ind for i, n in enumerate(np.diff(etdata_pr.evt[['s', 'e']]).squeeze(1))
                        for ind in itertools.repeat(i, n)]

    overlap = np.vstack((gt_evt_index, pr_evt_index)).T

    overlap_matrix = [_k + [len(list(_g)), False, False] for _k, _g in itertools.groupby(overlap.tolist())]
    overlap_matrix = pd.DataFrame(overlap_matrix, columns=['gt', 'pr', 'l', 'matched', 'selected'])
    overlap_matrix['gt_evt'] = etdata_gt.evt.loc[overlap_matrix['gt'], 'evt'].values
    overlap_matrix['pr_evt'] = etdata_pr.evt.loc[overlap_matrix['pr'], 'evt'].values


    while not(overlap_matrix['matched'].all()):
        #select longest overlap
        ind = overlap_matrix.loc[~overlap_matrix['matched'], 'l'].argmax()
        overlap_matrix.loc[ind, ['selected']]=True
        mask_matched = (overlap_matrix['gt']==overlap_matrix.loc[ind, 'gt']).values |\
                       (overlap_matrix['pr']==overlap_matrix.loc[ind, 'pr']).values
        overlap_matrix.loc[mask_matched, 'matched'] = True
    overlap_events = overlap_matrix.loc[overlap_matrix['selected'], ['gt', 'pr', 'gt_evt', 'pr_evt']]

    evt_gt = etdata_gt.evt.loc[overlap_events['gt'], 'evt']
    evt_pr = etdata_pr.evt.loc[overlap_events['pr'], 'evt']

    #add not matched events
    set_gt = set(etdata_gt.evt.index.values) - set(evt_gt.index.values)
    set_pr = set(etdata_pr.evt.index.values) - set(evt_pr.index.values)

    evt_gt = pd.concat((evt_gt, etdata_gt.evt.loc[set_gt, 'evt']))
    evt_pr = pd.concat((evt_pr, pd.DataFrame(np.zeros(len(set_gt)))))

    evt_gt = pd.concat((evt_gt, pd.DataFrame(np.zeros(len(set_pr)))))
    evt_pr = pd.concat((evt_pr, etdata_pr.evt.loc[set_pr, 'evt']))

    return overlap_events.values, np.squeeze(evt_gt.values.astype(np.int32)), np.squeeze(evt_pr.values.astype(np.int32))


#%%
def eval_evt(etdata_gt, etdata_pr, num_classes):
    #calculate events if not yet calculated
    if etdata_gt.evt is None:
        etdata_gt.calc_evt(fast=True)
    if etdata_pr.evt is None:
        etdata_pr.calc_evt(fast=True)

    ke = []
    ##evaluate all classes separately
    for evt in range(1, num_classes): #do not evaluate 0th class

        #makes a copy of the data, sets the class of interest to 1
        #and everything else to 0. In addition marks "unknown" class to 255
        _etdata_gt = copy.deepcopy(etdata_gt)
        mask_ext = _etdata_gt.data['evt']==0
        mask_evt = _etdata_gt.data['evt']==evt
        _etdata_gt.data['evt'][mask_evt]=1
        _etdata_gt.data['evt'][~mask_evt]=0
        _etdata_gt.data['evt'][mask_ext]=255
        _etdata_gt.calc_evt(fast=True)

        _etdata_pr = copy.deepcopy(etdata_pr)
        mask_ext = _etdata_pr.data['evt']==0
        mask_evt = _etdata_pr.data['evt']==evt
        _etdata_pr.data['evt'][mask_evt]=1
        _etdata_pr.data['evt'][~mask_evt]=0
        _etdata_pr.data['evt'][mask_ext]=255
        _etdata_pr.calc_evt(fast=True)

        #calculates overlap, excludes "unknown" class and calculates KE
        evt_overlap, evt_gt, evt_pr = calc_evt_overlap(_etdata_gt, _etdata_pr)
        mask = (evt_gt==255) & (evt_pr==255)
        evt_gt = evt_gt[~mask]
        evt_pr = evt_pr[~mask]
        ke.append(calc_k(evt_gt, evt_pr))

    ##evaluate all classes together
    #calculates overlap, excludes "unknown" class and calculates KE
    evt_overlap, evt_gt, evt_pr = calc_evt_overlap(etdata_gt, etdata_pr)
    mask = (evt_gt==0) & (evt_pr==0)
    evt_gt = evt_gt[~mask]
    evt_pr = evt_pr[~mask]
    ke_all = metrics.cohen_kappa_score(evt_gt, evt_pr)

    ke.extend([ke_all])

    return ke, (evt_overlap, evt_gt, evt_pr)
