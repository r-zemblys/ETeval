#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:14:05 2018

@author: rz
@email:r.zemblys@tf.su.lt
"""
#%% imports
import os, time
from tqdm import tqdm

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#plt.rcParams['image.cmap'] = 'gray'
#plt.rc("axes.spines", top=False, right=False)
#plt.ion()

###
import copy, fnmatch, argparse, json
from datetime import datetime
from utils_lib.ETeval import eval_evt
from utils_lib.etdata import ETData

def map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val
class_map_func  = np.vectorize(map_func)

def get_arguments():
    parser = argparse.ArgumentParser(description='ETeval')
    parser.add_argument('--config', type=str, default='example_job.json',
                        help='JSON file with the experiment data description')
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment')

    return parser.parse_args()

#%% setup parameters
args = get_arguments()
with open(args.config, 'r') as f:
    data=json.load(f)

etdata_gt = ETData()
etdata_pr = ETData()

columns = [
   'dataset', 'algorithm', 'fname',
   'ke_fix', 'ke_sacc', 'ke_pso'
]
rez_format = 'Event level scores\nFixations\t{:.3f}\nSaccades\t{:.3f}\nPSO\t\t{:.3f}'

#human format coding scheme
classes = [
    0, #undef
    1, #fixation
    2, #saccade
    3, #pso
    9  #everything else
]
#internal coding scheme
class_mapper = {k:v for k, v in zip (classes, np.arange(len(classes)))}

#%% code
eval_mode = args.exp
for _data in data:
    _suffix = 'dev' if eval_mode is None else '%s_dev' % eval_mode
    fpath_rez = '%s/%s/%s_%s.csv'%(_data['root'], _data['pr'], _data['alg'], _suffix)
    print (fpath_rez)
    fpath_log = '%s/%s/%s-%s.log'%(_data['root'], _data['pr'], _data['alg'],
                                   datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    FILES = []
    ddir = '%s/%s'%(_data['root'], _data['gt'])
    for _root, _dir, _files in os.walk(ddir):
        FILES.extend(['%s/%s' % (_root, _file)
                      for _file in fnmatch.filter(_files, "*.npy")])
    if not (len(FILES)):
        with open(fpath_log, 'a') as f:
            f.write('EMPTY\n')
        continue

    rez = []
    etdata_gt_meta = []
    etdata_pr_meta = []
    for fpath in tqdm(sorted(FILES)):
        fdir, fname = os.path.split(os.path.splitext(fpath)[0])

        fpath_pr = fpath.replace(_data['gt'], _data['pr'])
        if not(os.path.exists(fpath_pr)):
            with open(fpath_log, 'a') as f:
                f.write('MISSING:\t%s\n'%fpath_pr)
            continue

        etdata_gt.load(fpath)
        etdata_pr.load(fpath_pr)

        #leaves original predictions untouched
        _etdata_gt = copy.deepcopy(etdata_gt)
        _etdata_pr = copy.deepcopy(etdata_pr)

        #analyse only fixations, saccades and pso
        mask_gt = ~_etdata_gt.data['status'] #original undef
        _etdata_gt.data['evt'][mask_gt] = 0
        mask_pr = ~np.in1d(_etdata_pr.data['evt'], [1, 2, 3])
        _etdata_pr.data['evt'][mask_pr] = 9 #sets everything else to "other"
        _etdata_pr.data['evt'][mask_gt] = 0 #undef must be the same in both

        #internal class mapping
        _etdata_gt.data['evt'] = class_map_func(_etdata_gt.data['evt'], class_mapper)
        _etdata_pr.data['evt'] = class_map_func(_etdata_pr.data['evt'], class_mapper)

        #evaluate per trial
        ke, (evt_overlap, _evt_gt, _evt_pr) = eval_evt(_etdata_gt, _etdata_pr, len(classes), mode=eval_mode)
        flabel = fpath.replace(ddir, '')
        #save only fixation, saccade and pso
        rez.append((_data['dataset'], _data['alg'], flabel) + tuple(ke[:3]) )

        #stack everything into one meta trial; separate trials by 0s
        etdata_gt_meta.append(_etdata_gt.data)
        etdata_gt_meta.append(np.zeros(1).astype(_etdata_gt.dtype))
        etdata_pr_meta.append(_etdata_pr.data)
        etdata_pr_meta.append(np.zeros(1).astype(_etdata_pr.dtype))


    #evaluate meta trials
    print ("Evaluating meta trial")
    tic = time.time()
    etdata_gt_meta = np.concatenate(etdata_gt_meta)
    etdata_pr_meta = np.concatenate(etdata_pr_meta)
    _etdata_gt.load(etdata_gt_meta, **{'source':'array'})
    _etdata_pr.load(etdata_pr_meta, **{'source':'array'})
    ke, (evt_overlap, _evt_gt, _evt_pr) = eval_evt(_etdata_gt, _etdata_pr, len(classes), mode=eval_mode)

    #save only fixation, saccade and pso
    rez.append((_data['dataset'], _data['alg'], 'meta') + tuple(ke[:3]) )
    toc = time.time()
    print ("Done in %.2f seconds" %(toc-tic))
    print (rez_format.format(*list(ke[:3])))

    #save
    rez_df = pd.DataFrame(rez, columns=columns)
    rez_df.to_csv(fpath_rez, index=False)

