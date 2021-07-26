'''
<train.py> 2021-04-30 by Weby
Test CGINet
'''

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from sklearn import metrics

from model.model import CGINet
from utility.prepare import DataPreparation
from utility.classer import logger
from utility.function import test


parser = argparse.ArgumentParser(description='CGINet Test')
### device
parser.add_argument('--device', default=None, type=int, help='GPU id')
### path
parser.add_argument('--log_path', default="../out/log/", help='')
parser.add_argument('--log_file', default="te_log", help='')
parser.add_argument('--adj_pkl_path', default="../data/adj-pkl-all/", help='')
parser.add_argument('--edg_pkl_path', default="../data/edg-pkl-all/", help='')
parser.add_argument('--parameter_path', default="../out/param/", help='')
### model
parser.add_argument('--epoch', type=int, default=10, help='')
parser.add_argument('--hid_dims', type=int, nargs='+', default=[128,64,32,16], help="")
### metrics
parser.add_argument('--te_batch', default=True, type=bool, help='test the data per batch')
parser.add_argument('--te_batch_size', type=int, default=2048, help='test batch size')
parser.add_argument('--ap_k', type=int, default=20, help='k for apk')


def main():
    ### args
    args = parser.parse_args()
    ### GPU of CPU
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### log
    log = logger(args.log_path + args.log_file + "_" + str(args.device))

    log.info("Use {} for testing".format(args.device))
    ### paratmeter printing
    for k, v in args.__dict__.items():
        log.info("{}: {}".format(k, v))
    log.info(">" * 90)

    ### data preparation
    dp = DataPreparation(args)

    best_auroc = 0.
    best_auprc = 0.
    best_apk = 0.
    best_epoch = 0
    for epo_idx in range(args.epoch):

        ### load model
        net = CGINet(dp, args, log)
        net.load_state_dict(torch.load(args.parameter_path + "parameter_0_e{}.pkl".format(epo_idx+1)))
        net.to(args.device)

        ### test
        auroc_tmp, auprc_tmp, apk_tmp = test("te", epo_idx, net, dp, args, log)

        ### best test model
        if auprc_tmp >= best_auprc:
            best_auroc = auroc_tmp
            best_auprc = auprc_tmp
            best_apk = apk_tmp
            best_epoch = epo_idx

        log.info(">>>>>> Epoch: [{}/{}] testing finished !!! <<<<<<\n".format(epo_idx+1, args.epoch))

    ### print best result
    log.info(">"*90)
    log.info("The best val epoch: {} | avg AUROC {:.5f} | avg AUPRC: {:.5f} | avg AP@{}: {:.5f}".format(best_epoch+1, best_auroc, best_auprc, args.ap_k, best_apk))
    log.info(">"*90)



if __name__ == '__main__':
    main()
