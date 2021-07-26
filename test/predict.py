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


parser = argparse.ArgumentParser(description='CGINet Predict')
### device
parser.add_argument('--device', default=None, type=int, help='GPU id')
### path
parser.add_argument('--log_path', default="../out/log/", help='')
parser.add_argument('--log_file', default="pr_log", help='')
parser.add_argument('--adj_pkl_path', default="../data/adj-pkl/", help='')
parser.add_argument('--edg_pkl_path', default="../data/edg-pkl/", help='')
parser.add_argument('--parameter_path', default="../out/param/", help='')
### model
parser.add_argument('--epo_idx', type=int, default=10, help='')
parser.add_argument('--hid_dims', type=int, nargs='+', default=[128, 64, 32, 16], help="")


def main():
    ### args
    args = parser.parse_args()
    ### GPU of CPU
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### log
    log = logger(args.log_path + args.log_file + "_" + str(args.device))

    log.info("Use {} for predicting".format(args.device))
    ### paratmeter printing
    for k, v in args.__dict__.items():
        log.info("{}: {}".format(k, v))
    log.info(">" * 90)

    ### data preparation
    dp = DataPreparation(args)

    ### load model
    net = CGINet(dp, args, log)
    net.load_state_dict(torch.load(args.parameter_path + "parameter_cpu_e{}.pkl".format(args.epo_idx + 1)))
    net.to(args.device)









if __name__ == '__main__':
    main()