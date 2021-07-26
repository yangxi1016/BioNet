'''
<train.py> 2021-04-30 by Weby
Train CGINet
'''

import os
import sys
import argparse
import time
import datetime
import pickle
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import CGINet
from utility.prepare import DataPreparation
from utility.classer import BucketSampler, HingeLoss, logger
from utility.function import mkdir, test

parser = argparse.ArgumentParser(description='CGINet Train')

### GPU setting
parser.add_argument('--use_gpu', default=True, type=bool, help='gpu or cpu')
parser.add_argument("--gpu_devices", default=[0,1,2,3], type=int, nargs='+', help="")
parser.add_argument('--num_workers', default=None, type=int, help='')
parser.add_argument('--device', default=None, type=int, help='GPU id')
### distributed setting
parser.add_argument('--dist_backend', default='nccl', type=str, help='')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:12345', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='node number')
parser.add_argument('--node_id', default=0, type=int, help='current node id')
### path & data setting
parser.add_argument('--log_path', default="../out/log/", help='')
parser.add_argument('--log_file', default="tr_log", help='')
parser.add_argument('--adj_pkl_path', default="../data/adj-pkl-all/", help='')
parser.add_argument('--edg_pkl_path', default="../data/edg-pkl-all/", help='')
parser.add_argument('--parameter_path', default="../out/param/", help='')
parser.add_argument('--seed', default=20210505, type=int, help='random seed')
parser.add_argument('--distortion', default=0., help='negative sampling distortion')
### model parameters
parser.add_argument('--epoch', type=int, default=20, help='')
parser.add_argument('--save_all', default=True, type=bool, help='save all val model')
parser.add_argument('--batch_size', type=int, default=130, help='')
parser.add_argument('--print_step', type=int, default=250, help='')
parser.add_argument('--dropout', default=0.1, help='')
parser.add_argument('--hid_dims', type=int, nargs='+', default=[128,64,32,16], help="")
parser.add_argument('--lr', default=1e-4, help='optimizer learning rate')
parser.add_argument('--margin', default=0.1, help='hinge loss max margin')
### test & metrics
parser.add_argument('--te_batch', default=True, type=bool, help='test the data per batch')
parser.add_argument('--te_batch_size', type=int, default=2048, help='test batch size')
parser.add_argument('--ap_k', type=int, default=20, help='k for apk')


### main
def main():

    ### pytorch setting
    torch.autograd.set_detect_anomaly(True)  # enable anomaly detection to find the operation that failed to compute its gradient
    ### args
    args = parser.parse_args()
    ### make path
    mkdir(args.log_path)
    mkdir(args.parameter_path)

    if args.use_gpu:
        ### GPU
        gpu_devices = ','.join([str(i) for i in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node * args.world_size
        args.num_workers = args.world_size * 6
        ### multiprocess
        mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.num_workers = 1
        worker(None, None, args)


### worker
def worker(wid, ngpus_per_node, args):

    ### device setting
    if args.use_gpu:
        args.device = wid  # wid: worker id
        torch.cuda.set_device(args.device)
        rank = args.node_id * ngpus_per_node + args.device  # process id
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=rank)
    else:
        args.device = "cpu"

    ### log
    log = logger(args.log_path + args.log_file + "_" + str(args.device))
    log.info(">"*90)
    log.info("Use {} for training".format(args.device))
    ### paratmeter printing
    for k, v in args.__dict__.items():
        log.info("{}: {}".format(k, v))

    ### data preparation
    dp = DataPreparation(args)
    n_chems = dp.n_chems
    n_genes = dp.n_genes
    n_paths = dp.n_paths
    log.info("Node num: n_chems->{}, n_genes->{}, n_paths->{}".format(n_chems, n_genes, n_paths))
    n_rel_type_s1 = sum(dp.rt_num_dict_s1.values())
    n_rel_type_s2 = sum(dp.rt_num_dict_s2.values())
    log.info("Relation type num: s1->{}, s2->{}, all->{}".format(n_rel_type_s1, n_rel_type_s2, (n_rel_type_s1+n_rel_type_s2)))

    ### make model
    net = CGINet(dp, args, log)
    net.to(args.device)
    ### DistributedDataParallel
    if args.use_gpu:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.device], find_unused_parameters=True)
    ### parameters
    params_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info("Model parameters number: {}".format(params_num))

    ### load train edges pkl
    tr_edges_pkl = pickle.load(open(args.edg_pkl_path + "tr_edges.pkl", "rb"))

    #### loss, optimizer
    criterion = HingeLoss(args.margin)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    train(net, optimizer, criterion, tr_edges_pkl, dp, args, log)


### train
def train(net, optimizer, criterion, tr_edges_pkl, dp, args, log):
    log.info(">"*90)
    train_start = time.time()

    best_auroc = 0.
    best_auprc = 0.
    best_apk = 0.
    best_epoch = 0

    ### epoch
    for epo_idx in range(args.epoch):
        ### random seed
        np.random.seed(args.seed + epo_idx)

        ### train sampler, train loader
        tr_edges, rc_lens_list = dp.prepare_edges(tr_edges_pkl)
        tr_sampler = BucketSampler(dataset=tr_edges,
                                   rc_lens_list=rc_lens_list,
                                   batch_size=args.batch_size,
                                   is_dist=args.use_gpu)
        tr_loader = DataLoader(dataset=tr_edges,
                               batch_size=args.batch_size,
                               shuffle=(tr_sampler is None),
                               num_workers=args.num_workers,
                               sampler=tr_sampler,
                               pin_memory=True)

        ### train setting
        net.train()
        dp.is_train = True
        dp.dropout = args.dropout  # update dropout for training

        ### begin training
        epoch_start = time.time()
        train_loss = 0.
        for bt_idx, (tr_edges, tr_labels) in enumerate(tr_loader):
            bt_start = time.time()

            ### tr_edges
            tr_edges = tr_edges.to(args.device)
            tr_labels = tr_labels.to(args.device)
            i, j, k = dp.idx2rc_dict_s2[tr_labels[0].item()]  # current relation type & relation category

            ### inputs
            inputs = ((i, j), k, tr_edges)  # rel_type, rt_k, edges
            ### model
            preds_pos, preds_neg = net(inputs)
            ### compute loss
            loss = criterion(preds_pos, preds_neg)

            ### back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### average loss
            train_loss = train_loss + loss.item()
            bt_time = time.time() - bt_start

            if bt_idx % args.print_step == 0:
                log.info("Epoch: [{}/{}] | batch: [{}/{}] | type: [{},{},{}] |"
                         " train loss: {:.3f} |"
                         " batch time: {:.3f}s".format(
                    epo_idx+1, args.epoch, bt_idx+1, len(tr_loader), i, j, k,
                    train_loss / (bt_idx+1),
                    bt_time))

        ### val
        epoch_time = time.time() - epoch_start
        auroc_tmp, auprc_tmp, apk_tmp = test("va", epo_idx, net, dp, args, log)
        log.info(">>>>>> Epoch: [{}/{}] training finished !!! | epoch time: {:.3f}s <<<<<<\n".format(epo_idx + 1, args.epoch, epoch_time))

        ### best model
        if auprc_tmp >= best_auprc:
            best_auroc = auroc_tmp
            best_auprc = auprc_tmp
            best_apk = apk_tmp
            best_epoch = epo_idx
        ### save model
        if args.save_all:
            torch.save(net.state_dict(), args.parameter_path + "parameter_{}_e{}.pkl".format(args.device, epo_idx+1))
        else:
            if auprc_tmp >= best_auprc:
                torch.save(net.state_dict(), args.parameter_path + "parameter_best.pkl")


    ### print result
    log.info(">"*90)
    ### all training time
    train_time = time.time() - train_start
    # train_time = datetime.timedelta(seconds=train_time)
    log.info("All train & val time: {:.3f}s".format(train_time))
    ### save path
    log.info("The model(s) is(are) saved in: {}".format(args.parameter_path))
    ### print best result
    log.info("The best val epoch: {} | avg AUROC {:.5f} | avg AUPRC: {:.5f} | avg AP@{}: {:.5f}".format(best_epoch, best_auroc, best_auprc, args.ap_k, best_apk))
    log.info(">"*90)


if __name__ == '__main__':
    main()



