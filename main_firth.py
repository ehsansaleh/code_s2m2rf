import torch
import numpy as np
import torch.optim
import json
import torch.utils.data.sampler
import os
import random
import time
import pandas as pd
import tarfile
import hashlib
import io
import socket
import itertools
import data.feature_loader as feat_loader
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict
from collections import defaultdict
from typing import Dict, Any
from pathlib import Path

import argparse

PROJPATH = os.getcwd()
my_parser = argparse.ArgumentParser()
my_parser.add_argument('--configid', action='store', type=str, required=True)
my_parser.add_argument('--proc_rank', action='store', type=int, required=True,
                       help="The rank among the parallel pool of workers")
my_parser.add_argument('--proc_size', action='store', type=int, required=True,
                       help="The size of the parallel pool of workers")
args = my_parser.parse_args()
config_id = args.configid

proc_rank = args.proc_rank
proc_size = args.proc_size

should_print = (proc_rank == 0)
cfg_path = f'{PROJPATH}/configs/{config_id}.json'
if should_print:
    print(f'Reading Configuration from {cfg_path}', flush=True)

cfg_dir = f'{PROJPATH}/configs'
os.makedirs(cfg_dir, exist_ok=True)

results_dir = f'{PROJPATH}/results/{config_id}/'
Path(results_dir).mkdir(parents=True, exist_ok=True)

storage_dir = f'{PROJPATH}/storage/{config_id}/'
Path(storage_dir).mkdir(parents=True, exist_ok=True)

with open(cfg_path) as f:
    config_dict = json.load(f)
    start_seed = config_dict['start_seed']
    num_seeds = config_dict['num_seeds']
    n_shot_list = config_dict['n_shot_list']
    n_way_list = config_dict['n_way_list']
    split_list = config_dict['split_list']
    dataset_list = config_dict['dataset_list']
    method_list = config_dict['method_list']
    model_list = config_dict['model_list']
    firth_c_list = config_dict['firth_c_list']
    iter_num_input = config_dict['iter_num']
    torch_threads = config_dict['torch_threads']
    fine_tune_epochs = config_dict['fine_tune_epochs']
    force_s2m2r_features = config_dict.get('force_s2m2r_features', False)
    store_extra = False

rng_seed_list = list(range(start_seed, start_seed + num_seeds))
csv_path = f'{results_dir}/{config_id}_r{proc_rank}.csv'
tar_path = f'{storage_dir}/{config_id}_r{proc_rank}.tar'
torch.set_num_threads(torch_threads)


def feature_evaluation(cl_data_file, model, n_way=5,
                       n_support=5, n_query=15,
                       adaptation=False, **forward_kwargs):
    class_list = cl_data_file.keys()

    class_list_ok = [cl for cl in class_list if len(cl_data_file[cl]) >= (n_support+n_query)]
    if len(class_list_ok) == class_list:
        class_list_ok = class_list  # No changes

    select_class = random.sample(class_list_ok, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[ii]]) for ii in range(n_support+n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True, **forward_kwargs)
    else:
        scores = model.set_forward(z_all, is_feature=True, **forward_kwargs)

    acc = []
    for each_score in scores:
        pred = each_score.data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(n_way), n_query)
        acc.append(np.mean(pred == y)*100)
    return acc


def dict_hash(dictionary: Dict[str, Any]) -> str:
    # MD5 hash of a dictionary.
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


cntr = -1
for setting in itertools.product(rng_seed_list, dataset_list, split_list, method_list,
                                 model_list, n_shot_list, n_way_list, firth_c_list):
    cntr += 1
    if cntr % proc_size != proc_rank:
        continue

    (rng_seed, params_dataset, params_split, params_method, params_model,
     params_n_shot, params_n_way, firth_c) = setting

    if (params_split == 'val') and (params_n_way > 16) and (params_dataset in ['miniImagenet', 'cifar']):
        n_ways_ = 16
    elif (params_split == 'val') and (params_n_way > 97) and (params_dataset == 'tieredImagenet'):
        n_ways_ = 97
    else:
        n_ways_ = params_n_way

    params_train_n_way = n_ways_
    params_test_n_way = n_ways_

    if iter_num_input is None:
        if params_dataset == 'CUB':
            iter_num = 600
        else:
            iter_num = 10000
    else:
        iter_num = iter_num_input

    # Do not touch the following
    params_save_iter = -1
    params_adaptation = False
    params_train_aug = False

    results_dict = defaultdict(list)
    props_dict = dict()
    props_dict['firth_c'] = firth_c
    props_dict['max_iter_num'] = iter_num
    props_dict['n_shot'] = params_n_shot
    props_dict['train_n_way'] = params_n_way
    props_dict['test_n_way'] = params_n_way
    props_dict['dataset'] = params_dataset
    props_dict['method'] = params_method
    props_dict['model'] = params_model
    props_dict['split'] = params_split
    props_dict['seed'] = rng_seed
    props_dict['fine_tune_epochs'] = fine_tune_epochs
    rand_postfix = dict_hash(props_dict)[:16]
    npz_filename = f'{config_id}_{rand_postfix}.npz'
    props_dict['npz_filename'] = npz_filename
    props_dict['hostname'] = socket.gethostname()[:7]

    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    acc_all = []
    few_shot_params = dict(n_way=params_test_n_way, n_support=params_n_shot)

    if params_method == 'S2M2_R':
        model = BaselineFinetune(model_dict[params_model], **few_shot_params)
    elif params_method == 'ProtoNet':
        model = ProtoNet(model_dict[params_model], **few_shot_params)
    elif params_method == 'MatchingNet':
        model = MatchingNet(model_dict[params_model], **few_shot_params)
    elif params_method == 'RelationNet':
        model = RelationNet(model_dict[params_model], **few_shot_params)
    elif params_method == 'MAML':
        model = MAML(model_dict[params_model], **few_shot_params)
    else:
        raise Exception(f'Unknown method: {params_method}')

    if torch.cuda.is_available():
        model = model.cuda()

    checkpoint_dir = './checkpoints/%s/%s_%s' % (params_dataset, params_model,
                                                 params_method if not force_s2m2r_features else 'S2M2_R')

    split = params_split
    if params_save_iter != -1:
        split_str = split + "_" + str(params_save_iter)
    else:
        split_str = split

    novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split_str + ".hdf5")
    cl_data_file = feat_loader.init_loader(novel_file)

    acc_all1, acc_all2, acc_all3 = [], [], []

    if params_dataset == 'CUB':
        n_query = 15
    else:
        n_query = 600 - params_n_shot

    if should_print:
        print(novel_file, flush=True)
        print("evaluating over %d examples" % n_query, flush=True)

    st_time = time.time()
    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query=n_query, adaptation=params_adaptation,
                                 fine_tune_epochs=fine_tune_epochs, firth_c=firth_c, **few_shot_params)
        if len(acc) < 3:
            acc = acc + [acc[-1]] * (3-len(acc))

        acc_all1.append(acc[0])
        acc_all2.append(acc[1])
        acc_all3.append(acc[2])

        for key, val in props_dict.items():
            results_dict[key].append(val)
        results_dict['iter'].append(i)
        results_dict['acc_100'].append(acc[0])
        results_dict['acc_200'].append(acc[1])
        results_dict['acc_300'].append(acc[2])

        if i % 10 == 0:
            if should_print:
                sec_iter = (time.time() - st_time) / (i+1)
                print("%d steps reached and the mean acc is %g , %g , %g (%.3f sec/iter)" %
                      (i, np.mean(np.array(acc_all1)), np.mean(np.array(acc_all2)),
                       np.mean(np.array(acc_all3)), sec_iter), flush=True)
        if (i+1) % 100 == 0:
            # Let's flush the results out to the csv file
            some_rows_df = pd.DataFrame(results_dict)
            if os.path.exists(csv_path):
                old_df = pd.read_csv(csv_path)
                new_df = old_df.append(some_rows_df, ignore_index=True)
                new_df.to_csv(csv_path, index=False)
            else:
                some_rows_df.to_csv(csv_path, index=False)
            # Emptying the can for the next rounds
            results_dict = defaultdict(list)

    acc_mean1 = np.mean(acc_all1)
    acc_mean2 = np.mean(acc_all2)
    acc_mean3 = np.mean(acc_all3)
    acc_std1 = np.std(acc_all1)
    acc_std2 = np.std(acc_all2)
    acc_std3 = np.std(acc_all3)
    acc_ci1 = 1.96 * acc_std1 / np.sqrt(iter_num)
    acc_ci2 = 1.96 * acc_std2 / np.sqrt(iter_num)
    acc_ci3 = 1.96 * acc_std2 / np.sqrt(iter_num)
    if should_print:
        print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, acc_ci1), flush=True)
        print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, acc_ci2), flush=True)
        print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, acc_ci3), flush=True)

    # Storing the extra stuff
    if store_extra:
        tempf_ = io.BytesIO()
        np.savez(tempf_, acc_all1=acc_all1, acc_all2=acc_all2, acc_all3=acc_all3,
                 **{key: np.array(val) for key, val in results_dict.items()})
        tempf_.seek(0)
        archive = tarfile.open(tar_path, "a")
        info = tarfile.TarInfo(name=npz_filename)
        tempf_.seek(0, io.SEEK_END)
        info.size = tempf_.tell()
        info.mtime = time.time()
        tempf_.seek(0, io.SEEK_SET)
        archive.addfile(info, tempf_)
        archive.close()
        tempf_.close()
