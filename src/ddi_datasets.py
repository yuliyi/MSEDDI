import math
import os
from collections import Counter

import torch
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import pickle
from tqdm import tqdm


def split_train_valid(args, triplets):
    train_split = ShuffleSplit(n_splits=2, test_size=args.valid_ratio)
    train_index, valid_index = next(iter(train_split.split(X=triplets)))
    return triplets[train_index], triplets[valid_index]


def del_tri(args, triplets, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict):
    vecs = []  # smiles vector (d1, d2)
    rels = []
    unique_dict = {}
    pairs = []
    kges = []
    weaves = []
    mpnns = []
    afps = []
    for sample in triplets:
        d1, d2, r = sample
        del_unique(d1, unique_dict, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict, kges, vecs, weaves, mpnns, afps)
        del_unique(d2, unique_dict, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict, kges, vecs, weaves, mpnns, afps)
        pair = [d1, d2]
        rels.append(int(r))
        pairs.append(pair)
    labels = label_binarize(rels, classes=np.arange(args.rel_total))

    return np.array(pairs, dtype=str), unique_dict, np.array(weaves), np.array(mpnns), np.array(afps), torch.FloatTensor(np.array(vecs)), torch.FloatTensor(np.array(kges)), torch.FloatTensor(np.array(labels))


def load_ddi_data_fold(args, train_triplets, test_triplets, fpt_dict, kge_dict, rotate_dict, complex_dict, vec_dict, gcn_dict, gat_dict, weave_dict, mpnn_dict, afp_dict):
    train_triplets, valid_triplets = split_train_valid(args, train_triplets)
    train_tup = del_tri(args, train_triplets, fpt_dict, kge_dict, rotate_dict, complex_dict, vec_dict, gcn_dict, gat_dict, weave_dict, mpnn_dict, afp_dict)
    valid_tup = del_tri(args, valid_triplets, fpt_dict, kge_dict, rotate_dict, complex_dict, vec_dict, gcn_dict, gat_dict, weave_dict, mpnn_dict, afp_dict)
    test_tup = del_tri(args, test_triplets, fpt_dict, kge_dict, rotate_dict, complex_dict, vec_dict, gcn_dict, gat_dict, weave_dict, mpnn_dict, afp_dict)
    return train_tup, valid_tup, test_tup


def split_drugs(drugA, drugB, label, event_num, cross_ver_tim):
    temp_drug1 = [[] for i in range(event_num)]
    temp_drug2 = [[] for i in range(event_num)]
    for i in range(len(label)):
        temp_drug1[label[i]].append(drugA[i])
        temp_drug2[label[i]].append(drugB[i])
    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drug1[i])):
            drug1, drug2 = temp_drug1[i][j], temp_drug2[i][j]
            drug_cro_dict[drug1] = j % cross_ver_tim
            drug_cro_dict[drug2] = j % cross_ver_tim
    old_drugs = [[] for i in range(cross_ver_tim)]
    new_drugs = [[] for i in range(cross_ver_tim)]
    for i in range(cross_ver_tim):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                new_drugs[i].append(dr_key)
            else:
                old_drugs[i].append(dr_key)
        # print(f'new:{len(new_drugs[i])}, old:{len(old_drugs[i])}')
    return old_drugs, new_drugs


def generate_pair_triplets(args, fold_i, ddi, old_drug_names, new_drug_names, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict):
    pos_triplets_train = []
    pos_triplets_s1 = []
    pos_triplets_s2 = []

    for id1, id2, relation in zip(ddi['Drug1'], ddi['Drug2'],  ddi['Label']):
        if relation == 95:  # db2:The serum concentration of the active metabolites increase
            pos_triplets_s2.append([id1, id2, relation])
        if id1 in new_drug_names:
            if id2 in new_drug_names:
                pos_triplets_s1.append([id1, id2, relation])
            else:
                pos_triplets_s2.append([id1, id2, relation])
        else:
            if id2 in new_drug_names:
                pos_triplets_s2.append([id1, id2, relation])
            else:
                pos_triplets_train.append([id1, id2, relation])
    print(f'train:{len(pos_triplets_train)}, s1:{len(pos_triplets_s1)}, s2:{len(pos_triplets_s2)}')
    save_data(old_drug_names, fold_i, 'old', args)
    save_data(new_drug_names, fold_i, 'new', args)
    return load_ddi_data_fold_cold_start(args, np.array(pos_triplets_train), np.array(pos_triplets_s1), np.array(pos_triplets_s2), kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict)


def save_data(data, fold_i, type, args):
    filename = f'{args.data_dir}/drug_folds/{args.dataset}_{type}_drug_ids_fold{fold_i}.csv'
    np.savetxt(filename, data, fmt='%s', delimiter=',')
    # print(f'Data saved as {filename}!\n')


def total_num_rel(name):
    if name.lower() == 'db1': return 65
    if name.lower() == 'db2': return 100
    else: raise NotImplementedError


def load_file(args, filename):
    filename = (f'{args.data_dir}/{args.dataset}_{filename}')
    with open(filename, 'rb') as f:
        data = pickle.load(f)  # item(drug_id, array(617d))
    # print(f'Loading data {filename}!\n')
    return data


def del_unique(drug_id, unique_dict, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict, kges, vecs, weaves, mpnns, afps):
    flag = unique_dict.get(drug_id, -1)
    if flag == -1:
        index = len(vecs)
        unique_dict[drug_id] = index
        kges.append(kge_dict[drug_id])
        vecs.append(vec_dict[drug_id])
        weaves.append(weave_dict[drug_id])
        mpnns.append(mpnn_dict[drug_id])
        afps.append(afp_dict[drug_id])


def load_ddi_data_fold_cold_start(args, pos_triplets_train, pos_triplets_s1, pos_triplets_s2, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict):

    train_triplets, valid_triplets = split_train_valid(args, pos_triplets_train)
    train_tup = del_tri(args, train_triplets, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict)
    valid_tup = del_tri(args, valid_triplets, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict)
    s1_tup = del_tri(args, pos_triplets_s1, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict)
    s2_tup = del_tri(args, pos_triplets_s2, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict)

    return train_tup, valid_tup, s1_tup, s2_tup
