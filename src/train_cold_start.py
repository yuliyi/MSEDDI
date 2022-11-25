# S1:new-new, S2:old-new
import copy
import csv
import random
from time import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.preprocessing import label_binarize
from torch import optim
import torch.nn.functional as F
import models
from tqdm import tqdm
from ddi_datasets import total_num_rel, load_file, split_drugs, generate_pair_triplets
from loss import ContrastiveLoss, KGELoss
from custom_metrics import evaluate, evaluate_train, evaluate_valid
import argparse
import logging
import os
import math
from collections import OrderedDict, Counter

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--data_dir', nargs='?', default='../data', help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='db1', help='dataset')
parser.add_argument('--ddi_file', nargs='?', default='triplets.csv', help='Input data path.')
parser.add_argument('--drugs_file', nargs='?', default='drugs.csv', help='Input data path.')
parser.add_argument('--vec_file', nargs='?', default='smiles_vec.pkl', help='Input data path.')
parser.add_argument('--kge_file', nargs='?', default='kges_transe.pkl', help='Input data path.')
parser.add_argument('--weave_file', nargs='?', default='weave_pcba.pkl', help='Input data path.')
parser.add_argument('--mpnn_file', nargs='?', default='mpnn_pcba.pkl', help='Input data path.')
parser.add_argument('--afp_file', nargs='?', default='afp_pcba.pkl', help='Input data path.')
parser.add_argument('--result_dir', nargs='?', default='../result', help='Input data path.')
parser.add_argument('--save_dir', nargs='?', default='../model', help='Input data path.')
parser.add_argument('--log_dir', nargs='?', default='../log', help='Input data path.')
parser.add_argument('--fold', type=int, default=5, help='Fold on which to train on')
parser.add_argument('--tri_batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--valid_ratio', type=float, default=0.2, help='valid ratio')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Lambda when calculating l2 loss.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--ddi_epoch', type=int, default=350, help='Number of epoch.')
parser.add_argument('--ddi_stop_steps', type=int, default=20, help='Number of epoch for early stopping')
parser.add_argument('--hid_dim', type=int, default=64, help='hidden dim')
parser.add_argument('--vec_dim', type=int, default=64, help='vec dim')
parser.add_argument('--kge_dim', type=int, default=400, help='kge dim')
parser.add_argument('--smiles_max_len', type=int, default=100, help='smiles max length')
parser.add_argument('--cuda', type=bool, default=True, help='GPU or CPU')
args = parser.parse_args()


def early_stopping(dataset, model, model_name, epoch, best_epoch, valid_metric, best_metric, bad_counter):
    if (model_name == 'KGE' and valid_metric < best_metric) or (model_name == 'DDI' and valid_metric < best_metric):
        best_metric = valid_metric
        bad_counter = 0
        save_model(dataset, model, args.save_dir, model_name, epoch)
        best_epoch = epoch
    else:
        bad_counter += 1
    return bad_counter, best_metric, best_epoch


def save_model(dataset, model, model_dir, model_name, current_epoch):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, '{}_{}_cold_start.pth'.format(dataset, model_name))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)


def load_model(dataset, model, model_dir, model_name):
    model_path = os.path.join(model_dir, '{}_{}_cold_start.pth'.format(dataset, model_name))
    checkpoint = torch.load(model_path, map_location=get_device(args))

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]  # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model


def get_device(args):
    args.gpu = False
    if torch.cuda.is_available() and args.cuda:
        args.gpu = True
        # print(f'Training on GPU.')
    else:
        print(f'Training on CPU.')
    device = torch.device("cuda:0" if args.gpu else "cpu")
    return device


# ----------------------------------------define log information--------------------------------------------------------

# create log information
def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def run_batch(model_ddi, data, criterion, device, batch_size, optimizer=None):
    total_loss = 0
    y_pred = []
    y_truth = []
    y_score = []

    triplets, unique_dict, weaves, mpnns, afps, vecs, kges, labels = data
    kge_np = kges.detach().cpu().numpy()
    vec_np = vecs.detach().cpu().numpy()

    batch_count = math.ceil(len(triplets) / batch_size)
    for batch_id in range(batch_count):
        label_batch = labels[batch_id * batch_size: (batch_id + 1) * batch_size]
        triplet_batch = triplets[batch_id * batch_size: (batch_id + 1) * batch_size]
        kge_batch = [[kge_np[unique_dict[d1]], kge_np[unique_dict[d2]]] for d1, d2 in triplet_batch]
        kge_batch = torch.FloatTensor(np.array(kge_batch)).to(device)
        vec_batch = [[vec_np[unique_dict[d1]], vec_np[unique_dict[d2]]] for d1, d2 in triplet_batch]
        vec_batch = torch.FloatTensor(np.array(vec_batch)).to(device)
        weave_batch = [[weaves[unique_dict[d1]], weaves[unique_dict[d2]]] for d1, d2 in triplet_batch]
        weave_batch = torch.FloatTensor(np.array(weave_batch)).to(device)
        mpnn_batch = [[mpnns[unique_dict[d1]], mpnns[unique_dict[d2]]] for d1, d2 in triplet_batch]
        mpnn_batch = torch.FloatTensor(np.array(mpnn_batch)).to(device)
        afp_batch = [[afps[unique_dict[d1]], afps[unique_dict[d2]]] for d1, d2 in triplet_batch]
        afp_batch = torch.FloatTensor(np.array(afp_batch)).to(device)

        scores = model_ddi(weave_batch, mpnn_batch, afp_batch, kge_batch, vec_batch)
        loss = criterion(scores, label_batch)
        ss = F.softmax(scores, dim=1).detach().cpu().numpy()
        pred = copy.deepcopy(ss)
        pred_l = np.argmax(pred, axis=1)
        pred = label_binarize(pred_l, classes=np.arange(labels.shape[1]))
        y_pred.append(pred)
        y_truth.append(label_batch.detach().cpu().numpy())
        y_score.append(ss)
        if model_ddi.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    total_loss /= batch_count
    y_pred = np.concatenate(y_pred, axis=0)
    y_truth = np.concatenate(y_truth, axis=0)
    y_score = np.concatenate(y_score, axis=0)

    return total_loss, (y_pred, y_truth, y_score)


def run_test_batch(model_ddi, data, criterion, device, batch_size):
    y_pred = []
    y_truth = []
    y_score = []
    y_embed = []
    triplets, unique_dict, weaves, mpnns, afps, vecs, kges, labels = data

    kge_np = kges.detach().cpu().numpy()
    vec_np = vecs.detach().cpu().numpy()
    total_loss = 0
    batch_count = math.ceil(len(triplets) / batch_size)
    for batch_id in range(batch_count):
        label_batch = labels[batch_id * batch_size: (batch_id + 1) * batch_size]
        triplet_batch = triplets[batch_id * batch_size: (batch_id + 1) * batch_size]
        kge_batch = [[kge_np[unique_dict[d1]], kge_np[unique_dict[d2]]] for d1, d2 in triplet_batch]
        kge_batch = torch.FloatTensor(np.array(kge_batch)).to(device)
        vec_batch = [[vec_np[unique_dict[d1]], vec_np[unique_dict[d2]]] for d1, d2 in triplet_batch]
        vec_batch = torch.FloatTensor(np.array(vec_batch)).to(device)
        weave_batch = [[weaves[unique_dict[d1]], weaves[unique_dict[d2]]] for d1, d2 in triplet_batch]
        weave_batch = torch.FloatTensor(np.array(weave_batch)).to(device)
        mpnn_batch = [[mpnns[unique_dict[d1]], mpnns[unique_dict[d2]]] for d1, d2 in triplet_batch]
        mpnn_batch = torch.FloatTensor(np.array(mpnn_batch)).to(device)
        afp_batch = [[afps[unique_dict[d1]], afps[unique_dict[d2]]] for d1, d2 in triplet_batch]
        afp_batch = torch.FloatTensor(np.array(afp_batch)).to(device)

        scores = model_ddi(weave_batch, mpnn_batch, afp_batch, kge_batch, vec_batch)
        loss = criterion(scores, label_batch)
        ss = F.softmax(scores, dim=1).detach().cpu().numpy()
        pred = copy.deepcopy(ss)
        pred_l = np.argmax(pred, axis=1)
        pred = label_binarize(pred_l, classes=np.arange(labels.shape[1]))
        y_pred.append(pred)
        y_truth.append(label_batch.detach().cpu().numpy())
        y_score.append(ss)
        # y_embed.append(scores.detach().cpu().numpy())
        total_loss += loss.item()
    total_loss /= batch_count
    y_pred = np.concatenate(y_pred, axis=0)
    y_truth = np.concatenate(y_truth, axis=0)
    y_score = np.concatenate(y_score, axis=0)
    # y_embed = np.concatenate(y_embed, axis=0)

    return total_loss, y_pred, y_truth, y_score, y_embed


def save_result(args, task, result_type, result, header):
    filename = f'{args.result_dir}/{args.dataset}_{result_type}_{task}.csv'
    np.savetxt(filename, result, delimiter=",", header=header)


def deal_test(args, model_ddi, task, tup, criterion, device, pred_list, truth_list, score_list, embed_list, time_ddi, flag=True):
    loss, pred, truth, score, embed = run_test_batch(model_ddi, tup, criterion, device, args.tri_batch_size)

    if flag:
        # print(f'pred:{pred.shape}, truth:{truth.shape}, score:{score.shape}')
        pred_list = np.row_stack((pred_list, pred))
        truth_list = np.row_stack((truth_list, truth))
        score_list = np.row_stack((score_list, score))
        # embed_list = np.row_stack((embed_list, embed))
        embed_list = None
    result_all = evaluate_train(pred, truth, score, args.rel_total)
    logging.info(f'Final DDI_{task} Evaluation: best_epoch {best_ddi_epoch:04d}, total time: {time() - time_ddi:.1f}, loss: {loss:.4f}, acc: {result_all[0]:.4f}, aupr_mi: {result_all[1]:.4f}, auroc_mi: {result_all[2]:.4f}')
    return pred_list, truth_list, score_list, embed_list


def deal_result(task, pred_list, truth_list, score_list, embed_list):
    save_result(args, task, "pred_list", pred_list, "")
    save_result(args, task, "truth_list", truth_list, "")
    save_result(args, task, "score_list", score_list, "")
    # np.save(f'{args.result_dir}/{args.dataset}_embed_list_{task}', embed_list)
    print(f'pred:{pred_list.shape}, truth:{truth_list.shape}, score:{score_list.shape}')
    result_all, result_eve = evaluate(pred_list, truth_list, score_list, args.rel_total)
    save_result(args, task, "result_all", result_all, "acc, aupr_micro, auc_micro, f1_macro, precision_macro, recall_macro")
    save_result(args, task, "result_each", result_eve, "acc, aupr, auc, f1, precision, recall")


# seed
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
#cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = get_device(args)
log_save_id = create_log_id(args.log_dir)

# set log file
logging_config(folder=args.log_dir, name='log{:d}'.format(log_save_id), no_console=False)
logging.info(args)

args.rel_total = total_num_rel(name=args.dataset)
kge_dict = load_file(args, args.kge_file)
vec_dict = load_file(args, args.vec_file)
weave_dict = load_file(args, args.weave_file)
mpnn_dict = load_file(args, args.mpnn_file)
afp_dict = load_file(args, args.afp_file)
ddi = pd.read_csv(f'{args.data_dir}/{args.dataset}_{args.ddi_file}')
train_drugs, test_drugs = split_drugs(ddi["Drug1"], ddi["Drug2"], ddi["Label"], args.rel_total, args.fold)

time0 = time()
valid_truth_list = np.zeros((0, args.rel_total), dtype=float)
valid_score_list = np.zeros((0, args.rel_total), dtype=float)
valid_pred_list = np.zeros((0, args.rel_total), dtype=float)
s1_truth_list = np.zeros((0, args.rel_total), dtype=float)
s1_score_list = np.zeros((0, args.rel_total), dtype=float)
s1_pred_list = np.zeros((0, args.rel_total), dtype=float)
s1_embed_list = np.zeros((0, args.rel_total), dtype=float)
s2_truth_list = np.zeros((0, args.rel_total), dtype=float)
s2_score_list = np.zeros((0, args.rel_total), dtype=float)
s2_pred_list = np.zeros((0, args.rel_total), dtype=float)
s2_embed_list = np.zeros((0, args.rel_total), dtype=float)


for fold_i in range(args.fold):
    old_drugs, new_drugs = train_drugs[fold_i], test_drugs[fold_i]
    fold_i += 1
    print(f'Fold {fold_i} splited!')
    train_tup, valid_tup, s1_tup, s2_tup = generate_pair_triplets(args, fold_i, ddi, old_drugs, new_drugs, kge_dict, vec_dict, weave_dict, mpnn_dict, afp_dict)
    train_tup = [t.to(device) if i > 4 else t for i, t in enumerate(train_tup)]
    valid_tup = [t.to(device) if i > 4 else t for i, t in enumerate(valid_tup)]
    s1_tup = [t.to(device) if i > 4 else t for i, t in enumerate(s1_tup)]
    s2_tup = [t.to(device) if i > 4 else t for i, t in enumerate(s2_tup)]

    model_kge = models.SMILESformer(d_model=args.vec_dim, dropout=0.3, nhead=1, dim_feedforward=args.vec_dim * 2,
                                    num_encoder_layers=1, num_decoder_layers=1, seq_len=args.kge_dim)
    model_ddi = models.DDIModel(args)
    model_ddi.to(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_contrast = ContrastiveLoss()
    optimizer = optim.Adam(model_ddi.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    bad_ddi_counter = 0
    best_loss = 1000
    best_acc = 0
    best_ddi_epoch = 0
    time_ddi = time()
    for epoch_i in range(1, args.ddi_epoch + 1):
        model_ddi.train()
        # metrics: acc, auroc, f1_score, precision, recall, aupr, ap
        train_loss, train_metrics = run_batch(model_ddi, train_tup, criterion, device, args.tri_batch_size, optimizer)
        model_ddi.eval()
        with torch.no_grad():
            valid_loss, valid_metrics = run_batch(model_ddi, valid_tup, criterion, device, args.tri_batch_size)
            result_train = evaluate_train(*train_metrics, args.rel_total)
            result_valid = evaluate_train(*valid_metrics, args.rel_total)
            if epoch_i % 10 == 0:
                deal_test(args, model_ddi, "s2", s2_tup, criterion, device, s2_pred_list,
                          s2_truth_list, s2_score_list, s2_embed_list, time_ddi, False)
                deal_test(args, model_ddi, "s1", s1_tup, criterion, device, s1_pred_list,
                          s1_truth_list, s1_score_list, s1_embed_list, time_ddi, False)
        logging.info('DDI Training: Folder {:04d} | Epoch {:04d} | train_acc {:.4f} | valid_acc {:.4f} | '
                     'valid_aupr {:.4f} | valid_auc {:.4f} | valid_loss {:.4f}'.format(
            fold_i, epoch_i, result_train[0], result_valid[0], result_valid[1], result_valid[2], valid_loss))
        bad_ddi_counter, best_loss, best_ddi_epoch = early_stopping(args.dataset, model_ddi, 'DDI', epoch_i, best_ddi_epoch, valid_loss, best_loss, bad_ddi_counter)
        if bad_ddi_counter >= args.ddi_stop_steps or epoch_i == args.ddi_epoch:
            valid_pred_list = np.row_stack((valid_pred_list, valid_metrics[0]))
            valid_truth_list = np.row_stack((valid_truth_list, valid_metrics[1]))
            valid_score_list = np.row_stack((valid_score_list, valid_metrics[2]))
            model_ddi = load_model(args.dataset, model_ddi, args.save_dir, 'DDI')
            model_ddi.eval()
            with torch.no_grad():  # unique_dict, torch.FloatTensor(vecs), torch.LongTensor(key_padding_mask), rels, torch.FloatTensor(gnn_embeddings)
                s2_pred_list, s2_truth_list, s2_score_list, s2_embed_list = deal_test(args, model_ddi, "s2", s2_tup, criterion, device, s2_pred_list, s2_truth_list, s2_score_list, s2_embed_list, time_ddi)
                s1_pred_list, s1_truth_list, s1_score_list, s1_embed_list = deal_test(args, model_ddi, "s1", s1_tup, criterion, device, s1_pred_list, s1_truth_list, s1_score_list, s1_embed_list, time_ddi)
                break

logging.info(f'Final total time: {time() - time0:.1f}')
valid_result_all = evaluate_valid(valid_pred_list, valid_truth_list, valid_score_list, args.rel_total)
save_result(args, "valid", "result_all", valid_result_all, "acc, aupr_micro, auc_micro, f1_macro, precision_macro, recall_macro")
deal_result("s2", s2_pred_list, s2_truth_list, s2_score_list, s2_embed_list)
deal_result("s1", s1_pred_list, s1_truth_list, s1_score_list, s1_embed_list)
