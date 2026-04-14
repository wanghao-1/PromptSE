from __future__ import division
from __future__ import print_function

import os
import logging
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import load_data, get_links, dse_normalize, validation, save_all
from model_MLDSP import DSEModel
import warnings
import math
import time


#Disable warnings and enable anomaly detection
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/data/', help='path to data')
parser.add_argument('--result_path', type=str, default='/result/', help='path to result')
parser.add_argument('--model_path', type=str, default='/model/', help='path to save model')
parser.add_argument('--log_dir', nargs='?', default='/log', help='Input data path.')
parser.add_argument('--D_n', type=int, default=1020, help='number of drug node')
parser.add_argument('--S_n', type=int, default=5599, help='number of side-effect node')
parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hid_dim', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--fpt_dim', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--vec_dim', type=int, default=300, help='Number of hidden units.')
parser.add_argument('--gnn_dim', type=int, default=617, help='Number of hidden units.')
parser.add_argument('--vec_len', type=int, default=100, help='Number of hidden units.')
parser.add_argument('--kge_dim', type=int, default=400, help='Number of hidden units.')                 
parser.add_argument('--bio_dim', type=int, default=768, help='Number of hidden units.')
parser.add_argument('--alpha', type=float, default=0.02, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--min_indices_num', type=int, default=1020, help='min_indices_num')
parser.add_argument('--kk', type=int, default=2)

number=101

# ----------------------------------------define log information--------------------------------------------------------


def get_edge_index_and_weights(adj_matrix):
    row, col = np.nonzero(adj_matrix)
    edge_index = np.vstack((row, col))
    edge_weights = adj_matrix[row, col]
    
    return edge_index, edge_weights


def process_matrix(martix, k):

    row_nonzero = np.count_nonzero(martix, axis=1)
    sorted_indices = np.argsort(row_nonzero)[::-1]
    martix = martix[sorted_indices][:, sorted_indices]

    num = martix.shape[0]
    ones_matrix = np.ones((num, num))
    lower_triangular_ones = np.tril(ones_matrix)
    matrix = np.zeros((num, num))

    group_size_ratio = 1 / k
    start_index = 0
    while start_index < num:
        end_index = min(num, start_index + math.ceil(group_size_ratio * num))
        matrix[start_index:end_index, start_index:end_index] = 1
        start_index = end_index

    value_matrix = lower_triangular_ones + matrix
    value_matrix[value_matrix != 0] = 1
    martix = martix * value_matrix

    reverse_indices = np.argsort(sorted_indices)
    martix = martix[reverse_indices][:, reverse_indices]

    return martix.T



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

def get_device(args):
    args.gpu = False
    if torch.cuda.is_available() and args.cuda:
        args.gpu = True
        print(f'Training on GPU.')
    else:
        print(f'Training on CPU.')
    device = torch.device("cuda:0" if args.gpu else "cpu")
    return device  



def train(model, optimizer, 
          mask, target, train_idx, train_set,epoch): 
    model.train()
    optimizer.zero_grad()
    outputs, drug_embeddings , se_embeddings= model()
    output = torch.flatten(torch.mul(mask, outputs))
    loss_train = loss_m(output, target)
    loss_train.backward()
    optimizer.step()
    output = output[train_idx]
    noutput = torch.sigmoid(output).cpu().detach().numpy()
    metrics = validation(noutput, train_set)
    return loss_train.data.item(), metrics[0], metrics[1], outputs, drug_embeddings , se_embeddings

def compute_test(test_set, outputs, mask, test_idx, flag=False):
    output = torch.flatten(torch.mul(mask, outputs))[test_idx]
    noutput = torch.sigmoid(output).cpu().detach().numpy()
    metrics = validation(noutput, test_set, flag)
    return metrics

def get_twohop(data_set):
    data_set=data_set.reshape(args.D_n,args.S_n)
    data_set_to_drug_drug=np.dot(data_set, data_set.T)
    data_set_to_se_se=np.dot(data_set.T, data_set)
    data_set_to_drug_drug= torch.tensor(data_set_to_drug_drug)
    data_set_to_drug_drug = data_set_to_drug_drug.cuda()
    data_set_to_se_se= torch.tensor(data_set_to_se_se)
    data_set_to_se_se = data_set_to_se_se.cuda()
    D_u=data_set_to_drug_drug.sum(1)
    D_i=data_set_to_drug_drug.sum(0)

    user_size,item_size=args.D_n,args.D_n
    for i in range(user_size):
        if D_u[i]!=0:
            D_u[i]=1/D_u[i].sqrt()
    
    for i in range(item_size):
        if D_i[i]!=0:
            D_i[i]=1/D_i[i].sqrt()
    #(D_u)^{-0.5}*rate_matrix*(D_i)^{-0.5}
    data_set_to_drug_drug=D_u.unsqueeze(1)*data_set_to_drug_drug*D_i
   
    D_u1=data_set_to_se_se.sum(1)
    D_i1=data_set_to_se_se.sum(0)

    user_size1,item_size1=args.S_n,args.S_n
    for i in range(user_size1):
        if D_u1[i]!=0:
            D_u1[i]=1/D_u1[i].sqrt()
    
    for i in range(item_size1):
        if D_i1[i]!=0:
            D_i1[i]=1/D_i1[i].sqrt()
    #(D_u)^{-0.5}*rate_matrix*(D_i)^{-0.5}
    data_set_to_se_se=D_u1.unsqueeze(1)*data_set_to_se_se*D_i1
    return data_set_to_drug_drug,data_set_to_se_se

#--------------------------------main execution--------------------------------------
number
args = parser.parse_args()

log_save_id = create_log_id(args.log_dir)
logging_config(folder=args.log_dir, name='log{:d}'.format(number), no_console=False)
logging.info(args)
device = get_device(args)

#seed
seed = args.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


#------------------------------------data loading-------------------------------------
#Load all features and dataset
fpt_feature, mpnn_feature, weave_feature, afp_feature, nf_feature, vec_feature, llm_feature, w2v_feature, checkpoint = load_data()
data_set = get_links(path=args.data_path, dataset="drug_se_matrix.txt")

if args.cuda:
    fpt_feature = fpt_feature.to(device)
    mpnn_feature = mpnn_feature.to(device)
    weave_feature = weave_feature.to(device)
    afp_feature = afp_feature.to(device)
    nf_feature = nf_feature.to(device)
    vec_feature = vec_feature.to(device)


#----------------------------------loss functions------------------------------------
loss_m = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40]).cuda())
loss_f = torch.nn.MSELoss()


#----------------------------------data splitting------------------------------------
counter = 1
auc_arr = []
aupr_arr = []
mcc_arr = []
f1_arr = []
macro_f1_arr = []
prec_arr = []
recall_arr = []
ap_arr = []
mr_arr = []
valid_aupr_arr = []
index = 0

train_val_index, test_index = train_test_split(np.arange(len(data_set)), test_size=0.1, random_state=seed)
train_index, valid_index = train_test_split(train_val_index, test_size=0.05, random_state=seed)


train_set = data_set[train_index]
valid_set = data_set[valid_index]
test_set = data_set[test_index]

print("valid",valid_index[-5:])
print("test", test_index[-5:])
print("train shape:", train_set.shape, ", valid shape:", valid_set.shape, ", test shape:", test_set.shape)
logging.info('Begin {:02d}th folder, train_size:{:02d}, train_label:{:.2f}, valid_label:{:.2f}, test_label:{:.2f}'
                .format(counter, len(train_index), np.sum(train_set), np.sum(valid_set), np.sum(test_set)))


#--------------------------------mask preparation------------------------------------
train_mask = np.zeros(args.D_n * args.S_n)#
train_mask[train_index] = 1
target = np.multiply(data_set, train_mask)
matrix = target.reshape(args.D_n, args.S_n)#
logging.info('train_mask:{}, matrix {}'.format(np.sum(train_mask), np.sum(matrix)))


#-------------------------------graph processing------------------------------------
#process and get information
data_set_to_drug_drug=get_twohop(matrix)[0].float().cpu().numpy()
data_set_to_drug_drug=process_matrix(data_set_to_drug_drug,args.kk)
index_weights = get_edge_index_and_weights(data_set_to_drug_drug)
edge_index_to_drug_drug = torch.tensor(index_weights[0], dtype=torch.int64).to(device)
weight_drug=torch.tensor(index_weights[1],dtype=torch.float32).to(device)

train_mask = torch.from_numpy(train_mask.reshape(args.D_n, args.S_n)).to(device)#
target = torch.from_numpy(target).to(device)
drug_se_train, se_drug_train = dse_normalize(device, matrix, D_n=args.D_n, S_n=args.S_n)#

test_mask = np.zeros(args.D_n * args.S_n)#
test_mask[test_index] = 1
test_mask = torch.from_numpy(test_mask.reshape(args.D_n, args.S_n)).to(device)#

valid_mask = np.zeros(args.D_n * args.S_n)#
valid_mask[valid_index] = 1
valid_mask = torch.from_numpy(valid_mask.reshape(args.D_n, args.S_n)).to(device)#


#---------------------------------model setup-------------------------------------
model = DSEModel(args, edge_index_to_drug_drug, weight_drug, llm_feature, w2v_feature)
model.to(device)


#-------------------------------optimizer setup-----------------------------------
param_group_alpha = [model.alpha] 
param_group_rest = [p for p in model.parameters() if p is not model.alpha]  

optimizer = optim.Adam(
    [ {'params': param_group_rest, 'lr': args.lr},     
    {'params': param_group_alpha, 'lr': 0.0001}    ]
    , weight_decay=args.weight_decay)


#--------------------------------training loop-----------------------------------
t_total = time.time()
bad_counter = 0
best_epoch = 0
best_aupr = 0
final_outputs = []
best_valid_metrics = []
best_test_metrics = []
for epoch in range(args.epochs):
    t = time.time() 
    loss, train_mrr, train_aupr, outputs, drug_embeddings , se_embeddings = train(model, optimizer, 
                                                                                  train_mask, target, train_index, train_set,epoch) ###
    model.eval()
    outputs, _, _ = model() 
    valid_metrics = compute_test(valid_set, outputs, valid_mask, valid_index)
    valid_mrr, valid_aupr = valid_metrics[0], valid_metrics[1]
    test_metrics = compute_test(test_set, outputs, test_mask, test_index)
    test_mrr, test_aupr = test_metrics[0], test_metrics[1]


    logging.info('time: {:.4f}s, train_mrr: {:.4f}, train_aupr: {:.4f}, valid_mrr: {:.4f}, valid_aupr: {:.4f}, loss_train: {:.4f}'.format((time.time() - t), train_mrr, train_aupr, valid_mrr, valid_aupr, loss))
    logging.info('folder= {:02d}, Epoch: {:04d}, test_mrr: {:.4f}, test_aupr: {:.4f},  Best_epoch: {:04d}'.format(counter, (epoch+1), test_mrr, test_aupr, (best_epoch+1)))
    if valid_aupr > best_aupr:

        best_aupr = valid_aupr
        best_epoch = epoch
        bad_counter = 0
        final_outputs = outputs
        best_valid_metrics = compute_test(valid_set, outputs, valid_mask, valid_index, True)
        best_test_metrics = compute_test(test_set, outputs, test_mask, test_index, True)

        best_model_path = os.path.join(args.model_path, 'model_params_0410_ps.pth')
        torch.save({
            'epoch': best_epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_aupr': best_aupr,
        }, best_model_path)
        logging.info(f"Best model saved to {best_model_path} (epoch {best_epoch+1}, valid_aupr={best_aupr:.4f})")
    else:
        bad_counter += 1
    if bad_counter >= args.patience:
        break


#-----------------------------final evaluation----------------------------------
logging.info("Optimization Finished!")
logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
logging.info('Loading {:04d}th epoch'.format(best_epoch))

#test
save_all(final_outputs, test_mask, counter)
if best_valid_metrics is None:  
    best_valid_metrics = compute_test(valid_set, final_outputs, valid_mask, valid_index, True)
if best_test_metrics is None:
    best_test_metrics = compute_test(test_set, final_outputs, test_mask, test_index, True)
valid_auc, valid_aupr, valid_prec, valid_recall, valid_mcc, valid_f1, valid_macro_f1, valid_ap, valid_mr = best_valid_metrics
test_auc, test_aupr, prec, recall, mcc, f1, macro_f1, ap, mr = best_test_metrics


logging.info('Best valid results:, folder= {:02d}, valid_auc: {:.4f}, valid_aupr: {:.4f}, valid_mcc: {:.4f}, '
                'valid_f1: {:.4f}, valid_macrof1: {:.4f}, valid_ap: {:.4f}, valid_mr: {:.4f}, valid_prec: {:.4f}, valid_recall: {:.4f}'.format(
    counter, valid_auc, valid_aupr, valid_mcc, valid_f1, valid_macro_f1, valid_ap, valid_mr, valid_prec, valid_recall))
logging.info('Test set results:, folder= {:02d}, test_auc: {:.4f}, test_aupr: {:.4f}, test_mcc: {:.4f}, '
                'test_f1: {:.4f}, test_macrof1: {:.4f}, test_ap: {:.4f}, test_mr: {:.4f}, test_prec: {:.4f}, test_recall: {:.4f}'.format(
    counter, test_auc, test_aupr, mcc, f1, macro_f1, ap, mr, prec, recall))


#-------------------------------save results-----------------------------------
valid_aupr_arr.append(best_aupr)
auc_arr.append(test_auc)
aupr_arr.append(test_aupr)
mcc_arr.append(mcc)
f1_arr.append(f1)
macro_f1_arr.append(macro_f1)
prec_arr.append(prec)
recall_arr.append(recall)
ap_arr.append(ap)
mr_arr.append(mr)
np.savetxt(args.result_path + 'valid_aupr_avg', [counter, np.mean(np.array(valid_aupr_arr))])
np.savetxt(args.result_path + 'auc_avg', [counter, np.mean(np.array(auc_arr))])
np.savetxt(args.result_path + 'aupr_avg', [counter, np.mean(np.array(aupr_arr))])
np.savetxt(args.result_path + 'mcc_avg', [counter, np.mean(np.array(mcc_arr))])
np.savetxt(args.result_path + 'f1_avg', [counter, np.mean(np.array(f1_arr))])
np.savetxt(args.result_path + 'macro_f1_avg', [counter, np.mean(np.array(macro_f1_arr))])
np.savetxt(args.result_path + 'prec_avg', [counter, np.mean(np.array(prec_arr))])
np.savetxt(args.result_path + 'recall_avg', [counter, np.mean(np.array(recall_arr))])
np.savetxt(args.result_path + 'ap_avg', [counter, np.mean(np.array(ap_arr))])
np.savetxt(args.result_path + 'mr_avg', [counter, np.mean(np.array(mr_arr))])
np.savetxt(args.result_path + 'valid_aupr', np.array(valid_aupr_arr))
np.savetxt(args.result_path + 'auc', np.array(auc_arr))
np.savetxt(args.result_path + 'aupr', np.array(aupr_arr))
np.savetxt(args.result_path + 'mcc', np.array(mcc_arr))
np.savetxt(args.result_path + 'f1', np.array(f1_arr))
np.savetxt(args.result_path + 'macro_f1', np.array(macro_f1_arr))
np.savetxt(args.result_path + 'prec', np.array(prec_arr))
np.savetxt(args.result_path + 'recall', np.array(recall_arr))
np.savetxt(args.result_path + 'ap', np.array(ap_arr))
np.savetxt(args.result_path + 'mr', np.array(mr_arr))
