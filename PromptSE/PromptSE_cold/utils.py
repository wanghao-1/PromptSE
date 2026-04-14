import pickle
import numpy as np
import pandas as pd
import random
import torch
import pywt
import torch.nn as nn
import os
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, precision_recall_curve, \
    roc_curve, auc, f1_score, average_precision_score

#------------------------data processing functions----------------------------
def row_normalize(a_matrix):
    #Row-normalize a matrix
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


def standardization(data):
    #Standardize data (z-score normalization)
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    data = (data - mu) / (sigma)
    data[np.isnan(data) | np.isinf(data)] = 0.0
    return data


def dse_normalize(cuda, drug_se, D_n=1020, S_n=5599):
    #Normalize drug-side effect matrix
    se_drug = drug_se.T
    drug_se_normalize = torch.from_numpy(row_normalize(drug_se)).float()
    se_drug_normalize = torch.from_numpy(row_normalize(se_drug)).float()
    if cuda:
        drug_se_normalize = drug_se_normalize.cuda()
        se_drug_normalize = se_drug_normalize.cuda()
    return drug_se_normalize, se_drug_normalize


def gen_adj(A):
    #Generate normalized adjacency matrix
    D = torch.pow(A.sum(1), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def wavelet_encoder(seq):
    #Encode sequence using wavelet transform
    meta_drug = np.array(list(map(lambda x: int(x, 16), seq)))
    ca, cd = pywt.dwt(meta_drug, 'db1')
    drug_feature = ca / np.sum(ca)
    return drug_feature


#------------------------data loading functions----------------------------
def load_data(path="/data/", params_path="/params/", mpnn="mpnn_toxcast.npy", weave="weave_toxcast.npy", afp="afp_toxcast.npy",
              nf="nf_toxcast.npy", fpt="drugs.fpt", llm="0403_3plus.xlsx", w2v="w2v.xlsx", model_params="model_params.pth" ,num=1020):
    print('Loading features...')
    fpt_feature = np.zeros((num, 128))
    drug_file = open(path + fpt, "r")
    index = 0
    for line in drug_file:
        line = line.strip()
        if line == "":
            hex_arr = np.zeros(128)
        else:
            hex_arr = wavelet_encoder(line)
        fpt_feature[index] = hex_arr
        index += 1
    fpt_feature = torch.FloatTensor(standardization(fpt_feature))
    mpnn_feature = torch.FloatTensor(standardization(np.load(path + mpnn)))
    weave_feature = torch.FloatTensor(standardization(np.load(path + weave)))
    afp_feature = torch.FloatTensor(standardization(np.load(path + afp)))
    nf_feature = torch.FloatTensor(standardization(np.load(path + nf)))
    llm_feature = torch.FloatTensor(standardization(pd.read_excel(path + llm).values))
    w2v_feature = torch.FloatTensor(standardization(pd.read_excel(path + w2v).values))
    checkpoint = torch.load(os.path.join((params_path + model_params)), weights_only = False)
    with open(f"{path}mols_vec.pkl", 'rb') as f:
        vec_feature = torch.FloatTensor(standardization(pickle.load(f)))
    drug_file.close()
    return fpt_feature, mpnn_feature, weave_feature, afp_feature, nf_feature, vec_feature, llm_feature, w2v_feature, checkpoint


#------------------------dataset utilities----------------------------
def get_links(path, dataset):
    drug_se = np.loadtxt(path + dataset)
    data_set = drug_se.flatten()
    return data_set


def sample_links(data, seed, pos_count, neg_count):
    random.seed(seed)
    pos_list = []
    neg_list = []
    for data_tmp in data:
        if data_tmp[-1] == 1:
            pos_list.append(data_tmp)
        else:
            neg_list.append(data_tmp)
    pos_data = random.sample(pos_list, pos_count)
    neg_data = random.sample(neg_list, neg_count)
    return np.array(pos_data + neg_data)


#------------------------result saving functions----------------------------
def save_result(outputs, data_set, test_mask, fold, path="/result/",
                D_n=1020, S_n=5599):
    mask = torch.from_numpy(np.where(data_set.reshape(D_n, S_n) == 1, 0, 1)).cuda()
    matrix = torch.mul(torch.mul(outputs, mask), test_mask)
    result = []
    for i in range(D_n):
        for j in range(S_n):
            if matrix[i][j] != 0:
                # print(matrix[i][j])
                result.append([torch.sigmoid(matrix[i][j]).cpu().detach(), i, j])
    result.sort(key=lambda item: item[0], reverse=True)
    np.save(path + 'case_fold' + str(fold), result)


def save_all(eval_outputs, test_mask, fold, path="/result/"):
    np.save(path + 'result' + str(fold), eval_outputs.cpu().detach().numpy())
    np.save(path + 'mask' + str(fold), test_mask.cpu().detach().numpy())


#------------------------evaluation metrics----------------------------
def validation(y_pre, y, flag=False):
    prec, recall, _ = precision_recall_curve(y, y_pre)
    pr_auc = auc(recall, prec)
    mr = mrank(y, y_pre)
    if flag:
        fpr, tpr, threshold = roc_curve(y, y_pre)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y, y_pre)
        y_predict_class = y_pre
        y_predict_class[y_predict_class > 0.5] = 1
        y_predict_class[y_predict_class <= 0.5] = 0
        prec = precision_score(y, y_predict_class)
        recall = recall_score(y, y_predict_class)
        mcc = matthews_corrcoef(y, y_predict_class)
        f1 = f1_score(y, y_predict_class)
        macro_f1 = f1_score(y, y_predict_class, average='macro', zero_division=0)
        return roc_auc, pr_auc, prec, recall, mcc, f1, macro_f1, ap, mr
    return mr, pr_auc, _, _, _, _, _, _, _


def mrank(y, y_pre):
    #Calculate mean reciprocal rank
    index = np.argsort(-y_pre)
    r_label = y[index]
    r_index = np.array(np.where(r_label == 1)) + 1
    reci_sum = np.sum(1 / r_index)
    # reci_rank = np.mean(1 / r_index)
    return reci_sum


def normalize_adj(mx):
    #row-normalize sparse matrix
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def validation1(y_pre, y, flag=False):#for best test zhibiao
    #Enhanced validation with additional metrics
    prec, recall, _ = precision_recall_curve(y, y_pre)
    pr_auc = auc(recall, prec)
    mr = mrank(y, y_pre)
    
    if flag:
        fpr, tpr, _ = roc_curve(y, y_pre)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y, y_pre)
        
        #Calculate metrics at default threshold (0.5)
        y_pred_default = (y_pre > 0.5).astype(int)
        default_prec = precision_score(y, y_pred_default)
        default_recall = recall_score(y, y_pred_default)
        default_mcc = matthews_corrcoef(y, y_pred_default)
        default_f1 = f1_score(y, y_pred_default)
        
        #Find best possible F1 and MCC across all thresholds
        best_f1 = best_mcc = 0
        for threshold in np.linspace(0, 1, 101):
            y_pred = (y_pre > threshold).astype(int)
            best_f1 = max(best_f1, f1_score(y, y_pred))
            best_mcc = max(best_mcc, matthews_corrcoef(y, y_pred))
        
        return roc_auc, pr_auc, default_prec, default_recall, default_mcc, default_f1, ap, mr, best_f1, best_mcc
    
    return mr, pr_auc, _, _, _, _, _, _, _, _