import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd  
from torch_geometric.nn import GCNConv
from torch.nn import init
import random
import os

#------------------------random seed----------------------------
#Random seeds for reproducibility
seed = 10
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


#------------------------basic modules----------------------------
#Basic 1D convolution block
class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, dropout, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


#------------------------inception module----------------------------
#Multi-branch convolution module
class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, out_channels, dropout, kernel_size=1)

        self.branch3x3_1 = BasicConv1d(in_channels, out_channels, dropout, kernel_size=1)
        self.branch3x3_2 = BasicConv1d(out_channels, out_channels, dropout, kernel_size=3, padding=1)

        self.branch3x3dbl_1 = BasicConv1d(in_channels, out_channels, dropout, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(out_channels, out_channels, dropout, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(out_channels, out_channels, dropout, kernel_size=3, padding=1)

        self.branch_pool = BasicConv1d(in_channels, out_channels, dropout, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


#------------------------main model----------------------------
class DSEModel(nn.Module):
    def __init__(self, args, edge_index_to_drug_drug, weight_drug, llm_feature, w2v_feature):
        super(DSEModel, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        #-------------hyper-parameters-----------------
        self.D_n = args.D_n
        self.S_n = args.S_n
        self.args = args
        self.dropout = args.dropout
        self.hid_dim = args.hid_dim  # 64
        self.fpt_dim = args.fpt_dim  # 128
        self.vec_dim = args.vec_dim  # 300
        self.vec_len = args.vec_len  # 100
        self.kge_dim = args.kge_dim  # 400
        self.gnn_dim = args.gnn_dim  # 617
        self.gnn_num = 3
        self.bio_dim = args.bio_dim  # 768
        self.hid_len = 4
        self.row=200#

        #----------------------------features--------------------------------
        self.w2v = w2v_feature.to(device)
        self.LLM = llm_feature.to(device)


        # HiGCN for drugs
        self.edge_index_to_drug_drug=edge_index_to_drug_drug
        self.weight_drug=weight_drug

        self.drug_gnn_layer = GCNConv(512, 512,torch.float32, add_self_loops=True)
        torch.nn.init.eye_(self.drug_gnn_layer.lin.weight)
        if self.drug_gnn_layer.bias is not None:
            init.zeros_(self.drug_gnn_layer.bias)
        self.drug_gnn_layer.to(device)


        # Weights
        self.drug_alpha = nn.Parameter(torch.rand(1))  
        self.se_alpha = nn.Parameter(torch.rand(1))  
        self.drug_sum_alpha=nn.Parameter(torch.tensor(1.0))
        self.se_sum_alpha=nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.Tensor(1, self.hid_dim * 8))
        nn.init.normal_(self.alpha,mean=0,std=0.1)
        self.attn_alpha = nn.Parameter(torch.tensor(0.0))
        

        # ---------------------------Networks--------------------------------
        self.Wn_w2v = nn.Sequential(
            nn.Linear(200, self.hid_dim*8),  
            nn.GELU(),
            nn.Linear(self.hid_dim*8, self.hid_dim*8),  
            nn.BatchNorm1d(self.hid_dim * 8),
            nn.GELU(),
            nn.Linear(self.hid_dim*8, self.hid_dim*8), 
            ).to(device)
        
        self.Wn = nn.Sequential(
                                nn.Linear(self.hid_dim*12, self.hid_dim*8),
                                nn.GELU(),
                                nn.Linear(self.hid_dim*8, self.hid_dim*8),
                                nn.GELU(),
                                nn.Linear(self.hid_dim*8, self.hid_dim*8),
                                nn.BatchNorm1d(self.hid_dim * 8),
                                nn.GELU(),
                                nn.Linear(self.hid_dim*8, self.hid_dim*8),
                                ).to(device)
       
        self.Wns = nn.Sequential(
                                nn.Linear(self.hid_dim*8, self.hid_dim*8),
                                nn.BatchNorm1d(self.hid_dim * 8),
                                nn.GELU(),
                                nn.Linear(self.hid_dim*8, self.hid_dim*8),
                                ).to(device)
       
        for m in self.Wns:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)  
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)


       #------------------------drug initial features----------------------------
        torch.manual_seed(10)
        drug_feature = nn.Parameter(torch.Tensor(self.D_n, 512))
        drug_feature.data.normal_(0, 0.1)
        self.dea = drug_feature


        #------------------------network components---------------------------
        self.softmax = nn.Softmax(dim=2)
        self.conv_ds = nn.Sequential(
            InceptionA(in_channels=self.hid_dim, out_channels=self.hid_dim * 4, dropout=args.dropout))
        self.conv_se = nn.Sequential(
            InceptionA(in_channels=12*self.hid_dim, out_channels=self.hid_dim*2, dropout=args.dropout),)
        self.rnn_mol = nn.GRU(self.vec_dim, self.hid_dim // 2, num_layers=2, bidirectional=True)
        self.rnn_fpt = nn.GRU(self.fpt_dim, self.hid_dim // 2, num_layers=2, bidirectional=True)
        self.rnn_gnn = nn.GRU(self.gnn_dim, self.hid_dim // 2, num_layers=1, bidirectional=True)
        self.self_attn1 = nn.MultiheadAttention(embed_dim=self.hid_dim, num_heads=2, dropout=args.dropout)
        self.self_attn2 = nn.MultiheadAttention(embed_dim=8*self.hid_dim, num_heads=2, dropout=args.dropout)
        self.norm1 = nn.LayerNorm(self.hid_dim)


        #------------------------transformation layers---------------------------
        self.Wd = nn.Sequential(nn.Linear(self.hid_dim * 64, self.hid_dim * 32),
                                nn.BatchNorm1d(self.hid_dim * 32),
                                nn.GELU(),
                                nn.Linear(self.hid_dim * 32, self.hid_dim * 8))
        self.Wa = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim * 4),
                                nn.GELU(),
                                nn.Linear(self.hid_dim * 4, self.hid_dim * 8))
        
        self.gate_fc = nn.Linear(self.hid_dim*8, self.hid_dim*16).to(device)

        

    def forward(self):

        #-------------------------------drug---------------------------------
        # #HiGCN
        drug_random_tensor=self.dea
        self.drug_gnn_layer=self.drug_gnn_layer.to(torch.float32)
        drug_half = self.drug_gnn_layer(drug_random_tensor, self.edge_index_to_drug_drug,self.weight_drug)
        drug_half =torch.relu(drug_half)
        drug_random_tensor = drug_random_tensor + drug_half*self.drug_sum_alpha 
        drug_embeddings = drug_random_tensor 

        # --------------------------side effects ----------------------------
        #LLM feature processing
        se_features = self.LLM 
        se_features = se_features.unsqueeze(0)
        se_conv = self.conv_se(se_features.permute(0, 2, 1))
        se_conv = se_conv.permute(0, 2, 1)
        se_conv = se_conv.squeeze(0)
        se_LLM = self.Wn(self.LLM) + self.Wns(se_conv)

        se_W2V = self.Wn_w2v(self.w2v)
        se_attn = self.self_attn2(se_W2V.unsqueeze(0), se_LLM.unsqueeze(0), se_LLM.unsqueeze(0))[0]
        se_attn = se_attn.squeeze(0)
        se_embeddings = se_W2V + self.attn_alpha * se_attn

        #------------------------output prediction----------------------------
        outputs = torch.mm(drug_embeddings, se_embeddings.t()) 

        return outputs , drug_embeddings , se_embeddings
