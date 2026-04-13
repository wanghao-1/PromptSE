import pandas as pd
import numpy as np
from textmodel_unsort import vectorize

# ==================== Load Side Effect Index ====================
index_path = 'data_pt/se_id.xlsx'
index_df = pd.read_excel(index_path)
se_index = index_df["side effect"].tolist()

# ==================== Load Input Data ====================
input_file = 'data_pt/ds_0220_ns5599_nd1020_r20_plus_ca+ex_split.xlsx'
se_df = pd.read_excel(input_file)
side_effects = se_df['side effect'].tolist()

# Map side effects to their original indices
index = [se_index.index(x) for x in side_effects]
print(len(index))

cname = 'summary'

# ==================== Load and Prepare Masks ====================
mask = np.loadtxt('data_pt/drug_se_matrix.txt', dtype=float)
train_df = pd.read_excel('data/drug_eff.xlsx')
train_mask = train_df.iloc[:, 1:].values
test_mask = mask - train_mask

# Compute co‑occurrence matrices
B = mask.T @ mask
train_B = train_mask.T @ train_mask
test_B = test_mask.T @ test_mask

# Set diagonal to -1 to ignore self‑pairs
np.fill_diagonal(B, -1)
np.fill_diagonal(train_B, -1)
np.fill_diagonal(test_B, -1)

print(B.shape)

# Print statistics
print('all:', sum(sum(B == 0)), sum(sum(B > 0)))
print('train:', sum(sum(train_B == 0)), sum(sum(train_B > 0)))
print('test:', sum(sum(test_B == 0)), sum(sum(test_B > 0)))

# ==================== Generate Embeddings for Each Category ====================
se_embeddings_sum = np.zeros((5599, 768))

for cname in ["administration route", "metabolism pathway", "target selectivity", "structural properties"]:
    print(input_file, cname)

    # Vectorize the current category
    np_se_768 = vectorize(input_file,
                          'data_pt/768_se_plus_s2.xlsx',
                          cname)

    # Place embeddings into full 5599‑row array
    se_embeddings_all = np.zeros((5599, np_se_768.shape[1]))
    se_embeddings_all[index] = np_se_768

    # Accumulate for summed embedding
    se_embeddings_sum += se_embeddings_all

    # Save individual category embeddings
    out_file = 'data_vec/768_se_0220_ns5599_nd1020_r20_plus_ca+ex_{}1.xlsx'.format(cname)
    pd.DataFrame(se_embeddings_all).to_excel(out_file, index=False)

    # Compute similarity matrix A = embeddings * embeddings^T
    A = se_embeddings_all @ se_embeddings_all.T

    # Print mean similarities for positive vs zero pairs
    print(cname)
    print('all:', np.mean(A[B == 0]), np.mean(A[B > 0]))
    print('train:', np.mean(A[train_B == 0]), np.mean(A[train_B > 0]))
    print('test:', np.mean(A[test_B == 0]), np.mean(A[test_B > 0]))

# ==================== Save and Evaluate Summed Embeddings ====================
out_file = 'data_vec/768_se_0220_ns5599_nd1020_r20_plus_ca+ex_sum1.xlsx'
pd.DataFrame(se_embeddings_sum).to_excel(out_file, index=False)

A = se_embeddings_sum @ se_embeddings_sum.T
print('sum:')
print(np.mean(A[B == 0]), np.mean(A[B > 0]))
print('train:', np.mean(A[train_B == 0]), np.mean(A[train_B > 0]))
print('test:', np.mean(A[test_B == 0]), np.mean(A[test_B > 0]))
