import torch
import pandas as pd

import sys
sys.path.append('../model')
from amr_functions import *

# knowledge base
kb_df = pd.read_pickle('../../knowledge_base_data/multidoc2dial/kb_amr_df.csv')
new_kb_df = kb_df.copy()

device = torch.device('cuda')

# define embeddings generator
pooling_method = 'sum'
amr_embeddings_generator = AMR_EmbeddingsGenerator(pooling_method, device)

amr_embeddings_list = []

for i in range(len(new_kb_df)):

    row = new_kb_df.iloc[i]
    amr_list = get_amrs_from_file(row['amr_path'])

    # embedding
    amr_emb = amr_embeddings_generator.generate_pooled_embedding(amrs_list=amr_list, train=False)
    amr_embeddings_list.append(amr_emb)


new_kb_df['amr_embedding'] = amr_embeddings_list
 
# path_save = '../../knowledge_base_data/multidoc2dial/kb_amr_emb_df_mean.csv'
# path_save = '../../knowledge_base_data/multidoc2dial/kb_amr_emb_df_max.csv'
path_save = '../../knowledge_base_data/multidoc2dial/kb_amr_emb_df_sum.csv'
new_kb_df.to_pickle(path_save)

# check
df = pd.read_pickle(path_save)
print('DF CHECK')
print(df.shape)
print(df.columns)
