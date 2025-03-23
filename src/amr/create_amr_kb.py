import sys
sys.path.insert(1, '../../spring_env/spring')

import torch
import pandas as pd
import spacy
import pandas as pd

from spring_amr.penman import encode
from spring_amr.utils import instantiate_model_and_tokenizer

from predict_amrs_from_sentences import *



# load spring model

CHECKPOINT = '../../spring_env/spring/AMR3.parsing.pt'
device = torch.device('cuda')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spring_model, tokenizer = instantiate_model_and_tokenizer(
    'facebook/bart-large',
    dropout=0.,
    attention_dropout=0,
    penman_linearization=True,
    use_pointer_tokens=True,
)

spring_model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu')['model'])
spring_model.to(device)
spring_model.eval()


# load kb
kb_df = pd.read_pickle('../../knowledge_base_data/multidoc2dial/kb_embeddings_df.csv')

# load spacy model
nlp = spacy.load('en_core_web_sm')


# convert sentences to amrs
amr_paths = []

for i in range(len(kb_df)):
    row = kb_df.iloc[i]
    passage = row['passage']
    sentences = nlp(passage).sents
    sentences_list = [sent.text for sent in sentences]
    amrs = []

    for s in sentences_list:
        amr = predict_amrs([s], spring_model, tokenizer, device)
        amrs.append(amr[0])

    path_save = '../../knowledge_base_data/multidoc2dial/amr_files/kb_amr_' + str(i) + '.amr'
    amr_paths.append(path_save)

    with open(path_save, 'w') as f:
        for amr in amrs:
            f.write(f"{amr}\n\n")
    
    print(i)


kb_amr_df = kb_df.copy()
kb_amr_df['amr_path'] = amr_paths

df_save_path = '../../knowledge_base_data/multidoc2dial/kb_amr_df.csv'

kb_amr_df.to_pickle(df_save_path)

# check
df = pd.read_pickle(df_save_path)
print('DF CHECK')
print(df.shape)
print(df.columns)
