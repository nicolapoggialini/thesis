import wandb
import torch
import evaluate
import pandas as pd
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from my_LongT5 import LongT5ForConditionalGeneration
from torch.optim import Adam
from faiss import read_index
from amr_model_class import AMRDialogModel
from train_functions import train_epoch
from evaluate_functions import make_predictions

import warnings
warnings.filterwarnings("ignore")


# import data

test_df = pd.read_json('../../datasets/multidoc2dial/test.json')


# knowledge base
path_df = '../../knowledge_base_data/multidoc2dial/kb_amr_emb_df_max.csv'
kb_df = pd.read_pickle(path_df)


# read index
path_index = '../../knowledge_base_data/multidoc2dial/index.index'
faiss_index = read_index(path_index)


device = torch.device('cuda')

# load pre-trained models
retrieval_embedding_model = SentenceTransformer('all-mpnet-base-v2', device=device)


rerank_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
rerank_model = AutoModel.from_pretrained("distilroberta-base")

rerank_model.gradient_checkpointing_enable()
rerank_model = rerank_model.to(device)


gen_tokenizer = AutoTokenizer.from_pretrained('google/long-t5-tglobal-large')
gen_model = LongT5ForConditionalGeneration.from_pretrained('google/long-t5-tglobal-large')

gen_model.gradient_checkpointing_enable()
gen_model = gen_model.to(device)

print('Loaded pre-trained models')


# define model
k1 = 25
k2 = 5
rr_embedding_dim = 768
amr_embedding_dim = 768

### rr_classifier: 2 layers ###
mlp_ffnn = nn.Sequential(
    nn.Linear(rr_embedding_dim + amr_embedding_dim, 768, device=device),
    nn.ReLU(),
    nn.Linear(768, 1, device=device)
    # nn.Sigmoid()
)


# dialogue model

model = AMRDialogModel(retrieval_embedding_model, faiss_index, k1,
                    rerank_tokenizer, rerank_model, mlp_ffnn, k2,
                    # amr_embeddings_generator,
                    gen_tokenizer, gen_model,
                    device)


model = model.to(device)

checkpoint_path = '../../trained_models/ekodoc_amr_final/model_state.pt'
model.load_state_dict(torch.load(checkpoint_path))

# settings
batch_size = 8

# ground truth
refs = test_df['response'].tolist()


preds = make_predictions(model, test_df, kb_df, batch_size, device)
    
# compute rouge
rouge = evaluate.load('rouge')
results = rouge.compute(predictions=preds, references=refs)
rouge_l = results['rougeL']

print(rouge_l)

path_save = '../../results/amr_gen_results.txt'

results_string = 'Test set ROUGE-L: ' + str(rouge_l)

with open(path_save, "w") as f:
    f.write(results_string)