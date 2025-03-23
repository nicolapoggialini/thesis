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
from evaluate_functions import make_retrieval_predictions


import warnings
warnings.filterwarnings("ignore")


# import data
test_df = pd.read_json('../../datasets/multidoc2dial/test_ground.json')


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


model = AMRDialogModel(retrieval_embedding_model, faiss_index, k1,
                    rerank_tokenizer, rerank_model, mlp_ffnn, k2,
                    # amr_embeddings_generator,
                    gen_tokenizer, gen_model,
                    device)

model = model.to(device)

checkpoint_path = '../../trained_models/ekodoc_amr_final/model_state.pt'
model.load_state_dict(torch.load(checkpoint_path))

# settings
batch_size = 1


# retrieval inference
retrieval_predictions = make_retrieval_predictions(model, test_df, kb_df, batch_size, device)


total = len(retrieval_predictions)
correct_at_1 = 0
correct_at_5 = 0

for i in range(len(retrieval_predictions)):
    row =  test_df.iloc[i]
    ref = row['target_snippet']
    preds = retrieval_predictions[i]

    preds_at_1 = preds[0]
    preds_at_5 = preds[:5]

    if ref in preds_at_1:
        correct_at_1 = correct_at_1 + 1
    
    found = False

    for p in preds_at_5:
        if (ref in p) and (not found):
            correct_at_5 = correct_at_5 + 1
            found = True


print('@1: ' + str(correct_at_1))
print('@5: ' + str(correct_at_5))
print(total) 


path_save = '../../results/retr_results_amr_no_rr.txt'

results_string = '@1: ' + str(correct_at_1) + '\n@5: ' + str(correct_at_5) + '\nTotal: ' + str(total)

with open(path_save, "w") as f:
    f.write(results_string)
