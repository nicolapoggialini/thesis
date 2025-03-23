import wandb
import torch
import evaluate
import pandas as pd
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel # RobertaModel
from my_LongT5 import LongT5ForConditionalGeneration
from transformers.optimization import Adafactor
from torch.optim import Adam
from faiss import read_index
from model_class import DialogModel
from train_functions import train_epoch
from evaluate_functions import make_predictions

import warnings
warnings.filterwarnings("ignore")


# import data

# train
train_df = pd.read_json('../../datasets/multidoc2dial/train.json')

# validation
val_df = pd.read_json('../../datasets/multidoc2dial/val.json')


# knowledge base
path_df = '../../knowledge_base_data/multidoc2dial/kb_embeddings_df.csv'
kb_df = pd.read_pickle(path_df)

# read index
# path_index = '../../knowledge_base_data/multidoc2dial/index_sep.index'
path_index = '../../knowledge_base_data/multidoc2dial/index.index'
faiss_index = read_index(path_index)

# device = torch.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

### rr_classifier / ###
mlp_ffnn = nn.Linear(rr_embedding_dim, 1, device=device)

### rr_classifier: 2 layers ###
# mlp_ffnn = nn.Sequential(
#     nn.Linear(rr_embedding_dim, 300, device=device),
#     nn.ReLU(),
#     nn.Linear(300, 1, device=device),
#     nn.Sigmoid()
# )

### rr_classifier: 3 layers ###
# mlp_ffnn = nn.Sequential(
#     nn.Linear(rr_embedding_dim, 300, device=device),
#     nn.ReLU(),
#     nn.Linear(300, 100, device=device),
#     nn.ReLU(),
#     nn.Linear(100, 1, device=device),
#     nn.Sigmoid()
# )

# whether to compute the pseudo-labels in batch
no_batch = False

model = DialogModel(retrieval_embedding_model, faiss_index, k1,
                    rerank_tokenizer, rerank_model, mlp_ffnn, k2,
                    gen_tokenizer, gen_model,
                    device,
                    no_batch)

model = model.to(device)

loss_fn = nn.KLDivLoss(reduction="sum").to(device)
optimizer = Adam(model.parameters(), lr=5e-5)

# settings
epochs = 10
batch_size = 8

# ground truth
refs = val_df['response'].tolist()


############
# WANDB
##########


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="EkoDoc_training_process",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": "5e-5",
    "LR warmup": "False",
    "architecture": "EkoDoc",
    "dataset": "MultiDoc2Dial",
    "rr_classifier": "1 layer",
    "K1": k1,
    "K2": k2,
    "batch size": batch_size,
    "epochs": epochs,
    "generation model": "LongT5",
    "AMR augmentation": "False",
    "Retrieval": "True",
    "Re-rank": "True"
    }
)


best_rouge = -1


# training loop

for epoch in range(epochs):

    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)

    mean_losses, mean_dpr, mean_rr, mean_gen = train_epoch(model,
                                                           train_df,
                                                           kb_df,
                                                           batch_size,
                                                           loss_fn,
                                                           optimizer,
                                                           device)

    # inference on validation set
    preds = make_predictions(model, val_df, kb_df, batch_size, device)
    
    # compute rouge
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=preds, references=refs)
    rouge_l = results['rougeL']
    
    # log metrics to wandb
    wandb.log({"DPR Loss": mean_dpr, "RR Loss": mean_rr, "Generation Loss": mean_gen, "Loss": mean_losses, "Validation ROUGE-L": rouge_l})

    if rouge_l > best_rouge:

        # save checkpoint
        path_save = '../../trained_models/ekodoc_base/model_state.pt'
        torch.save(model.state_dict(), path_save)

        best_rouge = rouge_l
    

wandb.finish()