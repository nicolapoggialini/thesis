import torch
import evaluate
import pandas as pd
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from my_LongT5 import LongT5ForConditionalGeneration
from faiss import read_index
from model_class import DialogModel

import warnings
warnings.filterwarnings("ignore")


# import data

test_df = pd.read_json('../../datasets/multidoc2dial/test.json')


# knowledge base
path_df = '../../knowledge_base_data/multidoc2dial/kb_embeddings_df.csv'
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

### rr_classifier / ###
mlp_ffnn = nn.Linear(rr_embedding_dim, 1, device=device)


model = DialogModel(retrieval_embedding_model, faiss_index, k1,
                    rerank_tokenizer, rerank_model, mlp_ffnn, k2,
                    gen_tokenizer, gen_model,
                    device)


model = model.to(device)

checkpoint_path = '../../trained_models/ekodoc_base/model_state.pt'
model.load_state_dict(torch.load(checkpoint_path))

instance = test_df.iloc[100]

dialog_contexts = [instance['dialog_context']]
topic_documents_title = [instance['topic_document_title']]
topic_documents_text = [instance['topic_document_text']]

print(type(topic_documents_text[0]))

# forward
generated_responses = model.inference(dialog_contexts,
                                      topic_documents_title,
                                      topic_documents_text,
                                      kb_df,
                                      device)




print(type(generated_responses))
print(generated_responses)

path_save = '../../results/output_examples.txt'

save_string = ''

for c in dialog_contexts[0]:
    save_string = save_string + c + '\n'


save_string = save_string + '\nRESPONSE\n\n' + generated_responses[0]

save_string = save_string + '\nGROUND TRUTH\n\n' + instance['response']

with open(path_save, "w") as f:
    f.write(save_string)