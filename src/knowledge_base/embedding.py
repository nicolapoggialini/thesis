import csv
import pandas as pd
from sentence_transformers import SentenceTransformer


def create_kb_df(kb_list, embedding_model, path_save):

    '''
    Converts passages in embeddings and saves the dictionary as a df in a csv file
    '''

    kb_embeddings = {'title': [], 'passage': [], 'input_text': [], 'passage_embedding': []}
    delimiter = '####' 

    # iterate over sections
    for row in kb_list:

        title = row[0]
        passages_list = row[1].split(delimiter)

        for passage in passages_list:

            kb_embeddings['title'].append(title)
            kb_embeddings['passage'].append(passage)

            input_text = title + ' / ' + passage
            kb_embeddings['input_text'].append(input_text)

            embedding = embedding_model.encode(input_text)
            kb_embeddings['passage_embedding'].append(embedding)
                



    # check
    print(len(kb_embeddings['passage_embedding']))

    print(type(kb_embeddings['passage_embedding']))
    print(kb_embeddings['passage_embedding'][0].shape)
    
    # convert to dataframe
    kb_df = pd.DataFrame(kb_embeddings)

    # save
    kb_df.to_pickle(path_save)
    print('DF SAVED')


###################################################

# read data
path = '../../multidoc2dial/data/mdd_kb/mdd-token-all.csv'

all_rows = []

with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        all_rows.append(row)


path_save = '../../knowledge_base_data/multidoc2dial/kb_embeddings_df.csv'

# load sentence transformer
print('LOADING MODEL')
embedding_model = SentenceTransformer('all-mpnet-base-v2')


create_kb_df(all_rows, embedding_model, path_save)

# read df
df = pd.read_pickle(path_save)
print('DF CHECK')
print(df.shape)
print(df.columns)
