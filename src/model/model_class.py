import torch
import torch.nn as nn
import torch.nn.functional as F

from my_utils import *



class DialogModel(nn.Module):

    def __init__(self,
                 retrieval_embedding_model, index, k1,  # retrieval
                 rerank_tokenizer, rerank_model, rr_classifier, k2,  # rerank
                 gen_tokenizer, gen_model,  # generation
                 device,
                 no_batch=False):

        super().__init__()

        # retrieval
        self.retrieval_embedding_model = retrieval_embedding_model
        self.index = index
        self.k1 = k1

        # re-rank
        self.rerank_tokenizer = rerank_tokenizer
        self.rerank_model = rerank_model
        self.ffnn = rr_classifier
        self.k2 = k2

        # generation
        self.gen_tokenizer = gen_tokenizer
        self.gen_model = gen_model

        # pseudo-labels computed in batch
        self.no_batch = no_batch


    def forward(self,
                dialog_contexts,
                topic_documents_title,
                topic_documents_text,
                kb_df,
                responses,
                device):

        # inputs for the generation
        gen_inputs = [] # final len will be |batch|

        # each list's final len will be |batch|
        Q_coarse = []
        Q_fine = []
        P_attention = []

        # retrieval
        query_strings = prepare_retr_input(dialog_contexts, topic_documents_title, topic_documents_text)
        query_vectors = self.retrieval_embedding_model.encode(query_strings)
        retrieval_scores, retrieval_indices = self.index.search(query_vectors, k=self.k1)


        # get knowledge snippets for each instance in the batch
        for i in range(len(retrieval_indices)):
            
            index_list = retrieval_indices[i]
            retrieval_scores_np = retrieval_scores[i]
            snippets = kb_df.iloc[index_list].copy() # get the snippets from the knowledge base df

            td_title = topic_documents_title[i]
            td_text = topic_documents_text[i]
            dialog_context = dialog_contexts[i]


            # get ground truth for the instance
            response = responses[i]

            # rr_states_list = []
            rr_inputs = []

            for j in range(len(snippets)):
                snippet = snippets.iloc[j]
                snippet_txt = snippet['passage']
                snippet_title = snippet['title']
                rr_input = prepare_rr_input(snippet_txt, snippet_title,
                                            td_text, td_title, dialog_context)
                rr_inputs.append(rr_input)
            

            # re-rank
            rr_model_inputs = self.rerank_tokenizer(rr_inputs, padding=True, truncation=True, return_tensors="pt").to(device)
            rr_outputs = self.rerank_model(rr_model_inputs['input_ids'], rr_model_inputs['attention_mask'])

            # take the last hidden state of the first token
            last_hidden_states = rr_outputs.last_hidden_state
            reduced_hidden_states = last_hidden_states[:, 0, :].to(device)

            # re-rank scores with ffnn
            rr_scores = self.ffnn(reduced_hidden_states)

            rr_scores_flat = rr_scores.view(-1)


            # sort re-ranker scores
            sorted_rr_scores, sorted_indices = torch.sort(rr_scores_flat, descending=True)

            # get snippets
            k2_indices = sorted_indices[:self.k2].tolist()
            k2_snippets = snippets.iloc[k2_indices].copy()


            ### compute pseudo_labels ###

            # snippets_string_list = k2_snippets['passage'].tolist()
            snippets_string_list = []
            for q in range(len(k2_snippets)):
                snippet = k2_snippets.iloc[q]
                snippet_txt = snippet['passage']
                snippet_title = snippet['title']
                snippets_string_list.append(snippet_title + ' / ' + snippet_txt)


            if self.no_batch:
            
                A_D = torch.zeros(self.k2, requires_grad=False).to(device)

                for k in range(len(snippets_string_list)):
                    snippets_tok = self.gen_tokenizer(snippets_string_list[k], return_tensors="pt")
                    snippets_input_ids = snippets_tok.input_ids.to(device)

                    decoder_input = self.gen_tokenizer(response, return_tensors="pt")
                    decoder_input_ids = decoder_input.input_ids.to(device)

                    with torch.no_grad():
                        out = self.gen_model(input_ids = snippets_input_ids,
                                                decoder_input_ids = decoder_input_ids,
                                                output_attentions = True,
                                                output_pre_attention_weights = True)

                    # out.pre_attentions: tuple of num_layers tensors: each one has shape 1 x num_heads x output_len x input_len
                    pre_attention_scores = out.pre_attentions

                    for layer_outs in pre_attention_scores:
                        A_D[k] = A_D[k] + layer_outs.sum()# .item()

                    X_len = pre_attention_scores[0].shape[3]
                    Y_len = pre_attention_scores[0].shape[2]
                    num_heads = pre_attention_scores[0].shape[1]
                    num_layers = len(pre_attention_scores)

                    A_D[k] = A_D[k] / (num_layers*num_heads*X_len*Y_len)
            
            else:

                # tokenization
                snippets_tok = self.gen_tokenizer(snippets_string_list, padding=True, return_tensors="pt")
                snippets_input_ids = snippets_tok.input_ids.to(device)
                input_attention_mask = snippets_tok.attention_mask.to(device)
    
                repeated_responses = []
                for _ in range(self.k2):
                    repeated_responses.append(response)
    
                decoder_input = self.gen_tokenizer(repeated_responses, return_tensors="pt")
                decoder_input_ids = decoder_input.input_ids.to(device)
                decoder_attention_mask = decoder_input.attention_mask.to(device)

                # repeat response for each retrieved snippet
                # rep_decoder_input_ids = decoder_input_ids.repeat((self.k2, 1)).to(device)
                # rep_decoder_attention_mask = decoder_attention_mask.repeat((self.k2, 1)).to(device)

                with torch.no_grad():
                    out = self.gen_model(input_ids = snippets_input_ids, attention_mask = input_attention_mask, 
                                         decoder_input_ids = decoder_input_ids,
                                         decoder_attention_mask = decoder_attention_mask,
                                         output_attentions = True, # output_attentions = True -> returns cross attentions
                                         output_pre_attention_weights = True) # output_pre_attention_weights -> returns cross attentions before softmax 


                # mean of 'pre-attention scores'

                k2_sums = torch.zeros(self.k2, requires_grad=False).to(device)
    
                # out.pre_attentions: tuple of num_layers tensors: each one has shape k2 x num_heads x output_len x input_len
                # attention_scores = out.cross_attentions
                pre_attention_scores = out.pre_attentions
    
                for layer_outs in pre_attention_scores:
                    for k in range(len(layer_outs)):
                        k2_sums[k] = k2_sums[k] + layer_outs[k].sum()# .item()


                X_len = pre_attention_scores[0].shape[3]
                Y_len = pre_attention_scores[0].shape[2]
                num_heads = pre_attention_scores[0].shape[1]
                num_layers = len(pre_attention_scores)

                A_D = k2_sums / (num_layers*num_heads*X_len*Y_len)


            P_attention_D = F.softmax(A_D, -1) # .tolist()


            k2_retrieval_scores = torch.tensor(retrieval_scores_np[k2_indices], requires_grad=True).to(device)


            Q_coarse_D = F.log_softmax(k2_retrieval_scores, -1)
            Q_fine_D = F.log_softmax(sorted_rr_scores[:self.k2], -1)


            Q_coarse.append(Q_coarse_D)
            Q_fine.append(Q_fine_D)
            P_attention.append(P_attention_D)


            # generation input
            gen_input_string = prepare_gen_input(k2_snippets, dialog_context)
            gen_inputs.append(gen_input_string)

        
        Q_coarse_pt = torch.stack(Q_coarse) # .to(device)
        Q_fine_pt = torch.stack(Q_fine) # .cpu()
        P_attention_pt = torch.stack(P_attention) # .to(device)
        
        distributions_output = {'Q_coarse': Q_coarse_pt, 'Q_fine': Q_fine_pt, 'P_attention': P_attention_pt}

        gen_inputs_tok = self.gen_tokenizer(gen_inputs, padding=True, return_tensors='pt')
        gen_inputs_ids = gen_inputs_tok.input_ids.to(device)
        gen_attention_mask = gen_inputs_tok.attention_mask.to(device)

        decoder_input = self.gen_tokenizer(responses, padding=True, return_tensors="pt")
        decoder_input_ids = decoder_input.input_ids.to(device)
        decoder_attention_mask = decoder_input.attention_mask.to(device)

        generation_out = self.gen_model(input_ids = gen_inputs_ids,
                                        attention_mask = gen_attention_mask,
                                        labels = decoder_input_ids,
                                        decoder_attention_mask = decoder_attention_mask)

        return generation_out, distributions_output


    def inference(self,
                  dialog_contexts,
                  topic_documents_title,
                  topic_documents_text,
                  kb_df,
                  device):

        # inputs for the generation
        gen_inputs = [] # final len will be |batch|

        # retrieval
        query_strings = prepare_retr_input(dialog_contexts, topic_documents_title, topic_documents_text)
        query_vectors = self.retrieval_embedding_model.encode(query_strings)
        retrieval_scores, retrieval_indices = self.index.search(query_vectors, k=self.k1)


        # get knowledge snippets for each instance in the batch
        for i in range(len(retrieval_indices)):
            
            index_list = retrieval_indices[i]
            retrieval_scores_np = retrieval_scores[i]
            snippets = kb_df.iloc[index_list].copy() # get the snippets from the knowledge base df

            td_title = topic_documents_title[i]
            td_text = topic_documents_text[i]
            dialog_context = dialog_contexts[i]

            rr_inputs = []

            for j in range(len(snippets)):
                snippet = snippets.iloc[j]
                snippet_txt = snippet['passage']
                snippet_title = snippet['title']
                rr_input = prepare_rr_input(snippet_txt, snippet_title,
                                            td_text, td_title, dialog_context)
                rr_inputs.append(rr_input)


            # re-rank
            rr_model_inputs = self.rerank_tokenizer(rr_inputs, padding=True, truncation=True, return_tensors="pt").to(device)
            rr_outputs = self.rerank_model(rr_model_inputs['input_ids'], rr_model_inputs['attention_mask'])

            # take the last hidden state of the first token
            last_hidden_states = rr_outputs.last_hidden_state
            reduced_hidden_states = last_hidden_states[:, 0, :].to(device)
            # reduced_hidden_states = outputs.pooler_output.to(device)

            # re-rank scores with ffnn
            rr_scores = self.ffnn(reduced_hidden_states)
            
            rr_scores_flat = rr_scores.view(-1)

            # sort re-ranker scores
            sorted_rr_scores, sorted_indices = torch.sort(rr_scores_flat, descending=True)

            # get snippets
            k2_indices = sorted_indices.tolist()[:self.k2]
            k2_snippets = snippets.iloc[k2_indices].copy()

            # generation input
            gen_input_string = prepare_gen_input(k2_snippets, dialog_context)
            gen_inputs.append(gen_input_string)
        
        gen_inputs_tok = self.gen_tokenizer(gen_inputs, padding=True, return_tensors='pt').to(device)
        # gen_inputs_ids = gen_inputs_tok.input_ids.to(device)
        # gen_attention_mask = gen_inputs_tok.attention_mask.to(device)

        # outputs = model.generate(input_ids)
        outputs = self.gen_model.generate(**gen_inputs_tok)
        generated_responses = self.gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_responses



    def retrieval_inference(self,
                            dialog_contexts,
                            topic_documents_title,
                            topic_documents_text,
                            kb_df,
                            device,
                            re_ranking=False):


        # retrieval
        query_strings = prepare_retr_input(dialog_contexts, topic_documents_title, topic_documents_text)
        query_vectors = self.retrieval_embedding_model.encode(query_strings)
        retrieval_scores, retrieval_indices = self.index.search(query_vectors, k=self.k1)

        # the batch dimension will be set to 1, so retrieval indices contains just one instance
        ind = retrieval_indices[0]
        snippets = kb_df.iloc[ind].copy() # get the snippets from the knowledge base df

        if not re_ranking:
                
                passage_texts = snippets[:self.k2]['passage'].tolist()

        else:

                td_text = topic_documents_text[0]
                td_title = topic_documents_title[0]
                dialog_context = dialog_contexts[0]
        
                rr_inputs = []
        
                for i in range(len(snippets)):
                    snippet = snippets.iloc[i]
                    snippet_txt = snippet['passage']
                    snippet_title = snippet['title']
                    rr_input = prepare_rr_input(snippet_txt, snippet_title,
                                                td_text, td_title, dialog_context)
                    rr_inputs.append(rr_input)
        
        
                # re-rank
                rr_model_inputs = self.rerank_tokenizer(rr_inputs, padding=True, truncation=True, return_tensors="pt").to(device)
                rr_outputs = self.rerank_model(rr_model_inputs['input_ids'], rr_model_inputs['attention_mask'])
        
                # take the last hidden state of the first token
                last_hidden_states = rr_outputs.last_hidden_state
                reduced_hidden_states = last_hidden_states[:, 0, :].to(device)
        
                # re-rank scores with ffnn
                rr_scores = self.ffnn(reduced_hidden_states)
        
                rr_scores_flat = rr_scores.view(-1)
                sorted_rr_scores, sorted_indices = torch.sort(rr_scores_flat, descending=True)
        
                k2_indices = sorted_indices.tolist()[:self.k2]
        
                k2_snippets = snippets.iloc[k2_indices].copy()
        
                passage_texts = k2_snippets['passage'].tolist()

        return passage_texts
