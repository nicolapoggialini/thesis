import torch

def make_predictions(model,
                     df,
                     kb_df,
                     batch_size,
                     device):
    
    model = model.eval()

    # Create batches
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]

    preds = []

    with torch.no_grad():

        for batch in batches:

            dialog_contexts = batch['dialog_context'].tolist()
            topic_documents_title = batch['topic_document_title'].tolist()
            topic_documents_text = batch['topic_document_text'].tolist()

            # forward
            generated_responses = model.inference(dialog_contexts,
                                                  topic_documents_title,
                                                  topic_documents_text,
                                                  kb_df,
                                                  device)
            
            preds.extend(generated_responses)
    
    return preds



def make_retrieval_predictions(model,
                               df,
                               kb_df,
                               batch_size, ### this is will be set to 1 ###
                               device):

    model = model.eval()

    # Create batches
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]

    all_span_texts = []

    with torch.no_grad():

        for batch in batches:

            dialog_contexts = batch['dialog_context'].tolist()
            topic_documents_title = batch['topic_document_title'].tolist()
            topic_documents_text = batch['topic_document_text'].tolist()

            # this will output a list of spans
            retrieved_span_texts = model.retrieval_inference(dialog_contexts,
                                                             topic_documents_title,
                                                             topic_documents_text,
                                                             kb_df,
                                                             device)
            
            
            all_span_texts.append(retrieved_span_texts)
    
    return all_span_texts