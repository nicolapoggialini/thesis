### functions to define the input strings


# retrieval
def prepare_retr_input(dialog_contexts,
                       topic_documents_title,
                       topic_documents_text,
                       topic_document_bool=True):

    query_strings = []

    for i in range(len(dialog_contexts)):

        concatenated_dialog_context = ' </s> </s> '.join(dialog_contexts[i])

        if topic_document_bool:
            td_title = topic_documents_title[i]
            td_text = topic_documents_text[i]
            query_string =  td_title + ' / ' + td_text + ' </s> </s> ' +  concatenated_dialog_context # + '</s>'

        else:
            query_string = concatenated_dialog_context # + '</s>'

        query_strings.append(query_string)
    
    return query_strings



# re-rank
def prepare_rr_input(snippet_txt, snippet_title,
                     topic_document, topic_document_title,
                     dialog_context, topic_document_bool=True):


    # snippet
    snippet_string = snippet_title + ' / ' + snippet_txt + '</s></s>'

    # dialog
    dialog_string = '</s></s>'.join(dialog_context)

    if topic_document_bool:
        topic_doc_string = topic_document_title + ' / ' + topic_document + '</s></s>'
        input_string = topic_doc_string + snippet_string + dialog_string # + '</s>'

    else:
        input_string = snippet_string + dialog_string

    return input_string



# for amr model
def prepare_rr_text_input(snippet_title,
                          topic_document, topic_document_title,
                          dialog_context, topic_document_bool=True):

    # dialog
    dialog_string = '</s></s>'.join(dialog_context)

    if topic_document_bool:
        topic_doc_string = topic_document_title + ' / ' + topic_document + '</s></s>'
        input_string = topic_doc_string + snippet_title + '</s></s>' + dialog_string # + '</s>'
    
    else:
        input_string = snippet_title + '</s></s>' + dialog_string

    return input_string




# generation
def prepare_gen_input(k2_snippets, dialog_context):

    snippets_list = []

    for i in range(len(k2_snippets)):
        snippet = k2_snippets.iloc[i]
        snippet_txt = snippet['passage']
        snippet_title = snippet['title']
        # snippets_list.append(snippet_title + '</s>' + snippet_txt)
        snippets_list.append(snippet_title + ' / ' + snippet_txt)

    # snippets
    snippets_string = '</s> '.join(snippets_list)

    # dialog
    dialog_string = '</s> '.join(dialog_context)

    # input
    input_string = snippets_string + '</s> ' + dialog_string # + '</s>'

    return input_string