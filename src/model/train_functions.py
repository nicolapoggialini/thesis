import torch
import numpy as np


def train_epoch(model,
                train_df,
                kb_df,
                batch_size,
                loss_fn, # KL div
                optimizer,
                device):

  model = model.train()

  losses = []

  dpr_losses = []
  rr_losses = []
  generation_losses = []

  shuffled_df = train_df.sample(frac=1)  # frac=1 shuffles all rows

  # Create batches
  batches = [shuffled_df[i:i+batch_size] for i in range(0, len(shuffled_df), batch_size)]

  for batch in batches:

    dialog_contexts = batch['dialog_context'].tolist()
    topic_documents_title = batch['topic_document_title'].tolist()
    topic_documents_text = batch['topic_document_text'].tolist()
    responses = batch['response'].tolist()

    # forward
    generation_out, distributions_out = model(dialog_contexts,
                                              topic_documents_title,
                                              topic_documents_text,
                                              kb_df,
                                              responses,
                                              device)


    # compute losses

    generation_loss = generation_out.loss

    dpr_loss = loss_fn(distributions_out['Q_coarse'],
                       distributions_out['P_attention'])

    rr_loss = loss_fn(distributions_out['Q_fine'],
                      distributions_out['P_attention'])

    retrieval_loss = dpr_loss + rr_loss

    loss = (retrieval_loss + generation_loss)/len(batch)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    
    dpr_losses.append((dpr_loss/len(batch)).item())
    rr_losses.append((rr_loss/len(batch)).item())
    generation_losses.append((generation_loss/len(batch)).item())
  
    losses.append(loss.item())

  mean_losses = np.mean(losses)
  mean_dpr = np.mean(dpr_losses)
  mean_rr = np.mean(rr_losses)
  mean_gen = np.mean(generation_losses)

  return mean_losses, mean_dpr, mean_rr, mean_gen


#def train_model(model,
#                train_df,
#                kb_df,
#                epochs,
#                batch_size,
#                loss_fn,
#                optimizer,
#                device):
#    
#    history = []
#
#    for epoch in range(epochs):
#
#        print(f'Epoch {epoch + 1}/{epochs}')
#        print('-' * 10)
#
#        # train
#        epoch_loss = train_epoch(model,
#                                 train_df,
#                                 kb_df,
#                                 batch_size,
#                                 loss_fn,
#                                 optimizer,
#                                 device)
#
#        history.append(epoch_loss)
#
#    return history