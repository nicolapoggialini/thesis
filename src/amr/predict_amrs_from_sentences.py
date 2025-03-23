import penman
import torch
from pathlib import Path
#import tempfile

from spring_amr.penman import encode


def read_sentences_in_batches(sentences_list, batch_size=1000, max_length=100):

    data = []
    idx = 0
    for line in sentences_list:
    #for line in Path(sentences_list).read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        n = len(line.split())
        if n > max_length:
            continue
        data.append((idx, line, n))
        idx += 1

    def _iterator(data):

        data = sorted(data, key=lambda x: x[2], reverse=True)

        maxn = 0
        batch = []

        for sample in data:
            idx, line, n = sample
            if n > batch_size:
                if batch:
                    yield batch
                    maxn = 0
                    batch = []
                yield [sample]
            else:
                curr_batch_size = maxn * len(batch)
                cand_batch_size = max(maxn, n) * (len(batch) + 1)

                if 0 < curr_batch_size <= batch_size and cand_batch_size > batch_size:
                    yield batch
                    maxn = 0
                    batch = []
                maxn = max(maxn, n)
                batch.append(sample)

        if batch:
            yield batch

    return _iterator(data), len(data)


def predict_amrs(sentences_list, model, tokenizer, device, only_ok = False):

    amrs = []

    iterator, nsent = read_sentences_in_batches(sentences_list)
    
    for batch in iterator:
        if not batch:
            continue
        ids, sentences, _ = zip(*batch)
        x, _ = tokenizer.batch_encode_sentences(sentences, device=device)
        with torch.no_grad():
            model.amr_mode = True
            out = model.generate(**x, max_length=512, decoder_start_token_id=0, num_beams=1)
        
        bgraphs = []
        for idx, sent, tokk in zip(ids, sentences, out):
            graph, status, (lin, backr) = tokenizer.decode_amr(tokk.tolist(), restore_name_ops=False)
            if only_ok and ('OK' not in str(status)):
                continue
            graph.metadata['status'] = str(status)
            graph.metadata['source'] = '.'
            graph.metadata['nsent'] = str(idx)
            graph.metadata['snt'] = sent
            bgraphs.append((idx, graph))

        for i, g in bgraphs:
            amrs.append(encode(g))
            #print(encode(g))
            #print()


    #exit(0)

    #ids, graphs = zip(*sorted(results, key=lambda x:x[0]))

    #for g in graphs:
        #amrs.append[encode(g)]

    return amrs
