import torch
import torch.nn as nn

import json
import os

from typing import Dict, List, Tuple, Set, Optional

from torchcrf import CRF


class MultiHeadSelection(nn.Module):
    def __init__(self, hyper) -> None:
        super(MultiHeadSelection, self).__init__()

        self.hyper = hyper
        self.data_root = hyper.data_root

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.word_embeddings = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
                                            embedding_dim=hyper.emb_size)

        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
                                         embedding_dim=hyper.rel_emb_size)
        # bio + pad
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                    embedding_dim=hyper.rel_emb_size)

        if hyper.cell_name == 'gru':
            self.encoder = nn.GRU(hyper.emb_size,
                                  hyper.hidden_size,
                                  bidirectional=True,
                                  batch_first=True)
        elif hyper.cell_name == 'lstm':
            self.encoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
        else:
            raise ValueError('cell name should be gru/lstm!')

        if hyper.activation == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)

        self.selection_u = nn.Linear(hyper.hidden_size, hyper.rel_emb_size)
        self.selection_v = nn.Linear(hyper.hidden_size, hyper.rel_emb_size)
        self.selection_uv = nn.Linear(2 * hyper.rel_emb_size,
                                      hyper.rel_emb_size)
        # remove <pad>
        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)

        self.selection_loss = nn.BCEWithLogitsLoss()

        # self.accuracy = F1Selection()

    def inference(self, tokens, span_dict, selection_logits, output):
        span_dict = self.tagger.decode(span_dict)
        output['span_tags'] = span_dict['tags']

        selection_tags = torch.sigmoid(
            selection_logits) > self.config.binary_threshold
        output['selection_triplets'] = self.selection_decode(
            tokens, span_dict['tags'], selection_tags)

        return output

    def forward(self, sample) -> Dict[str, torch.Tensor]:

        # mask = get_text_field_mask(tokens)
        tokens = sample.tokens_id
        selection_gold = sample.selection_id
        bio_gold = sample.bio_id
        length = sample.length

        mask = tokens != self.word_vocab['<pad>']

        embedded = self.word_embeddings(tokens)

        o, h = self.rnn(embedded)

        o = (lambda a: sum(a) / 2)(torch.split(o, self.hidden_size, dim=2))

        emi = self.emission(o)

        output = {}

        crf_loss = 0
        if bio_gold is not None:
            crf_loss = self.tagger(emi, bio_gold, mask=mask)

        # forward multi head selection
        u = self.activation(self.selection_u(o)).unsqueeze(1)
        v = self.activation(self.selection_v(o)).unsqueeze(2)
        u = u + torch.zeros_like(v)
        v = v + torch.zeros_like(u)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))
        selection_logits = torch.einsum('bijh,rh->birj', uv,
                                        self.relation_emb.weight)

        # if inference
        # output = self.inference(tokens, span_dict, selection_logits, output)
        # self.accuracy(output['selection_triplets'], spo_list)
        selection_loss = 0
        if selection_gold is not None:
            selection_loss = self.selection_loss(selection_logits, selection_gold)
        
        output['loss'] = crf_loss + selection_loss

        return output

    def selection_decode(self, tokens, sequence_tags,
                         selection_tags: torch.Tensor
                         ) -> List[List[Dict[str, str]]]:
        # selection_tags[0, 0, 1, 1] = 1
        # temp

        text = [[
            self.vocab.get_token_from_index(token, namespace='tokens')
            for token in instance_token
        ] for instance_token in tokens['tokens'].tolist()]

        def find_entity(pos, text, sequence_tags):
            entity = []

            if len(sequence_tags) < len(text):
                return 'NA'

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))
            return ''.join(entity)

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            b, o, p, s = idx[i].tolist()
            object = find_entity(o, text[b], sequence_tags[b])
            subject = find_entity(s, text[b], sequence_tags[b])
            predicate = self.config.relation_vocab_from_idx[p]
            if object != 'NA' and subject != 'NA':
                triplet = {
                    'object': object,
                    'predicate': predicate,
                    'subject': subject
                }
                result[b].append(triplet)
        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.accuracy.get_metric(reset)
