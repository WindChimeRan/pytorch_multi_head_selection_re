import torch
import torch.nn as nn

import json
import os

from typing import Dict, List, Tuple, Set, Optional


class MultiHeadSelection(nn.Model):
    def __init__(self, hyper) -> None:
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
                                            embedding_dim=50)

        self.encoder = encoder
        self.tagger = tagger(vocab=vocab,
                             encoder=self.encoder,
                             text_field_embedder=self.word_embeddings)
        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
                                         embedding_dim=50)

        self.selection_u = nn.Linear(hyper.hidden_dim, 50)
        self.selection_v = nn.Linear(hyper.hidden_dim, 50)
        self.selection_uv = nn.Linear(100, 50)

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

    def forward(
            self,  # type: ignore
            tokens: Dict[str, torch.LongTensor],
            tags: torch.LongTensor = None,
            selection: torch.FloatTensor = None,
            spo_list: Optional[List[Dict[str, str]]] = None,
            # pylint: disable=unused-argument
            **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(self.word_embeddings(tokens), mask)

        output = {}

        if tags is not None:
            span_dict = self.tagger(tokens, tags)
            span_loss = span_dict['loss']
        else:
            span_dict = self.tagger(tokens)
            span_loss = 0

        # forward multi head selection
        u = torch.tanh(self.selection_u(encoded_text)).unsqueeze(1)
        v = torch.tanh(self.selection_v(encoded_text)).unsqueeze(2)
        u = u + torch.zeros_like(v)
        v = v + torch.zeros_like(u)
        uv = torch.tanh(self.selection_uv(torch.cat((u, v), dim=-1)))
        selection_logits = torch.einsum('bijh,rh->birj', uv,
                                        self.relation_emb.weight)

        # if inference
        output = self.inference(tokens, span_dict, selection_logits, output)
        self.accuracy(output['selection_triplets'], spo_list)

        selection_dict = {}
        if selection is not None:
            selection_loss = self.selection_loss(selection_logits, selection)
            selection_dict['loss'] = selection_loss
            output['loss'] = span_loss + selection_loss

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
