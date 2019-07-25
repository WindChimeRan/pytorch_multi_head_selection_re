import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from pytorch_transformers import *


class Conll_bert_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root

        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.relation_vocab_path = os.path.join(self.data_root,
                                                hyper.relation_vocab)

        self.bio_vocab = {}
        self.word_vocab = Counter()
        self.relation_vocab_set = set()
        self.relation_vocab_dict = None

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')

        self._one_pass_train()

    def _one_pass_train(self):
        # prepare for word_vocab, relation_vocab
        train_path = os.path.join(self.raw_data_root, self.hyper.train)
        self.relation_vocab_set = set()
        sent = []
        dic = {}

        with open(train_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    if sent != []:
                        self.word_vocab.update(sent)
                    sent = []
                    dic = {}
                else:

                    assert len(self._parse_line(line)) == 5

                    num, word, etype, relation, head_list = self._parse_line(
                        line)

                    head_list = eval(head_list)
                    relation = eval(relation)
                    sent.append(word)
                    if relation != ['N']:
                        self.relation_vocab_set.update(relation)
                        for r, h in zip(relation, head_list):
                            dic[word+'~'+r] = h
            self.word_vocab.update(sent)

    def prepare_bert(self, result):

        def align_pos(text, bert_tokens):
            i = 0
            align_list = []
            cur_text_tok = text[i].lower()
            for tok in bert_tokens:
                if tok.startswith('##'):
                    tok = tok[2:]

                # cur_text_tok is the full token
                # tok is just piece

                align_list.append(i) 
                if cur_text_tok == tok and i < len(text) - 1:
                    i += 1
                    cur_text_tok = text[i].lower()
                else:
                    cur_text_tok = cur_text_tok[len(tok):]


            assert len(text) - 1 == max(align_list)
            

            return align_list

        def align_bio(pos, bio):
            result = []
            for i, p in enumerate(pos):
                if 1 != 0 and p == pos[i-1]:
                    if bio[pos[i-1]] == 'B':
                        result.append('I')
                    else:
                        result.append(bio[p])  # add 'I' or 'O'
                else:
                    result.append(bio[p])
            return result

        def head2new(idx, aligned_tokens_pos):
            rev = list(reversed(aligned_tokens_pos))
            length = len(aligned_tokens_pos)
            new_idx = rev.index(idx)
            new_idx = length - new_idx - 1
            return new_idx

        def selection2new(selection, aligned_tokens_pos):
            # WARNING! strong side effect
            for triplet in selection:
                triplet['subject'] = head2new(
                    triplet['subject'], aligned_tokens_pos)
                triplet['object'] = head2new(
                    triplet['object'], aligned_tokens_pos)

        def spolist2new(spo_list, aligned_tokens_pos):
            # WARNING! strong side effect
            for triplet in spo_list:
                triplet['subject'] = self.bert_tokenizer.tokenize(' '.join(triplet['subject']))
                triplet['object'] = self.bert_tokenizer.tokenize(' '.join(triplet['object']))

        text = result['text']
        spo_list = result['spo_list']
        bio = result['bio']
        selection = result['selection']

        bert_tokens = self.bert_tokenizer.tokenize(' '.join(text))
        aligned_tokens_pos = align_pos(text, bert_tokens)

        aligned_bio = align_bio(aligned_tokens_pos, bio)

        assert len(text) - 1 == max(aligned_tokens_pos)
        assert len(aligned_tokens_pos) == len(bert_tokens)
        assert len(aligned_bio) == len(bert_tokens)

        spolist2new(spo_list, aligned_tokens_pos)
        selection2new(selection, aligned_tokens_pos)

        result = {'text': bert_tokens, 'spo_list': spo_list,
                  'bio': aligned_bio, 'selection': selection}
                  
        return result

    def _gen_one_data(self, dataset):
        sent = []
        bio = []
        dic = {}
        selection = []
        selection_dics = []  # temp
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        with open(source, 'r') as s, open(target, 'w') as t:
            for line in s:
                if line.startswith('#'):
                    if sent != []:
                        triplets = self._process_sent(
                            sent, selection_dics, bio)
                        result = {'text': sent, 'spo_list': triplets,
                                  'bio': bio, 'selection': selection_dics}
                        result = self.prepare_bert(result)
                        if len(result['text']) <= self.hyper.max_text_len - 2: # for [CLS] and [SEP]
                            t.write(json.dumps(result))
                            t.write('\n')
                    sent = []
                    bio = []
                    dic = {}  # temp
                    selection_dics = []  # temp
                    selection = []
                else:

                    assert len(self._parse_line(line)) == 5

                    num, word, etype, relation, head_list = self._parse_line(
                        line)

                    head_list = eval(head_list)
                    relation = eval(relation)
                    sent.append(word)
                    bio.append(etype[0])  # only BIO
                    if relation != ['N']:
                        self.relation_vocab_set.update(relation)
                        for r, h in zip(relation, head_list):
                            dic[word+'~'+r] = h
                            selection_dics.append(
                                {'subject': int(num), 'predicate': self.relation_vocab_dict[r], 'object': h})
            if len(sent) <= self.hyper.max_text_len:
                triplets = self._process_sent(sent, selection_dics, bio)
                result = {'text': sent, 'spo_list': triplets,
                          'bio': bio, 'selection': selection_dics}
                result = self.prepare_bert(result)
                t.write(json.dumps(result))

    def gen_all_data(self):
        self._gen_one_data(self.hyper.train)
        self._gen_one_data(self.hyper.dev)

    def gen_bio_vocab(self):
        result = {'<pad>': 3, 'B': 0, 'I': 1, 'O': 2}
        json.dump(result,
                  open(os.path.join(self.data_root, 'bio_vocab.json'), 'w'))

    def gen_relation_vocab(self):
        relation_vocab = {}
        i = 0
        for r in self.relation_vocab_set:
            relation_vocab[r] = i
            i += 1
        relation_vocab['N'] = i
        self.relation_vocab_dict = relation_vocab
        json.dump(relation_vocab,
                  open(self.relation_vocab_path, 'w'),
                  ensure_ascii=True)

    def gen_vocab(self, min_freq: int):
        target = os.path.join(self.data_root, 'word_vocab.json')
        result = {'<pad>': 0}
        i = 1
        for k, v in self.word_vocab.items():
            if v > min_freq:
                result[k] = i
                i += 1
        result['oov'] = i
        json.dump(result, open(target, 'w'), ensure_ascii=False)

    # TODO: fix bug: entity with multiple tokens
    @staticmethod
    def _find_entity(pos, text, sequence_tags):
        entity = []

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
        return entity

    def _process_sent(self, sent: List[str], dic: List[Dict[str, int]], bio: List[str]) -> Set[str]:
        id2relation = {v: k for k, v in self.relation_vocab_dict.items()}
        result = []
        for triplets_id in dic:
            s, p, o = triplets_id['subject'], triplets_id['predicate'], triplets_id['object']
            p = id2relation[p]
            s = self._find_entity(s, sent, bio)
            o = self._find_entity(o, sent, bio)

            result.append({'subject': s, 'predicate': p, 'object': o})
        return result

    @staticmethod
    def _parse_line(line):
        result = line.split()
        if len(result) == 5:
            return result
        else:
            a, b, c = result[:3]
            de = result[3:]
            d, e = [], []
            cur = d
            for t in de:
                cur.append(t)
                if t.endswith(']'):
                    cur = e
            return a, b, c, ''.join(d), ''.join(e)
