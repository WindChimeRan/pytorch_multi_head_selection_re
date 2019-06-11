import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from cached_property import cached_property


class Chinese_selection_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root
        self.schema_path = os.path.join(self.raw_data_root, 'all_50_schemas')

        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(
                'schema file not found, please check your downloaded data!')
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.relation_vocab_path = os.path.join(self.data_root,
                                                hyper.relation_vocab)

    @cached_property
    def relation_vocab(self):
        if os.path.exists(self.relation_vocab_path):
            pass
        else:
            self.gen_relation_vocab()
        return json.load(open(self.relation_vocab_path, 'r'))

    def gen_bio_vocab(self):
        result = {'<pad>': 3, 'B': 0, 'I': 1, 'O': 2}
        json.dump(result,
                  open(os.path.join(self.data_root, 'bio_vocab.json'), 'w'))

    def gen_relation_vocab(self):
        relation_vocab = {}
        i = 0
        for line in open(self.schema_path, 'r'):
            relation = json.loads(line)['predicate']
            if relation not in relation_vocab:
                relation_vocab[relation] = i
                i += 1
        relation_vocab['N'] = i
        json.dump(relation_vocab,
                  open(self.relation_vocab_path, 'w'),
                  ensure_ascii=False)

    def gen_vocab(self, min_freq: int):
        source = os.path.join(self.raw_data_root, self.hyper.train)
        target = os.path.join(self.data_root, 'word_vocab.json')

        cnt = Counter()  # 8180 total
        with open(source, 'r') as s:
            for line in s:
                line = line.strip("\n")
                if not line:
                    return None
                instance = json.loads(line)
                text = list(instance['text'])
                cnt.update(text)
        result = {'<pad>': 0}
        i = 1
        for k, v in cnt.items():
            if v > min_freq:
                result[k] = i
                i += 1
        result['oov'] = i
        json.dump(result, open(target, 'w'), ensure_ascii=False)

    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance['text']

        bio = None
        selection = None

        if 'spo_list' in instance:
            spo_list = instance['spo_list']

            if not self._check_valid(text, spo_list):
                return None
            spo_list = [{
                'predicate': spo['predicate'],
                'object': spo['object'],
                'subject': spo['subject']
            } for spo in spo_list]

            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)

            bio = self.spo_to_bio(text, entities)
            selection = self.spo_to_selection(text, spo_list)

        result = {
            'text': text,
            'spo_list': spo_list,
            'bio': bio,
            'selection': selection
        }
        return json.dumps(result, ensure_ascii=False)

    def _gen_one_data(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        with open(source, 'r') as s, open(target, 'w') as t:
            for line in s:
                newline = self._read_line(line)
                if newline is not None:
                    t.write(newline)
                    t.write('\n')

    def gen_all_data(self):
        self._gen_one_data(self.hyper.train)
        self._gen_one_data(self.hyper.dev)

    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False
        if len(text) > self.hyper.max_text_len:
            return False
        for t in spo_list:
            if t['object'] not in text or t['subject'] not in text:
                return False
        return True

    def spo_to_entities(self, text: str,
                        spo_list: List[Dict[str, str]]) -> List[str]:
        entities = set(t['object'] for t in spo_list) | set(t['subject']
                                                            for t in spo_list)
        return list(entities)

    def spo_to_relations(self, text: str,
                         spo_list: List[Dict[str, str]]) -> List[str]:
        return [t['predicate'] for t in spo_list]

    def spo_to_selection(self, text: str, spo_list: List[Dict[str, str]]
                         ) -> List[Dict[str, int]]:

        selection = []
        for triplet in spo_list:

            object = triplet['object']
            subject = triplet['subject']

            object_pos = text.find(object) + len(object) - 1
            relation_pos = self.relation_vocab[triplet['predicate']]
            subject_pos = text.find(subject) + len(subject) - 1

            selection.append({
                'subject': subject_pos,
                'predicate': relation_pos,
                'object': object_pos
            })

        return selection

    def spo_to_bio(self, text: str, entities: List[str]) -> List[str]:
        bio = ['O'] * len(text)
        for e in entities:
            begin = text.find(e)
            end = begin + len(e) - 1

            assert end <= len(text)

            bio[begin] = 'B'
            for i in range(begin + 1, end + 1):
                bio[i] = 'I'
        return bio
